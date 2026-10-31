"""Tests for the stage-10 codegen client (llm.codegen_*) and the Anthropic
adapter's thinking/effort/refusal-fallback request settings.

No test touches the network: the adapter's HTTP client is a stub.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from researchclaw.config import LlmConfig
from researchclaw.llm.anthropic_adapter import AnthropicAdapter
from researchclaw.llm.client import LLMClient


@dataclass(frozen=True)
class _RCStandIn:
    """Lightweight RCConfig stand-in: a dataclass, because the same-provider
    path rebuilds it with dataclasses.replace() like the real RCConfig."""

    llm: LlmConfig
    metaclaw_bridge: Any = None


def _make_rc_config(provider: str = "openai-compatible", **llm_kwargs):
    return _RCStandIn(llm=LlmConfig(provider=provider, **llm_kwargs))


# --------------------------------------------------------------------------
# Factory: codegen_from_rc_config
# --------------------------------------------------------------------------

def test_codegen_none_when_unset_or_same_as_primary() -> None:
    cfg = _make_rc_config(primary_model="glm", base_url="https://x/v1", api_key="k")
    assert LLMClient.codegen_from_rc_config(cfg) is None
    cfg = _make_rc_config(
        primary_model="glm", base_url="https://x/v1", api_key="k", codegen_model="glm"
    )
    assert LLMClient.codegen_from_rc_config(cfg) is None


def test_codegen_same_provider_keeps_main_endpoint_and_fallbacks() -> None:
    cfg = _make_rc_config(
        primary_model="glm",
        base_url="http://127.0.0.1:8083/v1",
        api_key="k",
        fallback_models=("deepseek",),
        codegen_model="kimi-k3",
    )
    cc = LLMClient.codegen_from_rc_config(cfg)
    assert cc is not None
    assert cc.config.primary_model == "kimi-k3"
    assert cc.config.base_url == "http://127.0.0.1:8083/v1"
    assert list(cc.config.fallback_models) == ["deepseek"]
    assert cc._anthropic is None


def test_codegen_independent_anthropic_provider_from_key_file(tmp_path) -> None:
    key_file = tmp_path / "key.txt"
    key_file.write_text("test-key-value\n", encoding="utf-8")
    cfg = _make_rc_config(
        primary_model="glm",
        base_url="http://127.0.0.1:8083/v1",
        api_key="mainkey",
        fallback_models=("deepseek",),
        codegen_model="claude-sonnet-5-5",
        codegen_provider="anthropic",
        codegen_api_key_file=str(key_file),
        codegen_thinking="adaptive",
        codegen_effort="high",
        codegen_refusal_fallback=True,
    )
    cc = LLMClient.codegen_from_rc_config(cfg)
    assert cc is not None
    assert cc.config.primary_model == "claude-sonnet-5-5"
    assert cc.config.base_url == "https://api.anthropic.com"
    assert cc.config.api_key == "test-key-value"
    # Main-provider fallback models must not be sent to Anthropic.
    assert list(cc.config.fallback_models) == []
    assert cc._anthropic is not None
    assert cc._anthropic.thinking == "adaptive"
    assert cc._anthropic.effort == "high"
    assert cc._anthropic.refusal_fallback is True


def test_codegen_api_key_env_wins_over_file(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    key_file = tmp_path / "key.txt"
    key_file.write_text("from-file", encoding="utf-8")
    monkeypatch.setenv("CODEGEN_KEY", "from-env")
    cfg = _make_rc_config(
        primary_model="glm",
        base_url="https://main/v1",
        api_key="mainkey",
        codegen_model="m2",
        codegen_provider="openai-compatible",
        codegen_base_url="https://cg/v1",
        codegen_api_key_env="CODEGEN_KEY",
        codegen_api_key_file=str(key_file),
    )
    cc = LLMClient.codegen_from_rc_config(cfg)
    assert cc is not None
    assert cc.config.api_key == "from-env"
    assert cc.config.base_url == "https://cg/v1"


def test_codegen_missing_key_file_gives_empty_key(tmp_path) -> None:
    cfg = _make_rc_config(
        primary_model="glm",
        base_url="https://main/v1",
        api_key="mainkey",
        codegen_model="claude-sonnet-5-5",
        codegen_provider="anthropic",
        codegen_api_key_file=str(tmp_path / "absent.txt"),
    )
    cc = LLMClient.codegen_from_rc_config(cfg)
    assert cc is not None
    # Empty key: the executor then keeps the primary client (no placeholder run).
    assert cc.config.api_key == ""


# --------------------------------------------------------------------------
# Adapter request body
# --------------------------------------------------------------------------

class _StubResponse:
    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        return {
            "content": [
                {"type": "thinking", "thinking": ""},
                {"type": "text", "text": "print('ok')"},
            ],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 5},
            "model": "claude-sonnet-5-5",
        }


class _StubClient:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def post(self, url: str, headers: dict[str, str], json: dict[str, Any]):
        self.calls.append({"url": url, "headers": headers, "json": json})
        return _StubResponse()


def _call(adapter: AnthropicAdapter, max_tokens: int = 1024) -> tuple[dict[str, Any], dict[str, Any]]:
    stub = _StubClient()
    adapter._client = stub  # type: ignore[assignment]
    out = adapter.chat_completion(
        "claude-sonnet-5-5",
        [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}],
        max_tokens=max_tokens,
        temperature=0.7,
    )
    return stub.calls[0], out


def test_adapter_thinking_effort_and_fallback_body() -> None:
    adapter = AnthropicAdapter(
        "https://api.anthropic.com", "k", thinking="adaptive", effort="high",
        refusal_fallback=True,
    )
    call, out = _call(adapter, max_tokens=1024)
    body = call["json"]
    assert body["thinking"] == {"type": "adaptive"}
    assert body["output_config"] == {"effort": "high"}
    assert "temperature" not in body
    assert body["max_tokens"] == 16000  # floored: thinking counts toward it
    assert body["fallbacks"] == "default"
    assert call["headers"]["anthropic-beta"] == "server-side-fallback-2026-07-01"
    # Thinking blocks never reach the caller; only text does.
    assert out["choices"][0]["message"]["content"] == "print('ok')"


def test_adapter_large_max_tokens_kept() -> None:
    adapter = AnthropicAdapter("https://api.anthropic.com", "k", thinking="adaptive")
    call, _ = _call(adapter, max_tokens=32768)
    assert call["json"]["max_tokens"] == 32768


def test_adapter_defaults_unchanged() -> None:
    adapter = AnthropicAdapter("https://api.anthropic.com", "k")
    call, _ = _call(adapter, max_tokens=1024)
    body = call["json"]
    assert body["temperature"] == 0.7
    assert body["max_tokens"] == 1024
    assert "thinking" not in body
    assert "output_config" not in body
    assert "fallbacks" not in body
    assert "anthropic-beta" not in call["headers"]
