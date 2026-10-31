"""Tests for the stage-10 codegen client (llm.codegen_*) and the Anthropic
adapter's thinking/effort/refusal-fallback request settings and streaming.

No test touches the network: the adapter's HTTP client is a stub.
"""

from __future__ import annotations

import json
import logging
import urllib.error
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
# Adapter: request body, streaming
# --------------------------------------------------------------------------

def _sse(*events: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    for ev in events:
        lines += [f"event: {ev['type']}", "data: " + json.dumps(ev), ""]
    return lines


_STREAM_OK = _sse(
    {"type": "message_start", "message": {"model": "claude-sonnet-5-5", "usage": {"input_tokens": 10}}},
    {"type": "content_block_start", "index": 0, "content_block": {"type": "thinking", "thinking": ""}},
    {"type": "content_block_delta", "index": 0, "delta": {"type": "thinking_delta", "thinking": "hidden"}},
    {"type": "content_block_stop", "index": 0},
    {"type": "content_block_delta", "index": 1, "delta": {"type": "text_delta", "text": "print("}},
    {"type": "content_block_delta", "index": 1, "delta": {"type": "text_delta", "text": "'ok')"}},
    {"type": "content_block_delta", "index": 2, "delta": {"type": "text_delta", "text": "done"}},
    {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 5}},
    {"type": "message_stop"},
)


class _StubResponse:
    status_code = 200

    def __init__(self, lines: list[str] | None = None) -> None:
        self._lines = lines or []

    def raise_for_status(self) -> None:
        return None

    def read(self) -> bytes:
        return b""

    def iter_lines(self):
        yield from self._lines

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


class _StubStream:
    def __init__(self, resp: _StubResponse) -> None:
        self._resp = resp

    def __enter__(self) -> _StubResponse:
        return self._resp

    def __exit__(self, *exc: Any) -> bool:
        return False


class _StubClient:
    def __init__(self, stream_lines: list[str] | None = None) -> None:
        self.calls: list[dict[str, Any]] = []
        self._stream_lines = stream_lines if stream_lines is not None else _STREAM_OK

    def post(self, url: str, headers: dict[str, str], json: dict[str, Any]):
        self.calls.append({"url": url, "headers": headers, "json": json, "mode": "post"})
        return _StubResponse()

    def stream(self, method: str, url: str, headers: dict[str, str], json: dict[str, Any]):
        self.calls.append({"url": url, "headers": headers, "json": json, "mode": "stream"})
        return _StubStream(_StubResponse(self._stream_lines))


def _call(
    adapter: AnthropicAdapter, max_tokens: int = 1024, stream_lines: list[str] | None = None
) -> tuple[dict[str, Any], dict[str, Any]]:
    stub = _StubClient(stream_lines)
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
    assert call["mode"] == "stream"
    assert body["stream"] is True
    assert body["thinking"] == {"type": "adaptive"}
    assert body["output_config"] == {"effort": "high"}
    assert "temperature" not in body
    assert body["max_tokens"] == 64000  # floored: thinking counts toward it
    assert body["fallbacks"] == "default"
    assert call["headers"]["anthropic-beta"] == "server-side-fallback-2026-07-01"
    # Thinking deltas never reach the caller; text blocks are joined.
    assert out["choices"][0]["message"]["content"] == "print('ok')\ndone"
    assert out["choices"][0]["finish_reason"] == "stop"
    assert out["usage"]["prompt_tokens"] == 10
    assert out["usage"]["completion_tokens"] == 5
    assert out["model"] == "claude-sonnet-5-5"


def test_adapter_max_tokens_above_floor_kept() -> None:
    adapter = AnthropicAdapter("https://api.anthropic.com", "k", thinking="adaptive")
    call, _ = _call(adapter, max_tokens=100000)
    assert call["json"]["max_tokens"] == 100000


def test_adapter_stream_error_event_raises_retryable_http_error() -> None:
    adapter = AnthropicAdapter("https://api.anthropic.com", "k", effort="high")
    lines = _sse(
        {"type": "message_start", "message": {"model": "claude-sonnet-5-5", "usage": {}}},
        {"type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}},
    )
    with pytest.raises(urllib.error.HTTPError) as exc:
        _call(adapter, stream_lines=lines)
    assert exc.value.code == 529


def test_adapter_empty_text_at_max_tokens_retries_deeper(caplog: pytest.LogCaptureFixture) -> None:
    """Empty text at max_tokens triggers one doubled-budget retry."""
    adapter = AnthropicAdapter("https://api.anthropic.com", "k", thinking="adaptive", effort="high")
    empty = _sse(
        {"type": "message_start", "message": {"model": "claude-sonnet-5-5", "usage": {"input_tokens": 3}}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "thinking_delta", "thinking": "x"}},
        {"type": "message_delta", "delta": {"stop_reason": "max_tokens"}, "usage": {"output_tokens": 64000}},
    )
    ok = _sse(
        {"type": "message_start", "message": {"model": "claude-sonnet-5-5", "usage": {"input_tokens": 3}}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "```yaml\nfiles: []\n```"}},
        {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 20}},
        {"type": "message_stop"},
    )

    stub = _StubClient()

    def _stream_by_budget(method, url, headers, json):
        lines = empty if json.get("max_tokens", 0) <= 64000 else ok
        stub.calls.append({"url": url, "headers": headers, "json": json, "mode": "stream"})
        return _StubStream(_StubResponse(lines))

    stub.stream = _stream_by_budget  # type: ignore[method-assign]
    adapter._client = stub  # type: ignore[assignment]
    with caplog.at_level(logging.WARNING):
        out = adapter.chat_completion(
            "claude-sonnet-5-5",
            [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}],
            max_tokens=8192,
            temperature=0.7,
        )
    assert "```yaml" in out["choices"][0]["message"]["content"]
    assert out["choices"][0]["finish_reason"] == "stop"
    assert "Retrying once with max_tokens=128000" in caplog.text
    # Second call must use the doubled budget.
    assert stub.calls[1]["json"]["max_tokens"] == 128000


def test_adapter_empty_text_retry_exhausted_keeps_empty(caplog: pytest.LogCaptureFixture) -> None:
    """If even the doubled budget yields no text, return empty (caller handles)."""
    adapter = AnthropicAdapter("https://api.anthropic.com", "k", thinking="adaptive", effort="high")
    lines = _sse(
        {"type": "message_start", "message": {"model": "claude-sonnet-5-5", "usage": {"input_tokens": 3}}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "thinking_delta", "thinking": "x"}},
        {"type": "message_delta", "delta": {"stop_reason": "max_tokens"}, "usage": {"output_tokens": 64000}},
    )
    with caplog.at_level(logging.WARNING):
        _, out = _call(adapter, stream_lines=lines)
    assert out["choices"][0]["message"]["content"] == ""
    assert out["choices"][0]["finish_reason"] == "length"
    assert "produced 0 text chars" in caplog.text


def test_adapter_defaults_unchanged() -> None:
    adapter = AnthropicAdapter("https://api.anthropic.com", "k")
    call, out = _call(adapter, max_tokens=1024)
    body = call["json"]
    assert call["mode"] == "post"
    assert "stream" not in body
    assert body["temperature"] == 0.7
    assert body["max_tokens"] == 1024
    assert "thinking" not in body
    assert "output_config" not in body
    assert "fallbacks" not in body
    assert "anthropic-beta" not in call["headers"]
    assert out["choices"][0]["message"]["content"] == "print('ok')"
