"""Anthropic Messages API adapter for ResearchClaw."""

import json
import logging
import urllib.error
from typing import Any

try:
    import httpx

    HAS_HTTPX = True
except ImportError:
    HAS_HTTPX = False

logger = logging.getLogger(__name__)

_JSON_MODE_INSTRUCTION = (
    "You MUST respond with valid JSON only. "
    "Do not include any text outside the JSON object."
)

# Map Anthropic stop_reason → OpenAI finish_reason
_STOP_REASON_MAP = {
    "end_turn": "stop",
    "max_tokens": "length",
    "stop_sequence": "stop",
    "tool_use": "tool_calls",
    "refusal": "content_filter",
}

# Thinking counts toward max_tokens and the model does not see the cap: at
# effort "high" a codegen prompt thought through all of a 16000 budget and
# returned no text (31 Oct 2026). Such calls are streamed (see
# _stream_messages), so the larger budget cannot hit the HTTP read timeout.
_THINKING_MIN_MAX_TOKENS = 64000

# Beta header for the server-side refusal fallback (`"fallbacks": "default"`):
# a declined request is re-run on another model, routed by refusal category.
_REFUSAL_FALLBACK_BETA = "server-side-fallback-2026-07-01"


class AnthropicAdapter:
    """Adapter to call Anthropic Messages API and return OpenAI-compatible response."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        timeout_sec: int = 300,
        thinking: str = "",
        effort: str = "",
        refusal_fallback: bool = False,
    ):
        if not HAS_HTTPX:
            raise ImportError(
                "httpx is required for Anthropic adapter. Install: pip install httpx"
            )
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout_sec = timeout_sec
        # Optional request settings (empty/False = not sent): thinking type
        # (e.g. "adaptive"), output_config.effort and the refusal fallback.
        self.thinking = thinking
        self.effort = effort
        self.refusal_fallback = refusal_fallback
        self._client: httpx.Client | None = None

    def close(self) -> None:
        """BUG-DA8-09: Close the httpx connection pool to prevent fd leaks."""
        if self._client is not None:
            try:
                self._client.close()
            except Exception:  # noqa: BLE001
                pass
            self._client = None

    def _stream_messages(
        self, url: str, headers: dict[str, str], body: dict[str, Any]
    ) -> dict[str, Any]:
        """POST a ``stream: true`` request and fold the SSE events into the
        shape of a non-streaming Messages API response (text blocks only,
        stop_reason, usage, model). Thinking deltas are not kept.

        HTTP errors surface as httpx.HTTPStatusError (body read first, so the
        caller can quote it); an ``error`` event mid-stream raises
        urllib.error.HTTPError (529 when overloaded) for the retry logic.
        """
        assert self._client is not None
        texts: dict[int, list[str]] = {}
        data: dict[str, Any] = {"model": body.get("model"), "usage": {}}
        with self._client.stream("POST", url, headers=headers, json=body) as response:
            if response.status_code >= 400:
                response.read()
                response.raise_for_status()
            for line in response.iter_lines():
                if not line.startswith("data:"):
                    continue
                try:
                    event = json.loads(line[5:].strip())
                except ValueError:
                    continue
                etype = event.get("type")
                if etype == "message_start":
                    message = event.get("message", {})
                    data["model"] = message.get("model", data["model"])
                    data["usage"].update(message.get("usage", {}))
                elif etype == "content_block_delta":
                    delta = event.get("delta", {})
                    if delta.get("type") == "text_delta":
                        texts.setdefault(event.get("index", 0), []).append(
                            delta.get("text", "")
                        )
                elif etype == "message_delta":
                    stop_reason = event.get("delta", {}).get("stop_reason")
                    if stop_reason:
                        data["stop_reason"] = stop_reason
                    data["usage"].update(event.get("usage", {}))
                elif etype == "error":
                    error_info = event.get("error", {})
                    raise urllib.error.HTTPError(
                        url,
                        529 if error_info.get("type") == "overloaded_error" else 500,
                        f"{error_info.get('type', 'api_error')}: {error_info.get('message', '')}",
                        {},
                        None,
                    )
        data["content"] = [
            {"type": "text", "text": "".join(parts)}
            for _, parts in sorted(texts.items())
        ]
        return data

    def chat_completion(
        self,
        model: str,
        messages: list[dict[str, str]],
        max_tokens: int,
        temperature: float,
        json_mode: bool = False,
    ) -> dict[str, Any]:
        """Call Anthropic Messages API and return OpenAI-compatible response.

        Raises urllib.error.HTTPError on API errors so the upstream retry
        logic in LLMClient._call_with_retry works unchanged.
        """
        # Extract and concatenate all system messages
        system_parts: list[str] = []
        user_messages: list[dict[str, str]] = []
        for msg in messages:
            if msg["role"] == "system":
                system_parts.append(msg["content"])
            else:
                user_messages.append(msg)

        system_msg = "\n\n".join(system_parts) if system_parts else None

        # Merge consecutive messages with the same role (Anthropic
        # requires strict user/assistant alternation)
        merged: list[dict[str, str]] = []
        for msg in user_messages:
            if merged and merged[-1]["role"] == msg["role"]:
                merged[-1] = {
                    "role": msg["role"],
                    "content": merged[-1]["content"] + "\n\n" + msg["content"],
                }
            else:
                merged.append(dict(msg))
        user_messages = merged

        # Ensure at least one user message and that it starts with "user"
        if not user_messages:
            user_messages = [{"role": "user", "content": "Hello."}]
        elif user_messages[0]["role"] != "user":
            user_messages.insert(0, {"role": "user", "content": "Continue."})

        # Prepend JSON instruction when json_mode is requested
        if json_mode:
            system_msg = (
                f"{_JSON_MODE_INSTRUCTION}\n\n{system_msg}"
                if system_msg
                else _JSON_MODE_INSTRUCTION
            )

        # Thinking-enabled Claude models require temperature=1.0
        # and do not accept other temperature values.
        _THINKING_MODELS = ("claude-3-7", "claude-4")
        _is_thinking = any(model.startswith(p) for p in _THINKING_MODELS)
        if _is_thinking and temperature != 1.0:
            logger.warning(
                "Overriding temperature=%.2f → 1.0 for thinking model %s "
                "(required by Anthropic API).",
                temperature,
                model,
            )
            temperature = 1.0

        # Build Anthropic request
        body: dict[str, Any] = {
            "model": model,
            "messages": user_messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if system_msg:
            body["system"] = system_msg
        if self.thinking or self.effort:
            # Current models (e.g. claude-sonnet-5-5) reject non-default
            # sampling values; leave temperature out entirely.
            body.pop("temperature")
            body["max_tokens"] = max(max_tokens, _THINKING_MIN_MAX_TOKENS)
            body["stream"] = True
            if self.thinking:
                body["thinking"] = {"type": self.thinking}
            if self.effort:
                body["output_config"] = {"effort": self.effort}
        if self.refusal_fallback:
            body["fallbacks"] = "default"

        url = f"{self.base_url}/v1/messages"
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        if self.refusal_fallback:
            headers["anthropic-beta"] = _REFUSAL_FALLBACK_BETA

        try:
            if self._client is None:
                self._client = httpx.Client(timeout=self.timeout_sec)
            if body.get("stream"):
                data = self._stream_messages(url, headers, body)
            else:
                response = self._client.post(url, headers=headers, json=body)
                response.raise_for_status()
                data = response.json()
        except httpx.HTTPStatusError as exc:
            # Convert to urllib.error.HTTPError for upstream retry logic.
            # Include Anthropic's error body so upstream logs show the
            # actual reason (e.g. "Prefilling not supported").
            detail = ""
            try:
                detail = exc.response.text[:500]
            except Exception:  # noqa: BLE001
                pass
            msg = f"{exc}: {detail}" if detail else str(exc)
            raise urllib.error.HTTPError(
                url,
                exc.response.status_code,
                msg,
                dict(exc.response.headers),
                None,
            ) from exc
        except httpx.HTTPError as exc:
            # Catch all transport errors (ConnectError, TimeoutException,
            # ReadError, RemoteProtocolError, PoolTimeout, etc.)
            raise urllib.error.URLError(str(exc)) from exc

        # Check for Anthropic error responses
        if data.get("type") == "error" or "error" in data:
            error_info = data.get("error", {})
            raise urllib.error.HTTPError(
                url,
                500,
                f"{error_info.get('type', 'api_error')}: {error_info.get('message', str(data))}",
                {},
                None,
            )

        # Extract ALL text content blocks (not just the first)
        content = ""
        if "content" in data and data["content"]:
            text_parts = [
                block.get("text", "")
                for block in data["content"]
                if block.get("type") == "text"
            ]
            content = "\n".join(text_parts)

        # Map Anthropic stop_reason to OpenAI finish_reason
        raw_stop_reason = data.get("stop_reason", "end_turn")
        finish_reason = _STOP_REASON_MAP.get(raw_stop_reason, "stop")
        if raw_stop_reason == "max_tokens" and not content.strip():
            # Extended thinking consumed the whole budget before any text
            # block started (verified 31 Oct: 16k budget -> 15,999 thinking
            # tokens, 0 text chars; 64k budget -> 23k thinking + full
            # blueprint text). Retry once with a doubled budget so codegen
            # does not silently receive an empty response.
            retry_budget = min(body["max_tokens"] * 2, 128000)
            if retry_budget > body["max_tokens"]:
                logger.warning(
                    "%s stopped at max_tokens=%d with no text (thinking "
                    "consumed the budget). Retrying once with max_tokens=%d.",
                    model,
                    body["max_tokens"],
                    retry_budget,
                )
                body["max_tokens"] = retry_budget
                try:
                    if body.get("stream"):
                        data = self._stream_messages(url, headers, body)
                    else:
                        response = self._client.post(url, headers=headers, json=body)
                        response.raise_for_status()
                        data = response.json()
                    if data.get("type") == "error" or "error" in data:
                        raise urllib.error.HTTPError(
                            url, 500,
                            f"{data.get('error', {}).get('type', 'api_error')}: "
                            f"{data.get('error', {}).get('message', str(data))}",
                            {}, None,
                        )
                    text_parts = [
                        block.get("text", "")
                        for block in data.get("content", [])
                        if block.get("type") == "text"
                    ]
                    content = "\n".join(text_parts)
                    raw_stop_reason = data.get("stop_reason", "end_turn")
                    finish_reason = _STOP_REASON_MAP.get(raw_stop_reason, "stop")
                    logger.warning(
                        "Retry with max_tokens=%d produced %d text chars "
                        "(stop=%s).",
                        retry_budget, len(content), raw_stop_reason,
                    )
                except Exception:  # noqa: BLE001
                    logger.exception("Thinking-budget retry failed")
                    raise

        return {
            "choices": [
                {
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": finish_reason,
                }
            ],
            "usage": {
                "prompt_tokens": data.get("usage", {}).get("input_tokens", 0),
                "completion_tokens": data.get("usage", {}).get("output_tokens", 0),
                "total_tokens": (
                    data.get("usage", {}).get("input_tokens", 0)
                    + data.get("usage", {}).get("output_tokens", 0)
                ),
            },
            "model": data.get("model", model),
        }
