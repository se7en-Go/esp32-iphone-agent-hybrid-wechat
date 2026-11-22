"""Stdlib OpenAI-compatible HTTP client (DeepSeek & compatible endpoints, disabled by default)."""

from __future__ import annotations

import json
import math
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from wechat_agent.config import require_bool

MAX_LLM_TIMEOUT_SECONDS = 120.0
DEFAULT_LLM_MAX_RETRIES = 2
DEFAULT_LLM_MAX_RESPONSE_BYTES = 1_048_576
MAX_RETRY_AFTER_SECONDS = 10.0


class LlmClientError(RuntimeError):
    """Raised on LLM configuration, network, HTTP, or response-format failures (never includes secrets)."""


def _parse_retry_after(header_value: str | None) -> float | None:
    if not header_value:
        return None
    raw = header_value.strip()
    try:
        secs = float(raw)
    except ValueError:
        return None
    if not math.isfinite(secs) or secs < 0:
        return None
    return min(secs, MAX_RETRY_AFTER_SECONDS)


def _clamp_llm_timeout(raw: Any, default: float = 30.0) -> float:
    val = default if raw is None else raw
    if isinstance(val, bool) or not isinstance(val, (int, float)):
        raise ValueError(f"llm.timeout_seconds must be a number, got {val!r}")
    f = float(val)
    if not math.isfinite(f):
        raise ValueError(f"llm.timeout_seconds must be finite, got {val!r}")
    return max(1.0, min(MAX_LLM_TIMEOUT_SECONDS, f))


@dataclass(frozen=True)
class LlmConfig:
    enabled: bool = False
    base_url: str = ""
    api_key: str = ""
    model: str = ""
    timeout_seconds: float = 30.0
    temperature: float = 0.2
    max_tokens: int = 2048
    max_retries: int = DEFAULT_LLM_MAX_RETRIES
    max_response_bytes: int = DEFAULT_LLM_MAX_RESPONSE_BYTES

    def __post_init__(self) -> None:
        clamped_timeout = _clamp_llm_timeout(self.timeout_seconds)
        object.__setattr__(self, "timeout_seconds", clamped_timeout)
        object.__setattr__(self, "max_retries", max(0, min(5, int(self.max_retries))))
        object.__setattr__(self, "max_response_bytes", max(4096, int(self.max_response_bytes)))

    @property
    def is_ready(self) -> bool:
        return bool(
            self.enabled
            and self.base_url.strip()
            and self.api_key.strip()
            and self.model.strip()
        )

    @classmethod
    def from_dict_and_env(cls, data: dict[str, Any] | None = None) -> LlmConfig:
        raw = data or {}
        if not isinstance(raw, dict):
            raise ValueError(f"llm config must be a mapping, got {type(raw).__name__}")
        raw_enabled = raw.get("enabled")
        enabled = require_bool(raw_enabled, "llm.enabled", default=False)
        base_url = os.environ.get("LLM_BASE_URL") or str(raw.get("base_url") or "")
        api_key = os.environ.get("LLM_API_KEY") or str(raw.get("api_key") or "")
        model = os.environ.get("LLM_MODEL") or str(raw.get("model") or "")
        if not enabled and base_url and api_key and model and raw_enabled is None:
            # Explicit env triad without explicit enabled=false enables the client cleanly.
            enabled = True
        return cls(
            enabled=enabled,
            base_url=base_url.strip(),
            api_key=api_key.strip(),
            model=model.strip(),
            timeout_seconds=_clamp_llm_timeout(raw.get("timeout_seconds"), default=30.0),
            temperature=float(raw.get("temperature", 0.2)),
            max_tokens=max(32, int(raw.get("max_tokens", 2048))),
            max_retries=int(raw.get("max_retries", DEFAULT_LLM_MAX_RETRIES)),
            max_response_bytes=int(raw.get("max_response_bytes", DEFAULT_LLM_MAX_RESPONSE_BYTES)),
        )


@runtime_checkable
class LlmClient(Protocol):
    def chat_completion(
        self,
        messages: list[dict[str, str]],
        *,
        json_mode: bool = False,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str: ...


class OpenAICompatibleLlmClient:
    """Zero-dependency OpenAI `/chat/completions` client built on `urllib.request`.

    Supports bounded retries on HTTP 429 / 5xx / network timeout (honoring Retry-After),
    strict response byte-size caps, and optional per-tick fail-fast circuit breaking.
    """

    def __init__(
        self,
        config: LlmConfig,
        *,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        self.config = config
        self._sleeper = sleeper
        self._tick_tripped_reason: str | None = None
        parsed = urllib.parse.urlparse(config.base_url.strip())
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            raise LlmClientError("LLM base_url must be a valid HTTP(S) URL")
        clean = config.base_url.strip().rstrip("/")
        self.endpoint = clean if clean.endswith("/chat/completions") else f"{clean}/chat/completions"

    def reset_tick_circuit(self) -> None:
        """Reset the per-tick fail-fast circuit breaker at the beginning of a monitor tick."""
        self._tick_tripped_reason = None

    @property
    def tick_circuit_tripped(self) -> bool:
        return self._tick_tripped_reason is not None

    def _trip_tick_circuit(self, reason: str) -> None:
        self._tick_tripped_reason = reason

    def chat_completion(
        self,
        messages: list[dict[str, str]],
        *,
        json_mode: bool = False,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        if self._tick_tripped_reason is not None:
            raise LlmClientError(f"LLM per-tick circuit breaker open: {self._tick_tripped_reason}")
        if not self.config.api_key or not self.config.model:
            raise LlmClientError("LLM api_key and model must be configured")

        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature if temperature is None else float(temperature),
            "max_tokens": self.config.max_tokens if max_tokens is None else int(max_tokens),
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(
            self.endpoint,
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.config.api_key}",
            },
            method="POST",
        )

        max_bytes = self.config.max_response_bytes
        raw_bytes: bytes | None = None
        last_exc: Exception | None = None

        for attempt in range(self.config.max_retries + 1):
            try:
                with urllib.request.urlopen(req, timeout=self.config.timeout_seconds) as resp:
                    try:
                        raw_bytes = resp.read(max_bytes + 1)
                    except TypeError:
                        raw_bytes = resp.read()
                if len(raw_bytes) > max_bytes:
                    err_msg = f"LLM response body exceeded max_response_bytes ({max_bytes})"
                    self._trip_tick_circuit(err_msg)
                    raise LlmClientError(err_msg)
                break
            except urllib.error.HTTPError as exc:
                retryable = exc.code == 429 or (500 <= exc.code < 600)
                last_exc = exc
                if retryable and attempt < self.config.max_retries:
                    retry_after = _parse_retry_after(
                        exc.headers.get("Retry-After") if exc.headers is not None else None
                    )
                    delay = retry_after if retry_after is not None else min(2.0**attempt, 4.0)
                    if delay > 0:
                        self._sleeper(delay)
                    continue
                err_msg = f"LLM endpoint HTTP {exc.code}: {exc.reason}"
                self._trip_tick_circuit(err_msg)
                raise LlmClientError(err_msg) from exc
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                last_exc = exc
                if attempt < self.config.max_retries:
                    delay = min(2.0**attempt, 4.0)
                    if delay > 0:
                        self._sleeper(delay)
                    continue
                err_msg = f"LLM request failed: {exc}"
                self._trip_tick_circuit(err_msg)
                raise LlmClientError(err_msg) from exc

        if raw_bytes is None:  # pragma: no cover - defensive
            err_msg = f"LLM request failed: {last_exc}"
            self._trip_tick_circuit(err_msg)
            raise LlmClientError(err_msg)

        try:
            data = json.loads(raw_bytes.decode("utf-8"))
            choice = data["choices"][0]
            finish_reason = choice.get("finish_reason")
            content = choice["message"].get("content")
        except (UnicodeDecodeError, json.JSONDecodeError, KeyError, IndexError, TypeError) as exc:
            raise LlmClientError(f"Invalid OpenAI-compatible response payload: {exc}") from exc

        if finish_reason == "length" and (not isinstance(content, str) or not content.strip()):
            raise LlmClientError(
                f"LLM response truncated at finish_reason='length' with empty content "
                f"(reasoning tokens likely exhausted max_tokens={payload['max_tokens']}; increase llm.max_tokens)"
            )
        if not isinstance(content, str):
            raise LlmClientError("LLM response content must be a string")
        return content
