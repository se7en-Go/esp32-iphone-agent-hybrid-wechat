"""Stdlib OpenAI-compatible HTTP client (DeepSeek & compatible endpoints, disabled by default)."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable


class LlmClientError(RuntimeError):
    """Raised on LLM configuration, network, HTTP, or response-format failures (never includes secrets)."""


@dataclass(frozen=True)
class LlmConfig:
    enabled: bool = False
    base_url: str = ""
    api_key: str = ""
    model: str = ""
    timeout_seconds: float = 20.0
    temperature: float = 0.2
    max_tokens: int = 256

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
        enabled = bool(raw.get("enabled", False))
        base_url = os.environ.get("LLM_BASE_URL") or str(raw.get("base_url") or "")
        api_key = os.environ.get("LLM_API_KEY") or str(raw.get("api_key") or "")
        model = os.environ.get("LLM_MODEL") or str(raw.get("model") or "")
        if not enabled and base_url and api_key and model and raw.get("enabled") is None:
            # Explicit env triad without explicit enabled=false enables the client cleanly.
            enabled = True
        return cls(
            enabled=enabled,
            base_url=base_url.strip(),
            api_key=api_key.strip(),
            model=model.strip(),
            timeout_seconds=max(1.0, float(raw.get("timeout_seconds", 20.0))),
            temperature=float(raw.get("temperature", 0.2)),
            max_tokens=max(32, int(raw.get("max_tokens", 256))),
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
    """Zero-dependency OpenAI `/chat/completions` client built on `urllib.request`."""

    def __init__(self, config: LlmConfig) -> None:
        self.config = config
        parsed = urllib.parse.urlparse(config.base_url.strip())
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            raise LlmClientError("LLM base_url must be a valid HTTP(S) URL")
        clean = config.base_url.strip().rstrip("/")
        self.endpoint = clean if clean.endswith("/chat/completions") else f"{clean}/chat/completions"

    def chat_completion(
        self,
        messages: list[dict[str, str]],
        *,
        json_mode: bool = False,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
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
        try:
            with urllib.request.urlopen(req, timeout=self.config.timeout_seconds) as resp:
                raw_bytes = resp.read()
        except urllib.error.HTTPError as exc:
            raise LlmClientError(f"LLM endpoint HTTP {exc.code}: {exc.reason}") from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise LlmClientError(f"LLM request failed: {exc}") from exc

        try:
            data = json.loads(raw_bytes.decode("utf-8"))
            content = data["choices"][0]["message"]["content"]
        except (UnicodeDecodeError, json.JSONDecodeError, KeyError, IndexError, TypeError) as exc:
            raise LlmClientError(f"Invalid OpenAI-compatible response payload: {exc}") from exc

        if not isinstance(content, str):
            raise LlmClientError("LLM response content must be a string")
        return content
