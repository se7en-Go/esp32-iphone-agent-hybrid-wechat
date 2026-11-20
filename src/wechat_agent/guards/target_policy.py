"""Strict target allowlist gate (M1 defaults strictly to 文件传输助手)."""

from __future__ import annotations

from wechat_agent.vision.geometry import canonicalize_chat_title


class PolicyViolationError(RuntimeError):
    """Raised when a target or payload violates local safety policy."""


class TargetPolicy:
    def __init__(self, allowed_targets: tuple[str, ...] | list[str]) -> None:
        normalized = tuple(canonicalize_chat_title(t) for t in allowed_targets if canonicalize_chat_title(t))
        if not normalized:
            raise PolicyViolationError("allowed_targets must not be empty")
        self.allowed_targets = normalized

    def is_allowed(self, target: str) -> bool:
        canon = canonicalize_chat_title(target)
        return bool(canon) and canon in self.allowed_targets

    def assert_allowed(self, target: str) -> str:
        canon = canonicalize_chat_title(target)
        if not canon:
            raise PolicyViolationError("Target conversation title must not be empty")
        if canon not in self.allowed_targets:
            raise PolicyViolationError(
                f"Target {canon!r} is blocked by target policy (allowed: {list(self.allowed_targets)})"
            )
        return canon
