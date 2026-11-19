"""Strict target allowlist gate (M1 defaults strictly to 文件传输助手)."""

from __future__ import annotations

from wechat_agent.vision.geometry import normalize_text


class PolicyViolationError(RuntimeError):
    """Raised when a target or payload violates local safety policy."""


class TargetPolicy:
    def __init__(self, allowed_targets: tuple[str, ...] | list[str]) -> None:
        normalized = tuple(normalize_text(t) for t in allowed_targets if normalize_text(t))
        if not normalized:
            raise PolicyViolationError("allowed_targets must not be empty")
        self.allowed_targets = normalized

    def is_allowed(self, target: str) -> bool:
        norm = normalize_text(target)
        return bool(norm) and norm in self.allowed_targets

    def assert_allowed(self, target: str) -> str:
        norm = normalize_text(target)
        if not norm:
            raise PolicyViolationError("Target conversation title must not be empty")
        if norm not in self.allowed_targets:
            raise PolicyViolationError(
                f"Target {norm!r} is blocked by target policy (allowed: {list(self.allowed_targets)})"
            )
        return norm
