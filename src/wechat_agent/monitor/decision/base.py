"""Base protocol and observe-only stub for Layer 1 DecisionGate."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from wechat_agent.monitor.models import DecisionContext, DecisionResult


@runtime_checkable
class DecisionGate(Protocol):
    def evaluate(self, ctx: DecisionContext) -> DecisionResult: ...


class ObserveOnlyGate:
    """Default passive gate: always skips replying while allowing observation logging."""

    def evaluate(self, ctx: DecisionContext) -> DecisionResult:
        return DecisionResult(
            should_reply=False,
            intent="observe_only",
            confidence=1.0,
            risk_score=0.0,
            matched_rule=None,
            reason="Observe-only mode: passive logging without replying",
            gate_name="observe_only",
        )
