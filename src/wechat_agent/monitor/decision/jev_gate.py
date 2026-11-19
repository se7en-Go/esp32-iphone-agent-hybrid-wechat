"""Reserved Jev structured decision gate extension slot (Noul / Choice / Score semantics)."""

from __future__ import annotations

from typing import Any

from wechat_agent.monitor.models import DecisionContext, DecisionResult


def to_jev_semantics(decision: DecisionResult) -> dict[str, Any]:
    """Map unified `DecisionResult` into Jev-style `Noul / Choice / Score` audit representation."""
    choice = "REPLY" if decision.should_reply else "SKIP"
    return {
        "Noul": decision.intent,
        "Choice": choice,
        "Score": round(decision.confidence * (1.0 - decision.risk_score), 4),
        "RiskScore": round(decision.risk_score, 4),
        "Reason": decision.reason,
    }


class JevGate:
    """Placeholder extension slot for future Jev structured decision models (fails closed by default)."""

    def __init__(self, *, endpoint: str = "") -> None:
        self.endpoint = endpoint.strip()

    def evaluate(self, ctx: DecisionContext) -> DecisionResult:
        return DecisionResult(
            should_reply=False,
            intent="jev_unconfigured",
            confidence=0.0,
            risk_score=1.0,
            matched_rule=None,
            reason=(
                "JevGate is a reserved extension slot and is not configured yet "
                "(set gate='hybrid' or 'rule' for active decisions)"
            ),
            gate_name="jev",
        )
