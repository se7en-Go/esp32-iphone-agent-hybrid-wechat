"""Layer 1 DecisionGate interfaces, implementations, and factory."""

from __future__ import annotations

from wechat_agent.monitor.llm_client import LlmClient

from .base import DecisionGate, ObserveOnlyGate
from .hybrid import HybridDecisionGate
from .jev_gate import JevGate, to_jev_semantics
from .llm_gate import LlmIntentGate
from .rules import RuleDecisionGate, RuleSpec, load_rules_yaml


def build_decision_gate(
    gate_type: str,
    *,
    rules: list[RuleSpec] | None = None,
    llm_client: LlmClient | None = None,
    mention_must_reply: bool = False,
    mention_confidence: float = 0.90,
) -> DecisionGate:
    kind = (gate_type or "hybrid").strip().lower().replace("-", "_")
    rule_list = rules or []

    if kind == "observe_only":
        return ObserveOnlyGate()
    if kind == "rule":
        return RuleDecisionGate(
            rule_list,
            enable_mention_trigger=True,
            mention_confidence=mention_confidence,
        )
    if kind == "llm":
        return LlmIntentGate(llm_client)
    if kind == "hybrid":
        rule_gate = RuleDecisionGate(
            rule_list,
            enable_mention_trigger=True,
            mention_confidence=mention_confidence,
        )
        llm_gate = LlmIntentGate(llm_client) if llm_client is not None else None
        return HybridDecisionGate(
            rule_gate,
            llm_gate,
            mention_must_reply=mention_must_reply,
        )
    if kind == "jev":
        return JevGate()
    raise ValueError(f"Unsupported DecisionGate type: {gate_type!r}")


__all__ = [
    "DecisionGate",
    "HybridDecisionGate",
    "JevGate",
    "LlmIntentGate",
    "ObserveOnlyGate",
    "RuleDecisionGate",
    "RuleSpec",
    "build_decision_gate",
    "load_rules_yaml",
    "to_jev_semantics",
]
