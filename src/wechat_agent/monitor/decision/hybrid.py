"""Hybrid DecisionGate combining deterministic YAML rules, @mention, and contextual LLM intent analysis."""

from __future__ import annotations

from wechat_agent.monitor.models import DecisionContext, DecisionResult

from .llm_gate import LlmIntentGate
from .rules import RuleDecisionGate


class HybridDecisionGate:
    """Two-tier trigger: deterministic YAML rules first, then @mention + contextual LLM intent gate."""

    def __init__(
        self,
        rule_gate: RuleDecisionGate,
        llm_gate: LlmIntentGate | None = None,
        *,
        mention_must_reply: bool = False,
        use_llm_for_mentions: bool = True,
    ) -> None:
        self.rule_gate = rule_gate
        self.llm_gate = llm_gate
        self.mention_must_reply = mention_must_reply
        self.use_llm_for_mentions = use_llm_for_mentions

    def evaluate(self, ctx: DecisionContext) -> DecisionResult:
        # 1. Deterministic YAML keyword/regex rules take top precedence without LLM latency.
        rule_hit = self.rule_gate.match_rule(ctx)
        if rule_hit is not None:
            return rule_hit

        # 2. Direct @mention handling.
        if ctx.message.mentioned_self:
            if self.mention_must_reply:
                return DecisionResult(
                    should_reply=True,
                    intent="direct_mention",
                    confidence=max(self.rule_gate.mention_confidence, 0.90),
                    risk_score=0.05,
                    matched_rule=None,
                    reason="Direct @mention forced reply policy",
                    gate_name="hybrid",
                )
            if self.llm_gate is not None and self.use_llm_for_mentions:
                res = self.llm_gate.evaluate(ctx)
                # If LLM errored/disabled while directly mentioned, fall back to rule mention signal cleanly.
                if res.intent not in {"llm_disabled", "llm_error_fallback", "parse_error_fallback"}:
                    return res
            mention_hit = self.rule_gate.evaluate_mention(ctx)
            if mention_hit is not None:
                return mention_hit

        # 3. Contextual semantic intent analysis via LLM (when enabled).
        if self.llm_gate is not None:
            return self.llm_gate.evaluate(ctx)

        return DecisionResult(
            should_reply=False,
            intent="no_match",
            confidence=0.0,
            risk_score=0.0,
            matched_rule=None,
            reason="No YAML rule matched and LLM intent gate is disabled",
            gate_name="hybrid",
        )
