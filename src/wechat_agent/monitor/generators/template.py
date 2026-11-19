"""Deterministic YAML rule template ReplyGenerator."""

from __future__ import annotations

import random

from wechat_agent.monitor.decision.rules import RuleSpec
from wechat_agent.monitor.models import DecisionContext, DecisionResult, GeneratedReply


class RuleTemplateGenerator:
    def __init__(self, rules: list[RuleSpec], *, rng: random.Random | None = None) -> None:
        self._by_id: dict[str, RuleSpec] = {r.rule_id: r for r in rules}
        self._rng = rng or random.Random()

    def generate(self, ctx: DecisionContext, decision: DecisionResult) -> GeneratedReply | None:
        if not decision.matched_rule:
            return None
        rule = self._by_id.get(decision.matched_rule)
        if rule is None or not rule.reply_templates:
            return None
        template = self._rng.choice(rule.reply_templates)
        sender = ctx.message.sender.strip() or "群友"
        rendered = template.replace("{sender}", sender).replace("{group}", ctx.group_title)
        if not rendered.strip():
            return None
        return GeneratedReply(
            text=rendered.strip(),
            generator_name="rule_template",
            rule_id=rule.rule_id,
            reason=f"Rendered template from rule {rule.rule_id!r}",
        )
