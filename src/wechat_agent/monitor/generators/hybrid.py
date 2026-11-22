"""Hybrid ReplyGenerator combining deterministic rule templates with optional LLM fallback."""

from __future__ import annotations

from wechat_agent.monitor.models import DecisionContext, DecisionResult, GeneratedReply

from .llm_generator import LlmReplyGenerator
from .template import RuleTemplateGenerator


class HybridReplyGenerator:
    """Uses matched YAML rule template when available; otherwise invokes LLM generator."""

    def __init__(
        self,
        template_generator: RuleTemplateGenerator,
        llm_generator: LlmReplyGenerator | None = None,
    ) -> None:
        self.template_generator = template_generator
        self.llm_generator = llm_generator

    def generate(self, ctx: DecisionContext, decision: DecisionResult) -> GeneratedReply | None:
        if decision.matched_rule:
            templated = self.template_generator.generate(ctx, decision)
            if templated is not None:
                return templated
        if self.llm_generator is not None:
            return self.llm_generator.generate(ctx, decision)
        return None

    def rewrite_safe(self, ctx: DecisionContext, decision: DecisionResult) -> GeneratedReply | None:
        """Delegate one-shot neutral safety rewrite to the LLM generator when configured."""
        if self.llm_generator is not None:
            return self.llm_generator.rewrite_safe(ctx, decision)
        return None
