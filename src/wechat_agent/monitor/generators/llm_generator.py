"""OpenAI-compatible contextual ReplyGenerator (Layer 2)."""

from __future__ import annotations

from wechat_agent.monitor.llm_client import LlmClient, LlmClientError
from wechat_agent.monitor.models import DecisionContext, DecisionResult, GeneratedReply


class LlmReplyGenerator:
    """Generates short, natural group replies using context turns, intent, and SKILL constraints."""

    def __init__(
        self,
        client: LlmClient | None,
        *,
        skill_instructions: str = "",
        max_chars: int = 120,
    ) -> None:
        self.client = client
        self.skill_instructions = skill_instructions.strip()
        self.max_chars = max(20, int(max_chars))

    def build_messages(self, ctx: DecisionContext, decision: DecisionResult) -> list[dict[str, str]]:
        sys_parts = [
            "你是微信群里协助答疑的普通群成员（Layer 2 ReplyGenerator）。",
            "要求：语气简短、口语、自然，像群里真人说话；不要用「亲」或机械客服腔；不用 emoji。",
            f"长度不超过 {self.max_chars} 个字，直接输出要发送的一句话；如果把握不足或无需回复，仅输出 SKIP。",
            "严禁编造未经群上下文确认的事实；严禁输出医疗/法律/投资承诺、手机号或身份证号。",
            "硬性用词禁令：回复正文中绝不能出现「红包、转账、打款、汇款、收款码、付款码、支付宝口令、验证码、密码、动态码、校验码、短信码、支付码」等任何字眼，即使是否定提醒（例如“不要私下转账”“切勿发红包”）也严禁出现。",
            "涉及下单或团购指引时，统一使用中性表述，例如「按群内官方团购链接/小程序下单或联系群主确认」。",
        ]
        if self.skill_instructions:
            sys_parts.append(f"附加群聊规范：\n{self.skill_instructions[:1200]}")

        history_lines = [turn.format_line() for turn in ctx.recent_context] or ["(暂无更多前文)"]
        sender = ctx.message.sender.strip() or "群友"
        role_block = (
            f"本号在该群的业务职责与静默边界：\n{ctx.role_prompt.strip()}\n"
            if ctx.role_prompt.strip()
            else ""
        )
        user_prompt = (
            f"群名称：{ctx.group_title}\n"
            + role_block
            + f"已识别意图：{decision.intent}（理由：{decision.reason}）\n"
            + f"最近上下文：\n"
            + "\n".join(history_lines)
            + f"\n\n待回复来信：\n[对方/{sender}]: {ctx.message.text}"
        )
        return [
            {"role": "system", "content": "\n".join(sys_parts)},
            {"role": "user", "content": user_prompt},
        ]

    def build_rewrite_messages(self, ctx: DecisionContext, decision: DecisionResult) -> list[dict[str, str]]:
        messages = self.build_messages(ctx, decision)
        rewrite_note = (
            "\n\n安全改写要求：上一版草稿触发了群聊内容安全词校验，请重新生成一句完全中性的简短回复。"
            "不要提及任何资金、收付款、口令或验证类词汇（哪怕是否定或提醒语气也不得出现），不要包含手机号或绝对化承诺；"
            "涉及下单方式请统一使用「按群内官方团购链接/小程序下单或联系群主确认」这类中性表述。若无法满足，仅输出 SKIP。"
        )
        return [
            messages[0],
            {"role": "user", "content": messages[1]["content"] + rewrite_note},
        ]

    def _normalize_reply_text(self, raw: str | None) -> str | None:
        text = (raw or "").strip().strip('"').strip("'").strip("「」")
        if not text or text.upper() == "SKIP":
            return None
        if len(text) > self.max_chars:
            text = text[: self.max_chars].rstrip()
        return text or None

    def _generate_with_messages(
        self,
        messages: list[dict[str, str]],
        *,
        rule_id: str | None,
        reason: str,
        rewritten: bool = False,
    ) -> GeneratedReply | None:
        if self.client is None:
            return None
        try:
            raw = self.client.chat_completion(messages, json_mode=False)
        except LlmClientError:
            return None
        text = self._normalize_reply_text(raw)
        if text is None:
            return None
        raw_meta: dict[str, object] = {"raw_length": len(raw or "")}
        if rewritten:
            raw_meta["rewritten"] = True
        return GeneratedReply(
            text=text,
            generator_name="llm_generator",
            rule_id=rule_id,
            reason=reason,
            raw=raw_meta,
        )

    def generate(self, ctx: DecisionContext, decision: DecisionResult) -> GeneratedReply | None:
        return self._generate_with_messages(
            self.build_messages(ctx, decision),
            rule_id=decision.matched_rule,
            reason=f"Generated reply for intent {decision.intent!r}",
        )

    def rewrite_safe(self, ctx: DecisionContext, decision: DecisionResult) -> GeneratedReply | None:
        """Regenerate once with a generic neutral-phrasing rewrite prompt when draft fails content guard."""
        return self._generate_with_messages(
            self.build_rewrite_messages(ctx, decision),
            rule_id=decision.matched_rule,
            reason=f"Safe-rewritten reply for intent {decision.intent!r}",
            rewritten=True,
        )
