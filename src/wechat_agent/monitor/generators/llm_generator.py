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
            "严禁编造未经群上下文确认的事实；严禁输出验证码、密码、转账/红包引导、隐私号码、医疗/法律/投资承诺。",
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

    def generate(self, ctx: DecisionContext, decision: DecisionResult) -> GeneratedReply | None:
        if self.client is None:
            return None
        try:
            raw = self.client.chat_completion(self.build_messages(ctx, decision), json_mode=False)
        except LlmClientError:
            return None
        text = (raw or "").strip().strip('"').strip("'").strip("「」")
        if not text or text.upper() == "SKIP":
            return None
        if len(text) > self.max_chars:
            text = text[: self.max_chars].rstrip()
        return GeneratedReply(
            text=text,
            generator_name="llm_generator",
            rule_id=decision.matched_rule,
            reason=f"Generated reply for intent {decision.intent!r}",
            raw={"raw_length": len(raw or "")},
        )
