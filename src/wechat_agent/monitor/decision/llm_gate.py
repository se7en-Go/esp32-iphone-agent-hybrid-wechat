"""LLM contextual intent & risk DecisionGate (strict JSON parsing, fail-closed on errors)."""

from __future__ import annotations

import json
import math
import re

from wechat_agent.monitor.llm_client import LlmClient, LlmClientError
from wechat_agent.monitor.models import DecisionContext, DecisionResult

_CODE_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)

_SYSTEM_PROMPT = """你是微信群聊监控的第一层意图与风险评估门控（DecisionGate）。
请结合最近群聊上下文与当前最新消息，判断机器人是否应该回复，并严格仅输出一个 JSON 对象，不得包含多余文字：
{
  "should_reply": boolean,
  "intent": string,
  "confidence": number,
  "risk_score": number,
  "reason": string
}
要求：
1. confidence 和 risk_score 必须在 0.0 到 1.0 之间。
2. 闲聊刷屏、群友互相对话且未涉及本号职责、信息不足、或存在高风险/敏感话题（验证码、密码、转账/红包、医疗/法律/投资建议、政治冲突）时：令 should_reply=false（或给出高 risk_score >= 0.7）。
3. 只有在上下文明确指向本号或真实需要补充信息且低风险时，才返回 should_reply=true。"""


def _strip_json_fences(raw: str) -> str:
    return _CODE_FENCE_RE.sub("", (raw or "").strip()).strip()


def _fail_closed(intent: str, reason: str, raw_text: str = "") -> DecisionResult:
    return DecisionResult(
        should_reply=False,
        intent=intent,
        confidence=0.0,
        risk_score=1.0,
        matched_rule=None,
        reason=reason,
        gate_name="llm_intent",
        raw={"raw_response": raw_text[:500]} if raw_text else {},
    )


class LlmIntentGate:
    """Contextual intent & risk scoring gate backed by an OpenAI-compatible LlmClient."""

    def __init__(self, client: LlmClient | None) -> None:
        self.client = client

    def build_messages(self, ctx: DecisionContext) -> list[dict[str, str]]:
        history_lines = [turn.format_line() for turn in ctx.recent_context] or ["(暂无更多前文)"]
        sender = ctx.message.sender.strip() or "群友"
        aliases = ", ".join(ctx.self_aliases) if ctx.self_aliases else "(未设置)"
        role_block = (
            f"本号在该群的业务职责与静默边界：\n{ctx.role_prompt.strip()}\n"
            if ctx.role_prompt.strip()
            else ""
        )
        user_prompt = (
            f"群名称：{ctx.group_title}\n"
            f"本号群昵称：{aliases}\n"
            + role_block
            + f"是否明确@本号：{ctx.message.mentioned_self}\n"
            + f"最近上下文（按时间顺序）：\n"
            + "\n".join(history_lines)
            + f"\n\n当前待判定最新来信：\n[对方/{sender}]: {ctx.message.text}"
        )
        return [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

    def parse_response(self, raw_text: str) -> DecisionResult:
        cleaned = _strip_json_fences(raw_text)
        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            return _fail_closed("parse_error_fallback", f"Malformed LLM JSON: {exc}", raw_text)

        if not isinstance(data, dict):
            return _fail_closed("parse_error_fallback", "LLM JSON output is not an object", raw_text)

        should_reply = data.get("should_reply")
        intent = data.get("intent")
        confidence = data.get("confidence")
        risk_score = data.get("risk_score")
        reason = data.get("reason", "")

        if type(should_reply) is not bool:
            return _fail_closed("parse_error_fallback", "LLM JSON 'should_reply' must be boolean", raw_text)
        if not isinstance(intent, str) or not intent.strip():
            return _fail_closed("parse_error_fallback", "LLM JSON 'intent' must be nonempty string", raw_text)
        if (
            isinstance(confidence, bool)
            or not isinstance(confidence, (int, float))
            or isinstance(risk_score, bool)
            or not isinstance(risk_score, (int, float))
        ):
            return _fail_closed("parse_error_fallback", "LLM JSON scores must be numeric", raw_text)

        conf_f = float(confidence)
        risk_f = float(risk_score)
        if not (math.isfinite(conf_f) and math.isfinite(risk_f) and 0.0 <= conf_f <= 1.0 and 0.0 <= risk_f <= 1.0):
            return _fail_closed("parse_error_fallback", "LLM JSON scores out of [0.0, 1.0]", raw_text)

        return DecisionResult(
            should_reply=should_reply,
            intent=intent.strip(),
            confidence=conf_f,
            risk_score=risk_f,
            matched_rule=None,
            reason=str(reason or "").strip(),
            gate_name="llm_intent",
            raw=data,
        )

    def evaluate(self, ctx: DecisionContext) -> DecisionResult:
        if self.client is None:
            return _fail_closed("llm_disabled", "LLM client is disabled or unconfigured")
        try:
            raw_text = self.client.chat_completion(self.build_messages(ctx), json_mode=True)
        except LlmClientError as exc:
            return _fail_closed("llm_error_fallback", f"LLM call failed: {exc}")
        return self.parse_response(raw_text)
