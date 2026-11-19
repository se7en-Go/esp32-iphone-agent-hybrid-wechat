"""Offline unit tests for M2 two-stage DecisionGate, ReplyGenerator, LlmClient, and JevGate slot."""

from __future__ import annotations

import io
import json
import urllib.error
import urllib.request

from wechat_agent.config import repo_root
from wechat_agent.monitor.decision import (
    HybridDecisionGate,
    JevGate,
    LlmIntentGate,
    RuleDecisionGate,
    RuleSpec,
    build_decision_gate,
    load_rules_yaml,
    to_jev_semantics,
)
from wechat_agent.monitor.generators import (
    HybridReplyGenerator,
    LlmReplyGenerator,
    RuleTemplateGenerator,
)
from wechat_agent.monitor.llm_client import LlmClientError, LlmConfig, OpenAICompatibleLlmClient
from wechat_agent.monitor.models import ContextTurn, DecisionContext, ExtractedMessage, MessageRole


class _StubLlmClient:
    def __init__(self, responses: list[str | Exception]) -> None:
        self.responses = list(responses)
        self.calls: list[dict] = []

    def chat_completion(
        self,
        messages: list[dict[str, str]],
        *,
        json_mode: bool = False,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        self.calls.append({"messages": messages, "json_mode": json_mode})
        nxt = self.responses.pop(0)
        if isinstance(nxt, Exception):
            raise nxt
        return nxt


def _ctx(
    text: str,
    *,
    mentioned: bool = False,
    group: str = "测试群",
    role_prompt: str = "",
) -> DecisionContext:
    return DecisionContext(
        group_title=group,
        message=ExtractedMessage(
            group_title=group,
            role=MessageRole.INCOMING,
            sender="张三",
            text=text,
            mentioned_self=mentioned,
            fingerprint="fp-x",
        ),
        recent_context=(
            ContextTurn(MessageRole.INCOMING, "李四", "今天门店开了吗？"),
            ContextTurn(MessageRole.SELF, "", "今天正常开到晚上八点"),
        ),
        self_aliases=("运营助手",),
        role_prompt=role_prompt,
    )


def test_rules_example_yaml_loads_and_rule_gate_matches() -> None:
    example_rules = load_rules_yaml(repo_root() / "config" / "rules.example.yaml")
    assert len(example_rules) >= 2

    rules = [
        RuleSpec(
            rule_id="hours",
            priority=100,
            keywords=("营业时间",),
            exclude_keywords=("退款",),
            require_mention=True,
            intent="ask_hours",
            confidence=0.95,
            risk_score=0.05,
            reply_templates=("@{sender} 早上9点到晚上8点哈",),
        ),
        RuleSpec(
            rule_id="addr",
            priority=80,
            groups=("VIP群",),
            regex=r"地址在哪",
            intent="ask_addr",
            reply_templates=("@{sender} 在A座3楼",),
        ),
    ]
    gate = RuleDecisionGate(rules)
    assert gate.evaluate(_ctx("营业时间几点", mentioned=False)).should_reply is False
    assert gate.evaluate(_ctx("@运营助手 营业时间几点，顺便问退款", mentioned=True)).matched_rule is None
    hit = gate.evaluate(_ctx("@运营助手 请问营业时间", mentioned=True))
    assert hit.matched_rule == "hours"
    assert hit.confidence == 0.95

    # Group scope isolation
    assert gate.evaluate(_ctx("地址在哪", group="测试群")).should_reply is False
    assert gate.evaluate(_ctx("地址在哪", group="VIP群")).matched_rule == "addr"


def test_llm_intent_gate_strict_json_and_fail_closed() -> None:
    # Valid fenced JSON with context passed through
    stub = _StubLlmClient(
        [
            '```json\n{"should_reply": true, "intent": "followup_question", "confidence": 0.84, "risk_score": 0.12, "reason": "asking schedule"}\n```',
            "not valid json",
            json.dumps({"should_reply": "true", "intent": "x", "confidence": 0.9, "risk_score": 0.1}),
            json.dumps({"should_reply": True, "intent": "x", "confidence": 1.5, "risk_score": 0.1}),
            LlmClientError("timeout"),
        ]
    )
    gate = LlmIntentGate(stub)
    res_ok = gate.evaluate(_ctx("明天也到八点吗？"))
    assert res_ok.should_reply is True
    assert res_ok.intent == "followup_question"
    assert res_ok.passes_thresholds(0.70, 0.30) is True
    assert "[对方/李四]: 今天门店开了吗？" in stub.calls[0]["messages"][1]["content"]
    assert stub.calls[0]["json_mode"] is True

    # Malformed, wrong-type bool, out-of-range score, and network error MUST all fail closed!
    for _ in range(4):
        bad = gate.evaluate(_ctx("你好"))
        assert bad.should_reply is False
        assert bad.risk_score == 1.0


def test_hybrid_gate_and_generator_and_jev_slot() -> None:
    rules = [
        RuleSpec(
            rule_id="hours",
            keywords=("营业时间",),
            require_mention=False,
            intent="ask_hours",
            reply_templates=("@{sender} 9点到20点",),
        )
    ]
    llm = _StubLlmClient(
        [
            json.dumps(
                {
                    "should_reply": True,
                    "intent": "contextual_help",
                    "confidence": 0.82,
                    "risk_score": 0.10,
                    "reason": "relevant followup",
                }
            ),
            "明天也是九点到八点哈",
        ]
    )
    gate = build_decision_gate("hybrid", rules=rules, llm_client=llm)
    assert isinstance(gate, HybridDecisionGate)
    gen = HybridReplyGenerator(RuleTemplateGenerator(rules), LlmReplyGenerator(llm))

    # 1. Deterministic YAML rule hits WITHOUT calling LLM!
    c1 = _ctx("请问营业时间是什么")
    d1 = gate.evaluate(c1)
    assert d1.matched_rule == "hours"
    assert len(llm.calls) == 0
    r1 = gen.generate(c1, d1)
    assert r1 is not None and r1.text == "@张三 9点到20点" and r1.generator_name == "rule_template"
    assert len(llm.calls) == 0

    # 2. Contextual non-rule message invokes LlmIntentGate then LlmReplyGenerator
    c2 = _ctx("那周末呢？")
    d2 = gate.evaluate(c2)
    assert d2.should_reply is True and d2.matched_rule is None
    r2 = gen.generate(c2, d2)
    assert r2 is not None and r2.text == "明天也是九点到八点哈" and r2.generator_name == "llm_generator"

    # 3. JevGate reserved extension slot fails closed + semantic mapping
    jev = build_decision_gate("jev")
    assert isinstance(jev, JevGate)
    dj = jev.evaluate(c2)
    assert dj.should_reply is False and dj.intent == "jev_unconfigured"
    sem = to_jev_semantics(d2)
    assert sem["Noul"] == "contextual_help" and sem["Choice"] == "REPLY" and sem["Score"] > 0


def test_openai_compatible_client_http_and_secret_redaction(monkeypatch) -> None:
    captured: dict = {}

    class _FakeResp:
        def read(self) -> bytes:
            return json.dumps({"choices": [{"message": {"content": "{\"ok\": true}"}}]}).encode("utf-8")

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return None

    def fake_urlopen(req, timeout=20.0):
        captured["url"] = req.full_url
        captured["auth"] = req.headers.get("Authorization")
        captured["body"] = json.loads(req.data.decode("utf-8"))
        return _FakeResp()

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    cfg = LlmConfig(
        enabled=True,
        base_url="https://api.deepseek.com/v1",
        api_key="sk-super-secret-xyz",
        model="deepseek-chat",
    )
    client = OpenAICompatibleLlmClient(cfg)
    out = client.chat_completion([{"role": "user", "content": "hi"}], json_mode=True)
    assert out == '{"ok": true}'
    assert captured["url"] == "https://api.deepseek.com/v1/chat/completions"
    assert captured["auth"] == "Bearer sk-super-secret-xyz"
    assert captured["body"]["response_format"] == {"type": "json_object"}

    def fake_urlopen_err(req, timeout=20.0):
        raise urllib.error.HTTPError(req.full_url, 500, "Server Error", {}, io.BytesIO(b""))

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen_err)
    try:
        client.chat_completion([{"role": "user", "content": "hi"}])
    except LlmClientError as exc:
        assert "sk-super-secret-xyz" not in str(exc)
    else:
        raise AssertionError("Expected LlmClientError")


def test_role_prompt_injection_and_omission_in_llm_prompts() -> None:
    gate = LlmIntentGate(None)
    gen = LlmReplyGenerator(None)

    duty = "仅解答营业时间与门店活动；退款投诉和闲聊请保持静默。"
    with_role = _ctx("请问周末几点开", role_prompt=duty)
    gate_msgs = gate.build_messages(with_role)
    assert duty in gate_msgs[1]["content"]
    assert "业务职责与静默边界" in gate_msgs[1]["content"]

    from wechat_agent.monitor.models import DecisionResult

    dec = DecisionResult(
        should_reply=True,
        intent="ask_hours",
        confidence=0.9,
        risk_score=0.1,
        reason="matched duty",
    )
    gen_msgs = gen.build_messages(with_role, dec)
    assert duty in gen_msgs[1]["content"]
    assert "业务职责与静默边界" in gen_msgs[1]["content"]

    no_role = _ctx("请问周末几点开", role_prompt="")
    assert "业务职责与静默边界" not in gate.build_messages(no_role)[1]["content"]
    assert "业务职责与静默边界" not in gen.build_messages(no_role, dec)[1]["content"]
