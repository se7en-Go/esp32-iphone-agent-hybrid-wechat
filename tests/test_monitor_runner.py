"""Multi-frame offline replay tests for MonitorRunner (baseline, dry-run red line, takeover, pause/halt)."""

from __future__ import annotations

import json
from pathlib import Path

from tests.fakes import FakePhoneDriver, MockOcrEngine, make_collapsed_chat_image, make_home_screen_image
from wechat_agent.guards.rate_limit import RateLimitConfig
from wechat_agent.models import Box, ClickResult, Observation, OcrBox
from wechat_agent.monitor.config import GroupPolicyConfig, MonitorConfig
from wechat_agent.monitor.decision import (
    HybridDecisionGate,
    LlmIntentGate,
    RuleDecisionGate,
    RuleSpec,
)
from wechat_agent.monitor.generators import HybridReplyGenerator, RuleTemplateGenerator
from wechat_agent.monitor.models import MonitorMode, TickOutcome
from wechat_agent.monitor.runner import MonitorRunner
from wechat_agent.monitor.store import MonitorStore


def _group_frame(
    ocr: MockOcrEngine,
    *,
    header: str = "产品测试群(12)",
    incoming: list[tuple[str, str, Box]] | None = None,
    outgoing: list[tuple[str, Box]] | None = None,
    extra: list[OcrBox] | None = None,
) -> bytes:
    in_boxes = [b for _, _, b in (incoming or [])]
    out_boxes = [b for _, b in (outgoing or [])]
    img = make_collapsed_chat_image(incoming_boxes=in_boxes, outgoing_boxes=out_boxes)
    boxes = [OcrBox(header, Box(0.32, 0.06, 0.68, 0.09))]
    for sender, text, b in incoming or []:
        if sender:
            boxes.append(OcrBox(sender, Box(b.left, b.top - 0.030, min(b.left + 0.12, b.right), b.top - 0.008)))
        boxes.append(OcrBox(text, Box(b.left + 0.02, b.top + 0.012, b.right - 0.02, b.bottom - 0.012)))
    for text, b in outgoing or []:
        boxes.append(OcrBox(text, Box(b.left + 0.02, b.top + 0.012, b.right - 0.02, b.bottom - 0.012)))
    if extra:
        boxes.extend(extra)
    ocr.register(img, boxes)
    return img


def _monitor_cfg(
    tmp_path: Path,
    *,
    mode: MonitorMode = MonitorMode.OBSERVE_ONLY,
    allow_live_send: bool = False,
    group_name: str = "产品测试群",
    allow_dry_run_paste: bool = False,
) -> MonitorConfig:
    return MonitorConfig(
        mode=mode,
        allow_live_send=allow_live_send,
        target_groups=(
            GroupPolicyConfig(
                name=group_name,
                self_aliases=("运营助手",),
                allow_dry_run_paste=allow_dry_run_paste,
            ),
        ),
        self_aliases=("运营助手",),
        state_db_path=tmp_path / "state" / "monitor.db",
        audit_log_path=tmp_path / "state" / "audit.jsonl",
        evidence_root=tmp_path / "runs",
        rate_limits=RateLimitConfig(
            global_min_interval_seconds=15.0,
            global_max_per_hour=15,
            per_group_cooldown_seconds=60.0,
            per_group_max_per_hour=6,
            human_takeover_cooldown_seconds=600.0,
            quiet_hours="off",
        ),
    )


def _rules_gate_and_gen() -> tuple[RuleDecisionGate, HybridReplyGenerator]:
    rules = [
        RuleSpec(
            rule_id="hours",
            keywords=("营业时间",),
            require_mention=False,
            intent="ask_hours",
            confidence=0.95,
            risk_score=0.05,
            reply_templates=("@{sender} 白天9点到晚上8点都在",),
        )
    ]
    return RuleDecisionGate(rules), HybridReplyGenerator(RuleTemplateGenerator(rules), None)


def test_baseline_observe_only_and_real_group_dry_run_red_line(tmp_path: Path) -> None:
    driver = FakePhoneDriver()
    ocr = MockOcrEngine()
    gate, gen = _rules_gate_and_gen()

    f_base = _group_frame(
        ocr,
        incoming=[("老用户", "之前问过营业时间了", Box(0.14, 0.25, 0.65, 0.31))],
    )
    f_new = _group_frame(
        ocr,
        incoming=[
            ("老用户", "之前问过营业时间了", Box(0.14, 0.25, 0.65, 0.31)),
            ("新同学", "请问营业时间是几点", Box(0.14, 0.42, 0.66, 0.48)),
        ],
    )
    driver.observe_queue = [
        Observation("s1", f_base),
        Observation("s2", f_new),
        Observation("s3", f_new),
    ]

    now = {"t": 1000.0}
    cfg = _monitor_cfg(tmp_path, mode=MonitorMode.DRY_RUN, allow_dry_run_paste=False)
    with MonitorStore(cfg.state_db_path, cfg.audit_log_path) as store:
        runner = MonitorRunner(driver, ocr, cfg, store, gate, gen, clock=lambda: now["t"], sleeper=lambda *_: None)
        r0 = runner.run_tick(0)
        assert r0.outcome == TickOutcome.BASELINE_SEEDED
        now["t"] += 10.0
        r1 = runner.run_tick(1)
        assert r1.outcome == TickOutcome.DRY_RUN_LOGGED
        assert r1.reply_text == "@新同学 白天9点到晚上8点都在"
        assert r1.decision is not None and r1.decision["jev"]["Choice"] == "REPLY"

    # Real-group dry-run MUST perform ZERO device clicks/focus/paste!
    assert not any(c[0] in {"point", "click_observed", "paste_flow", "skill_prepare_send"} for c in driver.calls)

    # Cross-restart check: reopening store & seeing same frame s3 must NOT re-trigger
    with MonitorStore(cfg.state_db_path, cfg.audit_log_path) as store2:
        runner2 = MonitorRunner(driver, ocr, cfg, store2, gate, gen, clock=lambda: now["t"] + 20.0)
        r2 = runner2.run_tick(0)
        assert r2.outcome == TickOutcome.NO_NEW_MESSAGES


def test_high_risk_score_blocked_by_threshold(tmp_path: Path) -> None:
    class _HighRiskLlm:
        def chat_completion(self, messages, *, json_mode=False, temperature=None, max_tokens=None):
            return json.dumps(
                {
                    "should_reply": True,
                    "intent": "dispute",
                    "confidence": 0.90,
                    "risk_score": 0.85,
                    "reason": "sensitive dispute",
                }
            )

    driver = FakePhoneDriver()
    ocr = MockOcrEngine()
    gate = HybridDecisionGate(RuleDecisionGate([]), LlmIntentGate(_HighRiskLlm()))
    gen = HybridReplyGenerator(RuleTemplateGenerator([]), None)

    f0 = _group_frame(ocr, incoming=[])
    f1 = _group_frame(ocr, incoming=[("群友A", "你们这退款赔不赔", Box(0.14, 0.30, 0.65, 0.36))])
    driver.observe_queue = [Observation("s0", f0), Observation("s1", f1)]

    cfg = _monitor_cfg(tmp_path, mode=MonitorMode.LIVE, allow_live_send=True)
    with MonitorStore(cfg.state_db_path, cfg.audit_log_path) as store:
        runner = MonitorRunner(driver, ocr, cfg, store, gate, gen, clock=lambda: 2000.0)
        assert runner.run_tick(0).outcome == TickOutcome.BASELINE_SEEDED
        r1 = runner.run_tick(1)
        assert r1.outcome == TickOutcome.OBSERVED
        assert "Blocked by threshold" in r1.message
    assert not any(c[0] == "paste_flow" for c in driver.calls)


def test_human_takeover_silences_while_bot_own_reply_does_not(tmp_path: Path) -> None:
    from tests.fakes import make_keyboard_chat_image

    driver = FakePhoneDriver()
    ocr = MockOcrEngine()
    gate, gen = _rules_gate_and_gen()

    f_base = _group_frame(ocr, incoming=[])
    f_ask = _group_frame(ocr, incoming=[("小王", "请问营业时间", Box(0.14, 0.25, 0.56, 0.31))])

    # Frames for live SendMessageWorkflow (collapsed -> focused -> pasted -> sent bubble)
    kb_empty = make_keyboard_chat_image(blue_send=True, with_caret=True)
    ocr.register(
        kb_empty,
        [
            OcrBox("产品测试群(12)", Box(0.32, 0.06, 0.68, 0.09)),
            OcrBox("空格", Box(0.45, 0.885, 0.53, 0.905)),
            OcrBox("发送", Box(0.84, 0.885, 0.90, 0.905)),
        ],
    )
    reply_text = "@小王 白天9点到晚上8点都在"
    kb_pasted = kb_empty + b"pasted"
    ocr.register(
        kb_pasted,
        [
            OcrBox("产品测试群(12)", Box(0.32, 0.06, 0.68, 0.09)),
            OcrBox(reply_text, Box(0.18, 0.53, 0.58, 0.55)),
            OcrBox("空格", Box(0.45, 0.885, 0.53, 0.905)),
            OcrBox("发送", Box(0.84, 0.885, 0.90, 0.905)),
        ],
    )
    f_bot_bubble = _group_frame(
        ocr,
        incoming=[("小王", "请问营业时间", Box(0.14, 0.25, 0.56, 0.31))],
        outgoing=[(reply_text, Box(0.46, 0.38, 0.88, 0.44))],
    )
    f_human_bubble = _group_frame(
        ocr,
        incoming=[
            ("小王", "请问营业时间", Box(0.14, 0.25, 0.56, 0.31)),
            ("小李", "今天营业时间变了吗", Box(0.14, 0.52, 0.62, 0.58)),
        ],
        outgoing=[
            (reply_text, Box(0.46, 0.38, 0.88, 0.44)),
            ("我真人来回复一下", Box(0.50, 0.64, 0.88, 0.70)),
        ],
    )

    driver.observe_queue = [
        Observation("m0", f_base),       # tick 0 baseline
        Observation("m1", f_ask),        # tick 1 runner observe -> triggers live send
        Observation("s0", f_ask),        # live workflow initial observe
        Observation("s2", kb_pasted),    # live workflow post-paste observe
        Observation("m2", f_bot_bubble), # tick 2 sees bot's OWN bubble -> must NOT trigger takeover!
        Observation("m3", f_human_bubble),  # tick 3 sees manual human bubble + new question -> MUST silence!
    ]
    click_seq = [
        ClickResult("c-kb", clicked=True, command_acknowledged=True, image_bytes=kb_empty),
        ClickResult("c-sent", clicked=True, command_acknowledged=True, effect_verified=True, image_bytes=f_bot_bubble),
    ]
    driver.click_handler = lambda *args: click_seq.pop(0)

    now = {"t": 3000.0}
    cfg = _monitor_cfg(tmp_path, mode=MonitorMode.LIVE, allow_live_send=True)
    with MonitorStore(cfg.state_db_path, cfg.audit_log_path) as store:
        runner = MonitorRunner(driver, ocr, cfg, store, gate, gen, clock=lambda: now["t"])
        assert runner.run_tick(0).outcome == TickOutcome.BASELINE_SEEDED
        now["t"] += 10.0
        r1 = runner.run_tick(1)
        assert r1.outcome == TickOutcome.REPLIED
        now["t"] += 70.0
        r2 = runner.run_tick(2)
        assert r2.outcome == TickOutcome.NO_NEW_MESSAGES
        assert r2.human_takeover_detected is False
        now["t"] += 70.0
        r3 = runner.run_tick(3)
        assert r3.outcome == TickOutcome.SKIPPED_HUMAN_TAKEOVER
        assert r3.human_takeover_detected is True


def test_pause_on_lock_screen_left_group_and_halt_on_risk_phrase(tmp_path: Path) -> None:
    driver = FakePhoneDriver()
    ocr = MockOcrEngine()
    gate, gen = _rules_gate_and_gen()

    lock_img = make_collapsed_chat_image() + b"lock"
    ocr.register(lock_img, [OcrBox("向上轻扫以解锁", Box(0.30, 0.90, 0.70, 0.93))])

    other_chat_img = make_collapsed_chat_image() + b"other"
    ocr.register(other_chat_img, [OcrBox("另一个群(5)", Box(0.35, 0.06, 0.65, 0.09))])

    home_img = make_home_screen_image()
    ocr.register(home_img, [OcrBox("微信", Box(0.18, 0.305, 0.24, 0.325))])

    risk_img = make_collapsed_chat_image() + b"risk"
    ocr.register(
        risk_img,
        [
            OcrBox("产品测试群(12)", Box(0.32, 0.06, 0.68, 0.09)),
            OcrBox("账号异常，请验证身份", Box(0.20, 0.45, 0.80, 0.50)),
        ],
    )

    driver.observe_queue = [
        Observation("s-lock", lock_img),
        Observation("s-left", other_chat_img),
        Observation("s-home", home_img),
        Observation("s-risk", risk_img),
    ]

    cfg = _monitor_cfg(tmp_path, mode=MonitorMode.OBSERVE_ONLY)
    with MonitorStore(cfg.state_db_path, cfg.audit_log_path) as store:
        runner = MonitorRunner(driver, ocr, cfg, store, gate, gen, sleeper=lambda *_: None)
        reports = runner.run_loop(max_ticks=4)

    assert [r.outcome for r in reports] == [
        TickOutcome.PAUSED_LOCK_SCREEN,
        TickOutcome.PAUSED_LEFT_GROUP,
        TickOutcome.PAUSED_NON_WECHAT,
        TickOutcome.HALTED_RISK_PHRASE,
    ]
    assert reports[-1].should_halt is True
    # Zero blind clicks during pause/halt
    assert not any(c[0] in {"point", "click_observed", "paste_flow"} for c in driver.calls)


def test_group_monitor_example_config_and_send_error_halt(tmp_path: Path) -> None:
    from wechat_agent.config import repo_root
    from wechat_agent.models import PasteFlowResult, PasteRecoveryState
    from wechat_agent.monitor.config import load_monitor_config
    from tests.fakes import make_keyboard_chat_image

    example = load_monitor_config(repo_root() / "config" / "group-monitor.example.yaml")
    assert example.mode == MonitorMode.OBSERVE_ONLY
    assert example.allow_live_send is False
    assert example.target_groups == ()
    assert example.llm.enabled is False
    assert (example.poll_interval_min_seconds, example.poll_interval_max_seconds) == (8.0, 12.0)
    assert example.rate_limits.per_group_cooldown_seconds == 60.0
    assert example.rate_limits.per_group_max_per_hour == 6
    assert example.rate_limits.global_min_interval_seconds == 15.0
    assert example.rate_limits.global_max_per_hour == 15
    assert example.rate_limits.human_takeover_cooldown_seconds == 600.0

    # Live mode paste recovery failure MUST halt the monitor runner immediately
    driver = FakePhoneDriver()
    ocr = MockOcrEngine()
    gate, gen = _rules_gate_and_gen()

    f_base = _group_frame(ocr, incoming=[])
    f_ask = _group_frame(ocr, incoming=[("小王", "请问营业时间", Box(0.14, 0.25, 0.56, 0.31))])
    kb_empty = make_keyboard_chat_image(blue_send=True, with_caret=True)
    ocr.register(
        kb_empty,
        [
            OcrBox("产品测试群(12)", Box(0.32, 0.06, 0.68, 0.09)),
            OcrBox("空格", Box(0.45, 0.885, 0.53, 0.905)),
            OcrBox("发送", Box(0.84, 0.885, 0.90, 0.905)),
        ],
    )
    driver.observe_queue = [Observation("m0", f_base), Observation("m1", f_ask), Observation("s0", f_ask)]
    driver.click_handler = lambda *args: ClickResult("c-kb", clicked=True, command_acknowledged=True, image_bytes=kb_empty)
    driver.paste_flow_handler = lambda *args, **kwargs: PasteFlowResult(
        ok=False,
        cmd_v_sent=0,
        phase="copied",
        effect_verified=False,
        recovery_state=PasteRecoveryState.COPIED_NOT_PASTED,
        next_action="resolve clipboard",
        raw={"recovery_state": "copied_not_pasted"},
    )

    cfg = _monitor_cfg(tmp_path, mode=MonitorMode.LIVE, allow_live_send=True)
    with MonitorStore(cfg.state_db_path, cfg.audit_log_path) as store:
        runner = MonitorRunner(driver, ocr, cfg, store, gate, gen, sleeper=lambda *_: None)
        reports = runner.run_loop(max_ticks=5)
    assert [r.outcome for r in reports] == [TickOutcome.BASELINE_SEEDED, TickOutcome.HALTED_SEND_ERROR]
    assert reports[-1].should_halt is True


def test_repeated_identical_phrase_and_discontinuity_and_scroll_up(tmp_path: Path) -> None:
    driver = FakePhoneDriver()
    ocr = MockOcrEngine()
    rules = [
        RuleSpec(
            rule_id="zai_ma",
            keywords=("在吗", "价格多少"),
            require_mention=False,
            intent="faq",
            confidence=0.95,
            risk_score=0.05,
            reply_templates=("在的，价格见群公告哈",),
        )
    ]
    gate = RuleDecisionGate(rules)
    gen = HybridReplyGenerator(RuleTemplateGenerator(rules), None)

    # Hidden group nicknames -> sender=""
    f_base = _group_frame(
        ocr,
        incoming=[("", "大家好", Box(0.14, 0.22, 0.44, 0.28))],
    )
    f_ask1 = _group_frame(
        ocr,
        incoming=[
            ("", "大家好", Box(0.14, 0.22, 0.44, 0.28)),
            ("", "在吗", Box(0.14, 0.34, 0.35, 0.40)),
        ],
    )
    # Screen scrolled slightly (大家好 scrolled off, 好的收到 added, then same "在吗" asked again later!)
    f_ask_mid = _group_frame(
        ocr,
        incoming=[
            ("", "在吗", Box(0.14, 0.22, 0.35, 0.28)),
            ("", "好的收到", Box(0.14, 0.34, 0.48, 0.40)),
        ],
    )
    f_ask2 = _group_frame(
        ocr,
        incoming=[
            ("", "好的收到", Box(0.14, 0.22, 0.48, 0.28)),
            ("", "在吗", Box(0.14, 0.34, 0.36, 0.40)),
        ],
    )
    # Scroll up / partial previous viewport (bottom "在吗" off-screen): NO_NEW_MESSAGES & keep anchor
    f_scroll_up = _group_frame(
        ocr,
        incoming=[("", "好的收到", Box(0.14, 0.22, 0.48, 0.28))],
    )
    # Complete screen discontinuity (zero overlap with [好的收到, 在吗]) -> safely re-anchor
    f_jump = _group_frame(
        ocr,
        incoming=[
            ("王五", "价格多少", Box(0.14, 0.25, 0.50, 0.31)),
            ("赵六", "在吗", Box(0.14, 0.40, 0.38, 0.46)),
        ],
    )

    driver.observe_queue = [
        Observation("o0", f_base),
        Observation("o1", f_ask1),
        Observation("o2", f_ask1),  # after restart on unchanged frame
        Observation("o3", f_ask_mid),
        Observation("o4", f_ask2),
        Observation("o5", f_scroll_up),
        Observation("o6", f_jump),
    ]

    now = {"t": 1000.0}
    cfg = _monitor_cfg(tmp_path, mode=MonitorMode.DRY_RUN, allow_dry_run_paste=False)
    with MonitorStore(cfg.state_db_path, cfg.audit_log_path) as store:
        runner = MonitorRunner(driver, ocr, cfg, store, gate, gen, clock=lambda: now["t"])
        assert runner.run_tick(0).outcome == TickOutcome.BASELINE_SEEDED
        now["t"] += 10.0
        r1 = runner.run_tick(1)
        assert r1.outcome == TickOutcome.DRY_RUN_LOGGED
        assert r1.reply_text == "在的，价格见群公告哈"

    # Reopen store on unchanged screen -> cross-restart idempotent NO_NEW_MESSAGES
    with MonitorStore(cfg.state_db_path, cfg.audit_log_path) as store2:
        runner2 = MonitorRunner(driver, ocr, cfg, store2, gate, gen, clock=lambda: now["t"] + 10.0)
        assert runner2.run_tick(2).outcome == TickOutcome.NO_NEW_MESSAGES

        # Non-matching chatter advances viewport, then repeated identical "在吗" MUST trigger again!
        now["t"] += 20.0
        r3 = runner2.run_tick(3)
        assert r3.outcome == TickOutcome.OBSERVED  # "好的收到" ignored by rule
        now["t"] += 30.0
        r4 = runner2.run_tick(4)
        assert r4.outcome == TickOutcome.DRY_RUN_LOGGED
        assert r4.reply_text == "在的，价格见群公告哈"

        rows = store2._conn.execute(
            "SELECT fingerprint FROM reply_history WHERE status = 'dry_run_logged' ORDER BY id ASC"
        ).fetchall()
        assert len(rows) == 2 and rows[0]["fingerprint"] != rows[1]["fingerprint"]

        # Scroll up keeps anchor [好的收到, 在吗]
        assert runner2.run_tick(5).outcome == TickOutcome.NO_NEW_MESSAGES
        assert store2.get_last_viewport("产品测试群") == [
            ("incoming", "", "好的收到"),
            ("incoming", "", "在吗"),
        ]

        # Complete discontinuity re-anchors safely without blasting replies
        r6 = runner2.run_tick(6)
        assert r6.outcome == TickOutcome.BASELINE_SEEDED
        assert "discontinuity" in r6.message


def test_multi_message_tick_prioritizes_at_self_and_limits_one_reply_per_tick(tmp_path: Path) -> None:
    driver = FakePhoneDriver()
    ocr = MockOcrEngine()
    gate, gen = _rules_gate_and_gen()

    f0 = _group_frame(ocr, incoming=[])
    # Multiple new incoming messages in one polling interval:
    # Msg1 mentions @运营助手 + asks about 营业时间, Msg2 is casual tail chatter -> Msg1 must NOT be swallowed!
    f1 = _group_frame(
        ocr,
        incoming=[
            ("小王", "@运营助手\u2005请问营业时间", Box(0.14, 0.25, 0.65, 0.31)),
            ("小李", "哈哈好的", Box(0.14, 0.39, 0.45, 0.45)),
        ],
    )
    # Two qualifying messages in same tick -> at most 1 reply dispatched
    f2 = _group_frame(
        ocr,
        incoming=[
            ("小王", "@运营助手\u2005请问营业时间", Box(0.14, 0.25, 0.65, 0.31)),
            ("小李", "哈哈好的", Box(0.14, 0.39, 0.45, 0.45)),
            ("小赵", "明天营业时间呢", Box(0.14, 0.52, 0.58, 0.58)),
            ("小周", "@运营助手\u2005周末营业时间呢", Box(0.14, 0.65, 0.72, 0.71)),
        ],
    )
    driver.observe_queue = [Observation("m0", f0), Observation("m1", f1), Observation("m2", f2)]

    cfg = _monitor_cfg(tmp_path, mode=MonitorMode.DRY_RUN, allow_dry_run_paste=False)
    with MonitorStore(cfg.state_db_path, cfg.audit_log_path) as store:
        runner = MonitorRunner(driver, ocr, cfg, store, gate, gen, clock=lambda: 5000.0)
        assert runner.run_tick(0).outcome == TickOutcome.BASELINE_SEEDED
        r1 = runner.run_tick(1)
        assert r1.outcome == TickOutcome.DRY_RUN_LOGGED
        assert r1.new_incoming_count == 2
        assert r1.reply_text == "@小王 白天9点到晚上8点都在"

        # On tick 2, @小周 is prioritized over non-@ 小赵, and ONLY 1 reply is recorded for this tick
        r2 = runner.run_tick(2)
        assert r2.outcome == TickOutcome.DRY_RUN_LOGGED
        assert r2.new_incoming_count == 2
        assert r2.reply_text == "@小周 白天9点到晚上8点都在"
        logged = store._conn.execute(
            "SELECT COUNT(*) AS cnt FROM reply_history WHERE status = 'dry_run_logged'"
        ).fetchone()
        assert int(logged["cnt"]) == 2


def test_zero_confidence_threshold_and_role_prompt_propagation(tmp_path: Path) -> None:
    from wechat_agent.monitor.config import load_monitor_config
    from wechat_agent.monitor.generators import LlmReplyGenerator

    yaml_path = tmp_path / "monitor.yaml"
    yaml_path.write_text(
        """
mode: dry-run
role_prompt: "全局默认职责"
target_groups:
  - name: "产品测试群"
    gate: "llm"
    min_confidence: 0.0
    mention_min_confidence: 0.0
    role_prompt: "仅负责门店营业咨询；售后退款一律静默。"
""",
        encoding="utf-8",
    )
    loaded_cfg = load_monitor_config(yaml_path)
    assert loaded_cfg.role_prompt == "全局默认职责"
    assert loaded_cfg.active_group is not None
    assert loaded_cfg.active_group.min_confidence == 0.0
    assert loaded_cfg.active_group.mention_min_confidence == 0.0
    assert loaded_cfg.active_group.role_prompt == "仅负责门店营业咨询；售后退款一律静默。"

    class _CaptureLlm:
        def __init__(self) -> None:
            self.calls: list[list[dict[str, str]]] = []
            self._replies = [
                json.dumps(
                    {
                        "should_reply": True,
                        "intent": "low_conf_but_allowed",
                        "confidence": 0.15,
                        "risk_score": 0.10,
                        "reason": "confidence is >= 0.0 threshold",
                    }
                ),
                "今天九点到八点营业哦",
            ]

        def chat_completion(self, messages, *, json_mode=False, temperature=None, max_tokens=None):
            self.calls.append(messages)
            return self._replies.pop(0)

    llm = _CaptureLlm()
    gate = LlmIntentGate(llm)
    gen = LlmReplyGenerator(llm)

    driver = FakePhoneDriver()
    ocr = MockOcrEngine()
    f0 = _group_frame(ocr, incoming=[])
    f1 = _group_frame(ocr, incoming=[("小王", "大概几点开呀", Box(0.14, 0.25, 0.56, 0.31))])
    driver.observe_queue = [Observation("s0", f0), Observation("s1", f1)]

    cfg = MonitorConfig(
        mode=MonitorMode.DRY_RUN,
        target_groups=loaded_cfg.target_groups,
        self_aliases=("运营助手",),
        role_prompt="全局默认职责",
        min_confidence=0.70,
        mention_min_confidence=0.50,
        state_db_path=tmp_path / "state" / "m.db",
        audit_log_path=tmp_path / "state" / "audit.jsonl",
        evidence_root=tmp_path / "runs",
        rate_limits=RateLimitConfig(quiet_hours="off"),
    )
    with MonitorStore(cfg.state_db_path, cfg.audit_log_path) as store:
        runner = MonitorRunner(driver, ocr, cfg, store, gate, gen, clock=lambda: 6000.0)
        assert runner.run_tick(0).outcome == TickOutcome.BASELINE_SEEDED
        r1 = runner.run_tick(1)
        assert r1.outcome == TickOutcome.DRY_RUN_LOGGED
        assert r1.reply_text == "今天九点到八点营业哦"

    # Both LlmIntentGate and LlmReplyGenerator received per-group role_prompt override
    assert len(llm.calls) == 2
    assert "仅负责门店营业咨询；售后退款一律静默。" in llm.calls[0][1]["content"]
    assert "仅负责门店营业咨询；售后退款一律静默。" in llm.calls[1][1]["content"]


