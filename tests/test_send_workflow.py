"""Comprehensive M1 offline unit tests for SendMessageWorkflow."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.fakes import (
    FakePhoneDriver,
    MockOcrEngine,
    make_chat_list_image,
    make_collapsed_chat_image,
    make_home_screen_image,
    make_keyboard_chat_image,
)
from wechat_agent.adapters.driver import ConflictError
from wechat_agent.config import WeChatSendConfig
from wechat_agent.models import (
    Box,
    ClickResult,
    MeasuredPointer,
    Observation,
    OcrBox,
    PasteFlowResult,
    PasteRecoveryState,
    PointResult,
    SendStatus,
)
from wechat_agent.workflows.send_message import SendMessageWorkflow


def _config(tmp_path: Path, **overrides) -> WeChatSendConfig:
    defaults = dict(
        allowed_targets=("文件传输助手",),
        evidence_root=tmp_path / "runs",
        point_rehome_retries=1,
        pre_send_point_retries=3,
        require_mounted_skill=True,
    )
    defaults.update(overrides)
    return WeChatSendConfig(**defaults)


def _register_happy_chat_flow(
    driver: FakePhoneDriver,
    ocr: MockOcrEngine,
    *,
    message_text: str = "测试消息 M1",
    with_draft: str | None = None,
    bubble_after_send: bool = True,
) -> dict[str, bytes]:
    """Wire deterministic screens for collapsed chat -> keyboard -> pasted -> post-send."""
    collapsed_img = make_collapsed_chat_image()
    ocr.register(
        collapsed_img,
        [OcrBox("文件传输助手", Box(0.35, 0.06, 0.65, 0.09))],
    )

    focused_empty_img = make_keyboard_chat_image(blue_send=True, with_caret=True)
    focused_empty_boxes = [
        OcrBox("文件传输助手", Box(0.35, 0.06, 0.65, 0.09)),
        OcrBox("空格", Box(0.45, 0.885, 0.53, 0.905)),
        OcrBox("发送", Box(0.84, 0.885, 0.90, 0.905)),
    ]
    if with_draft is not None:
        focused_empty_boxes.append(OcrBox(with_draft, Box(0.18, 0.53, 0.45, 0.55)))
    ocr.register(focused_empty_img, focused_empty_boxes)

    pasted_img = make_keyboard_chat_image(blue_send=True, with_caret=True)
    # Slight pixel tweak so sha256 differs from empty keyboard image
    pasted_img = pasted_img + b"pasted"
    ocr.register(
        pasted_img,
        [
            OcrBox("文件传输助手", Box(0.35, 0.06, 0.65, 0.09)),
            OcrBox(message_text, Box(0.18, 0.53, 0.52, 0.55)),
            OcrBox("空格", Box(0.45, 0.885, 0.53, 0.905)),
            OcrBox("发送", Box(0.84, 0.885, 0.90, 0.905)),
        ],
    )

    if bubble_after_send:
        sent_img = make_collapsed_chat_image(outgoing_boxes=[Box(0.55, 0.40, 0.90, 0.46)])
        ocr.register(
            sent_img,
            [
                OcrBox("文件传输助手", Box(0.35, 0.06, 0.65, 0.09)),
                OcrBox(message_text, Box(0.58, 0.415, 0.86, 0.445)),
            ],
        )
    else:
        sent_img = make_collapsed_chat_image(outgoing_boxes=[])
        ocr.register(
            sent_img,
            [OcrBox("文件传输助手", Box(0.35, 0.06, 0.65, 0.09))],
        )

    return {
        "collapsed": collapsed_img,
        "focused_empty": focused_empty_img,
        "pasted": pasted_img,
        "sent": sent_img,
    }


def test_target_policy_blocks_real_contacts_and_groups(tmp_path: Path) -> None:
    driver = FakePhoneDriver()
    wf = SendMessageWorkflow(driver, MockOcrEngine(), _config(tmp_path))

    for bad_target in ("张三", "测试群", "", "   "):
        rep = wf.run(bad_target, "hello", send=False)
        assert rep.status == SendStatus.ABORTED_POLICY
        assert rep.succeeded is False
    assert driver.calls == []


def test_preflight_blocks_when_unready_or_skill_unmounted(tmp_path: Path) -> None:
    driver = FakePhoneDriver(skills=["wechat-group-reply"])  # missing wechat-send
    wf = SendMessageWorkflow(driver, MockOcrEngine(), _config(tmp_path))
    rep = wf.run("文件传输助手", "hello", send=False)
    assert rep.status == SendStatus.ABORTED_PREFLIGHT
    assert "HIDAGENT_SKILLS" in rep.message

    tripped = FakePhoneDriver(
        status_payload={"device": True, "screen": True, "calibrated": True, "ready": True, "guard": {"tripped": True, "reason": "风控"}}
    )
    rep2 = SendMessageWorkflow(tripped, MockOcrEngine(), _config(tmp_path)).run("文件传输助手", "hello")
    assert rep2.status == SendStatus.ABORTED_PREFLIGHT


def test_navigation_from_visible_home_icon_skips_search_app(tmp_path: Path) -> None:
    driver = FakePhoneDriver()
    ocr = MockOcrEngine()
    frames = _register_happy_chat_flow(driver, ocr, message_text="从桌面进微信")

    home_img = make_home_screen_image()
    wechat_label = OcrBox("微信", Box(0.18, 0.305, 0.24, 0.325))
    ocr.register(home_img, [wechat_label, OcrBox("设置", Box(0.45, 0.305, 0.51, 0.325))])

    chat_list_img = make_chat_list_image()
    chat_title_box = OcrBox("文件传输助手", Box(0.202, 0.281, 0.480, 0.304))
    chat_time_box = OcrBox("14:14", Box(0.878, 0.279, 0.962, 0.296))
    ocr.register(
        chat_list_img,
        [
            OcrBox("微信", Box(0.45, 0.05, 0.55, 0.08)),
            chat_title_box,
            chat_time_box,
            OcrBox("微信", Box(0.08, 0.90, 0.16, 0.925)),
            OcrBox("通讯录", Box(0.32, 0.90, 0.42, 0.925)),
            OcrBox("发现", Box(0.58, 0.90, 0.66, 0.925)),
            OcrBox("我", Box(0.84, 0.90, 0.90, 0.925)),
        ],
    )

    driver.observe_queue = [
        Observation(screen_id="scr-home", image_bytes=home_img),
        Observation(screen_id="scr-pasted", image_bytes=frames["pasted"]),
    ]

    click_seq = [
        ClickResult(screen_id="scr-list", clicked=True, command_acknowledged=True, image_bytes=chat_list_img),
        ClickResult(screen_id="scr-chat", clicked=True, command_acknowledged=True, image_bytes=frames["collapsed"]),
        ClickResult(screen_id="scr-kb", clicked=True, command_acknowledged=True, image_bytes=frames["focused_empty"]),
    ]
    driver.click_handler = lambda *args: click_seq.pop(0)

    rep = SendMessageWorkflow(driver, ocr, _config(tmp_path)).run("文件传输助手", "从桌面进微信", send=False)
    assert rep.status == SendStatus.DRY_RUN_READY
    assert not any(name == "search_app" for name, _ in driver.calls)
    # First click on home screen must be strictly above the "微信" label (the icon body)
    point_calls = [args for name, args in driver.calls if name == "point"]
    assert point_calls[0]["y"] < wechat_label.box.top
    # Second point (chat list row) must aim in the right-side blank area between title and timestamp
    chat_row_aim_x = point_calls[1]["x"]
    assert chat_row_aim_x > chat_title_box.box.right
    assert chat_row_aim_x < chat_time_box.box.left


def test_navigation_falls_back_to_search_app_when_wechat_absent(tmp_path: Path) -> None:
    driver = FakePhoneDriver()
    ocr = MockOcrEngine()
    frames = _register_happy_chat_flow(driver, ocr, message_text="Spotlight打开")

    home_other_img = make_home_screen_image() + b"page2"
    ocr.register(home_other_img, [OcrBox("日历", Box(0.18, 0.305, 0.24, 0.325))])

    spotlight_img = make_home_screen_image() + b"spotlight"
    ocr.register(
        spotlight_img,
        [
            OcrBox("WeChat", Box(0.16, 0.08, 0.36, 0.11)),
            OcrBox("取消", Box(0.82, 0.08, 0.94, 0.11)),
            OcrBox("微信", Box(0.20, 0.24, 0.32, 0.27)),
            OcrBox("打开", Box(0.78, 0.24, 0.88, 0.27)),
        ],
    )

    driver.observe_queue = [
        Observation(screen_id="scr-home2", image_bytes=home_other_img),
        Observation(screen_id="scr-pasted", image_bytes=frames["pasted"]),
    ]
    driver.search_app_handler = lambda text, sid: Observation(screen_id="scr-spot", image_bytes=spotlight_img)
    click_seq = [
        ClickResult(screen_id="scr-chat", clicked=True, command_acknowledged=True, image_bytes=frames["collapsed"]),
        ClickResult(screen_id="scr-kb", clicked=True, command_acknowledged=True, image_bytes=frames["focused_empty"]),
    ]
    driver.click_handler = lambda *args: click_seq.pop(0)

    rep = SendMessageWorkflow(driver, ocr, _config(tmp_path)).run("文件传输助手", "Spotlight打开", send=False)
    assert rep.status == SendStatus.DRY_RUN_READY
    search_calls = [args["text"] for name, args in driver.calls if name == "search_app"]
    assert search_calls == ["WeChat"]


def test_existing_draft_aborts_without_clearing_or_pasting(tmp_path: Path) -> None:
    driver = FakePhoneDriver()
    ocr = MockOcrEngine()
    frames = _register_happy_chat_flow(driver, ocr, with_draft="重要草稿不能删")

    driver.observe_queue = [Observation(screen_id="scr-kb-draft", image_bytes=frames["focused_empty"])]
    rep = SendMessageWorkflow(driver, ocr, _config(tmp_path)).run("文件传输助手", "新文本", send=True)

    assert rep.status == SendStatus.ABORTED_EXISTING_DRAFT
    assert rep.observed_draft == "重要草稿不能删"
    assert not any(name == "paste_flow" for name, _ in driver.calls)
    assert not any(name == "click_observed" for name, _ in driver.calls)


@pytest.mark.parametrize(
    "recovery_state",
    [
        PasteRecoveryState.NOT_COPIED,
        PasteRecoveryState.CLAIMED_NOT_CONFIRMED,
        PasteRecoveryState.COPIED_NOT_PASTED,
        PasteRecoveryState.PASTE_SENT_UNCONFIRMED,
    ],
)
def test_paste_flow_recovery_states_halt_immediately_never_retry(
    tmp_path: Path,
    recovery_state: PasteRecoveryState,
) -> None:
    driver = FakePhoneDriver()
    ocr = MockOcrEngine()
    frames = _register_happy_chat_flow(driver, ocr)

    driver.observe_queue = [Observation(screen_id="scr-kb", image_bytes=frames["focused_empty"])]
    driver.paste_flow_handler = lambda *args, **kwargs: PasteFlowResult(
        ok=False,
        cmd_v_sent=1 if recovery_state == PasteRecoveryState.PASTE_SENT_UNCONFIRMED else 0,
        phase="failed",
        effect_verified=False,
        recovery_state=recovery_state,
        next_action=f"Stop for {recovery_state.value}",
        error=f"stopped at {recovery_state.value}",
        raw={"recovery_state": recovery_state.value, "cmd_v_sent": 0},
    )

    rep = SendMessageWorkflow(driver, ocr, _config(tmp_path)).run("文件传输助手", "测试消息 M1", send=True)
    assert rep.status == SendStatus.ABORTED_PASTE_RECOVERY
    assert rep.recovery_state == recovery_state
    assert len([c for c in driver.calls if c[0] == "paste_flow"]) == 1
    assert not any(c[0] == "skill_prepare_send" for c in driver.calls)
    assert not any(c[0] == "click_observed" for c in driver.calls)


def test_dry_run_stops_before_send_never_calls_prepare_or_send_click(tmp_path: Path) -> None:
    driver = FakePhoneDriver()
    ocr = MockOcrEngine()
    frames = _register_happy_chat_flow(driver, ocr, message_text="只贴不发")

    driver.observe_queue = [
        Observation(screen_id="scr-chat", image_bytes=frames["collapsed"]),
        Observation(screen_id="scr-pasted", image_bytes=frames["pasted"]),
    ]
    driver.click_handler = lambda *args: ClickResult(
        screen_id="scr-kb",
        clicked=True,
        command_acknowledged=True,
        image_bytes=frames["focused_empty"],
    )

    rep = SendMessageWorkflow(driver, ocr, _config(tmp_path)).run("文件传输助手", "只贴不发", send=False)
    assert rep.status == SendStatus.DRY_RUN_READY
    assert rep.succeeded is True
    assert not any(c[0] == "skill_prepare_send" for c in driver.calls)
    assert not any(c[0] == "skill_confirm_send" for c in driver.calls)
    send_clicks = [args for name, args in driver.calls if name == "click_observed" and args["target_label"]]
    assert send_clicks == []


def test_explicit_send_happy_path_with_pre_send_409_caret_retry(tmp_path: Path) -> None:
    driver = FakePhoneDriver()
    ocr = MockOcrEngine()
    frames = _register_happy_chat_flow(driver, ocr, message_text="真机发送验收", bubble_after_send=True)

    driver.observe_queue = [
        Observation(screen_id="scr-kb-ready", image_bytes=frames["focused_empty"]),
        Observation(screen_id="scr-pasted", image_bytes=frames["pasted"]),
        Observation(screen_id="scr-pre-send-reobs", image_bytes=frames["pasted"]),
    ]
    driver.observe_for_click_queue = [
        Observation(screen_id="scr-fc-send", image_bytes=frames["pasted"]),
    ]

    point_attempts = {"n": 0}

    def handle_point(x, y, sid, target_box, rehome):
        point_attempts["n"] += 1
        # Simulate caret blink HTTP 409 on first pre-send point attempt
        if point_attempts["n"] == 1:
            raise ConflictError("HTTP 409: caret blink", status_code=409)
        return PointResult(screen_id="scr-pt-ok", measured_pointer=MeasuredPointer(x, y), pointer_seen=True)

    driver.point_handler = handle_point
    driver.click_handler = lambda *args: ClickResult(
        screen_id="scr-sent",
        clicked=True,
        command_acknowledged=True,
        effect_verified=True,
        image_bytes=frames["sent"],
    )

    rep = SendMessageWorkflow(driver, ocr, _config(tmp_path)).run("文件传输助手", "真机发送验收", send=True)
    assert rep.status == SendStatus.SENT_CONFIRMED
    assert rep.succeeded is True

    # Point retried once on 409; click_observed ran ONLY ONCE with target_label="发送",
    # expanded blue box, and explicit caret_scope matching the full white input box!
    paste_calls = [args for name, args in driver.calls if name == "paste_flow"]
    assert len(paste_calls) == 1
    assert paste_calls[0]["target_box"].right >= 0.75

    point_calls = [args for name, args in driver.calls if name == "point"]
    assert len(point_calls) == 2
    assert point_calls[1]["rehome"] is True
    assert point_calls[0]["caret_scope"] is not None
    assert point_calls[1]["caret_scope"] is not None
    assert point_calls[1]["caret_scope"].right >= 0.75

    click_calls = [args for name, args in driver.calls if name == "click_observed"]
    assert len(click_calls) == 1
    assert click_calls[0]["target_label"] == "发送"
    assert click_calls[0]["target_box"].left >= 0.74  # expanded blue button avoiding space bar
    assert click_calls[0]["caret_scope"] == point_calls[1]["caret_scope"]

    confirm_calls = [args for name, args in driver.calls if name == "skill_confirm_send"]
    assert len(confirm_calls) == 1
    assert confirm_calls[0]["bubble_visible"] is True


def test_explicit_send_click_failure_never_retries_click(tmp_path: Path) -> None:
    driver = FakePhoneDriver()
    ocr = MockOcrEngine()
    frames = _register_happy_chat_flow(driver, ocr, message_text="禁止重发点击")

    driver.observe_queue = [
        Observation(screen_id="scr-kb", image_bytes=frames["focused_empty"]),
        Observation(screen_id="scr-pasted", image_bytes=frames["pasted"]),
    ]
    driver.click_handler = lambda *args: (_ for _ in ()).throw(
        ConflictError("HTTP 409: wait click frame changed", status_code=409)
    )

    rep = SendMessageWorkflow(driver, ocr, _config(tmp_path)).run("文件传输助手", "禁止重发点击", send=True)
    assert rep.status == SendStatus.SEND_UNCONFIRMED
    assert rep.succeeded is False
    click_calls = [c for c in driver.calls if c[0] == "click_observed"]
    assert len(click_calls) == 1
    assert not any(c[0] == "skill_confirm_send" for c in driver.calls)


def test_explicit_send_without_visible_bubble_stays_unconfirmed(tmp_path: Path) -> None:
    driver = FakePhoneDriver()
    ocr = MockOcrEngine()
    # bubble_after_send=False -> input cleared, hardware ok, but NO new green bubble!
    frames = _register_happy_chat_flow(driver, ocr, message_text="无气泡不能确认", bubble_after_send=False)

    driver.observe_queue = [
        Observation(screen_id="scr-kb", image_bytes=frames["focused_empty"]),
        Observation(screen_id="scr-pasted", image_bytes=frames["pasted"]),
        Observation(screen_id="scr-reobs-after-send", image_bytes=frames["sent"]),
    ]
    driver.click_handler = lambda *args: ClickResult(
        screen_id="scr-sent-empty",
        clicked=True,
        command_acknowledged=True,
        effect_verified=True,
        image_bytes=frames["sent"],
    )

    rep = SendMessageWorkflow(driver, ocr, _config(tmp_path)).run("文件传输助手", "无气泡不能确认", send=True)
    assert rep.status == SendStatus.SEND_UNCONFIRMED
    assert not any(c[0] == "skill_confirm_send" for c in driver.calls)


def test_risk_phrase_triggers_immediate_circuit_breaker(tmp_path: Path) -> None:
    driver = FakePhoneDriver()
    ocr = MockOcrEngine()
    risk_img = make_collapsed_chat_image()
    ocr.register(
        risk_img,
        [
            OcrBox("文件传输助手", Box(0.35, 0.06, 0.65, 0.09)),
            OcrBox("当前操作频繁，请稍后再试", Box(0.20, 0.45, 0.80, 0.50)),
        ],
    )
    driver.observe_queue = [Observation(screen_id="scr-risk", image_bytes=risk_img)]

    rep = SendMessageWorkflow(driver, ocr, _config(tmp_path)).run("文件传输助手", "测试", send=True)
    assert rep.status == SendStatus.ABORTED_RISK_PHRASE
    assert not any(c[0] == "paste_flow" for c in driver.calls)
    assert not any(c[0] == "click_observed" for c in driver.calls)


def test_navigation_recovers_from_chat_list_point_409_by_reobserving_and_relocating_row(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("wechat_agent.workflows.navigation.time.sleep", lambda *_: None)
    driver = FakePhoneDriver()
    ocr = MockOcrEngine()
    frames = _register_happy_chat_flow(driver, ocr, message_text="409重观察进会话")

    # Initial chat-list frame right after opening WeChat (row at y=0.22), then shifted frame after sync (y=0.31)
    list_img_1 = make_chat_list_image() + b"1"
    ocr.register(
        list_img_1,
        [
            OcrBox("微信", Box(0.45, 0.05, 0.55, 0.08)),
            OcrBox("文件传输助手", Box(0.20, 0.21, 0.48, 0.235)),
            OcrBox("16:38", Box(0.87, 0.21, 0.96, 0.23)),
            OcrBox("微信", Box(0.08, 0.90, 0.16, 0.925)),
            OcrBox("通讯录", Box(0.32, 0.90, 0.42, 0.925)),
        ],
    )
    list_img_2 = make_chat_list_image() + b"2"
    shifted_row_title = OcrBox("文件传输助手", Box(0.20, 0.30, 0.48, 0.325))
    ocr.register(
        list_img_2,
        [
            OcrBox("微信", Box(0.45, 0.05, 0.55, 0.08)),
            shifted_row_title,
            OcrBox("16:38", Box(0.87, 0.30, 0.96, 0.32)),
            OcrBox("微信", Box(0.08, 0.90, 0.16, 0.925)),
            OcrBox("通讯录", Box(0.32, 0.90, 0.42, 0.925)),
        ],
    )

    driver.observe_queue = [
        Observation(screen_id="scr-list-1", image_bytes=list_img_1),
        Observation(screen_id="scr-list-2-fresh", image_bytes=list_img_2),
        Observation(screen_id="scr-pasted", image_bytes=frames["pasted"]),
    ]

    point_calls = {"n": 0}

    def handle_point(x, y, sid, tb, rh):
        point_calls["n"] += 1
        if point_calls["n"] == 1:
            raise ConflictError("HTTP 409: screen changed since observation; observe again", status_code=409)
        return PointResult(screen_id=f"{sid}-pt", measured_pointer=MeasuredPointer(x, y), pointer_seen=True)

    driver.point_handler = handle_point
    click_seq = [
        ClickResult(screen_id="scr-chat", clicked=True, command_acknowledged=True, image_bytes=frames["collapsed"]),
        ClickResult(screen_id="scr-kb", clicked=True, command_acknowledged=True, image_bytes=frames["focused_empty"]),
    ]
    driver.click_handler = lambda *args: click_seq.pop(0)

    rep = SendMessageWorkflow(driver, ocr, _config(tmp_path)).run("文件传输助手", "409重观察进会话", send=False)
    assert rep.status == SendStatus.DRY_RUN_READY
    # Second point used the relocated shifted row y (~0.31), NOT stale y (~0.22)!
    pts = [args for name, args in driver.calls if name == "point"]
    assert abs(pts[0]["y"] - 0.2225) < 0.01
    assert abs(pts[1]["y"] - shifted_row_title.box.center[1]) < 0.01
    # Zero duplicate click_observed retries
    assert len([c for c in driver.calls if c[0] == "click_observed"]) == 2


def test_focus_input_recovers_on_pre_click_409_and_passive_keyboard_settle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("wechat_agent.workflows.send_message.time.sleep", lambda *_: None)
    driver = FakePhoneDriver()
    ocr = MockOcrEngine()
    frames = _register_happy_chat_flow(driver, ocr, message_text="聚焦409恢复")

    driver.observe_queue = [
        Observation(screen_id="scr-chat-1", image_bytes=frames["collapsed"]),
        Observation(screen_id="scr-chat-2", image_bytes=frames["collapsed"]),
        # Passive re-observe right after focus click returns the settled keyboard frame
        Observation(screen_id="scr-kb-settled", image_bytes=frames["focused_empty"]),
        Observation(screen_id="scr-pasted", image_bytes=frames["pasted"]),
    ]

    pt_n = {"n": 0}

    def handle_point(x, y, sid, tb, rh):
        pt_n["n"] += 1
        if pt_n["n"] == 1:
            raise ConflictError("HTTP 409: screen changed since observation", status_code=409)
        return PointResult(screen_id=f"{sid}-pt", measured_pointer=MeasuredPointer(x, y), pointer_seen=True)

    driver.point_handler = handle_point
    # Click frame itself still shows collapsed bar before keyboard finishes animating upward
    driver.click_handler = lambda *args: ClickResult(
        screen_id="scr-click-animating",
        clicked=True,
        command_acknowledged=True,
        image_bytes=frames["collapsed"],
    )

    rep = SendMessageWorkflow(driver, ocr, _config(tmp_path)).run("文件传输助手", "聚焦409恢复", send=False)
    assert rep.status == SendStatus.DRY_RUN_READY
    # Focus click ran ONCE (never double-clicked)
    assert len([c for c in driver.calls if c[0] == "click_observed"]) == 1

