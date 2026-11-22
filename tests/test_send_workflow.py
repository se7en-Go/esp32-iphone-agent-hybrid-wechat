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
        status_payload={
            "device": True,
            "screen": True,
            "calibrated": True,
            "ready": True,
            "guard": {"tripped": True, "reason": "风控"},
        }
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


def _make_chat_list_frame(ocr: MockOcrEngine, header: str, suffix: bytes) -> bytes:
    img = make_chat_list_image() + suffix
    ocr.register(
        img,
        [
            OcrBox(header, Box(0.36, 0.05, 0.64, 0.085)),
            OcrBox("文件传输助手", Box(0.20, 0.24, 0.48, 0.27)),
            OcrBox("13:44", Box(0.87, 0.24, 0.96, 0.265)),
            OcrBox("微信", Box(0.08, 0.90, 0.16, 0.925)),
            OcrBox("通讯录", Box(0.32, 0.90, 0.42, 0.925)),
        ],
    )
    return img


def test_navigation_waits_for_chat_list_sync_header_before_clicking(tmp_path: Path) -> None:
    driver = FakePhoneDriver()
    ocr = MockOcrEngine()
    frames = _register_happy_chat_flow(driver, ocr, message_text="等同步完成再点击")

    sync_1 = _make_chat_list_frame(ocr, "收取中...", b"sync1")
    sync_2 = _make_chat_list_frame(ocr, "Connecting...", b"sync2")
    ready_list = _make_chat_list_frame(ocr, "微信", b"ready")

    driver.observe_queue = [
        Observation(screen_id="scr-sync-1", image_bytes=sync_1),
        Observation(screen_id="scr-sync-2", image_bytes=sync_2),
        Observation(screen_id="scr-list-ready", image_bytes=ready_list),
        Observation(screen_id="scr-pasted", image_bytes=frames["pasted"]),
    ]

    seen_point_sids: list[str] = []

    def handle_point(x, y, sid, tb, rh):
        seen_point_sids.append(sid)
        return PointResult(screen_id=f"{sid}-pt", measured_pointer=MeasuredPointer(x, y), pointer_seen=True)

    driver.point_handler = handle_point
    click_seq = [
        ClickResult(screen_id="scr-chat", clicked=True, command_acknowledged=True, image_bytes=frames["collapsed"]),
        ClickResult(screen_id="scr-kb", clicked=True, command_acknowledged=True, image_bytes=frames["focused_empty"]),
    ]
    driver.click_handler = lambda *args: click_seq.pop(0)

    sleeps: list[float] = []
    rep = SendMessageWorkflow(driver, ocr, _config(tmp_path), sleeper=sleeps.append).run(
        "文件传输助手", "等同步完成再点击", send=False
    )
    assert rep.status == SendStatus.DRY_RUN_READY
    assert sleeps == [1.2, 1.2]
    # First point in chat list only happened AFTER header settled to "微信" (`scr-list-ready`)
    assert seen_point_sids[0] == "scr-list-ready"


def test_navigation_409_progressive_backoff_recovers_and_caps(tmp_path: Path) -> None:
    # Part A: 3 consecutive HTTP 409 responses recover on the 4th attempt using progressive backoff
    driver_ok = FakePhoneDriver()
    ocr_ok = MockOcrEngine()
    frames_ok = _register_happy_chat_flow(driver_ok, ocr_ok, message_text="409退避恢复")
    ready_list = _make_chat_list_frame(ocr_ok, "微信", b"ready-a")

    driver_ok.observe_queue = [
        Observation(screen_id="scr-l0", image_bytes=ready_list),
        Observation(screen_id="scr-l1", image_bytes=ready_list),
        Observation(screen_id="scr-l2", image_bytes=ready_list),
        Observation(screen_id="scr-l3", image_bytes=ready_list),
        Observation(screen_id="scr-pasted", image_bytes=frames_ok["pasted"]),
    ]
    attempts_ok = {"n": 0}

    def handle_point_ok(x, y, sid, tb, rh):
        attempts_ok["n"] += 1
        if attempts_ok["n"] <= 3:
            raise ConflictError("HTTP 409: screen changed since observation; observe again", status_code=409)
        return PointResult(screen_id=f"{sid}-pt", measured_pointer=MeasuredPointer(x, y), pointer_seen=True)

    driver_ok.point_handler = handle_point_ok
    click_seq_ok = [
        ClickResult(screen_id="scr-chat", clicked=True, command_acknowledged=True, image_bytes=frames_ok["collapsed"]),
        ClickResult(
            screen_id="scr-kb",
            clicked=True,
            command_acknowledged=True,
            image_bytes=frames_ok["focused_empty"],
        ),
    ]
    driver_ok.click_handler = lambda *args: click_seq_ok.pop(0)

    sleeps_ok: list[float] = []
    rep_ok = SendMessageWorkflow(driver_ok, ocr_ok, _config(tmp_path), sleeper=sleeps_ok.append).run(
        "文件传输助手", "409退避恢复", send=False
    )
    assert rep_ok.status == SendStatus.DRY_RUN_READY
    assert sleeps_ok == [0.8, 1.5, 2.2]

    # Part B: 6 consecutive HTTP 409 responses exhaust all 5 retries with full progressive backoff and abort cleanly
    driver_fail = FakePhoneDriver()
    ocr_fail = MockOcrEngine()
    ready_list_b = _make_chat_list_frame(ocr_fail, "微信", b"ready-b")
    driver_fail.observe_queue = [
        Observation(screen_id=f"scr-f{i}", image_bytes=ready_list_b) for i in range(6)
    ]

    def handle_point_fail(x, y, sid, tb, rh):
        raise ConflictError("HTTP 409: screen changed since observation; observe again", status_code=409)

    driver_fail.point_handler = handle_point_fail
    sleeps_fail: list[float] = []
    rep_fail = SendMessageWorkflow(driver_fail, ocr_fail, _config(tmp_path), sleeper=sleeps_fail.append).run(
        "文件传输助手", "409耗尽中止", send=False
    )
    assert rep_fail.status == SendStatus.ABORTED_NAVIGATION
    assert "exceeded 5 pre-click 409" in rep_fail.message
    assert sleeps_fail == [0.8, 1.5, 2.2, 2.5, 3.0]
    assert not any(c[0] == "click_observed" for c in driver_fail.calls)


def test_dry_run_4line_multiline_paste_with_adjacent_incoming_bubble_succeeds(tmp_path: Path) -> None:
    expected_text = "@测试群友A 群里的特卖品按群内官方团购链接或小程序下单就行，找不到入口的话直接问群主确认。"
    driver = FakePhoneDriver()
    ocr = MockOcrEngine()

    collapsed_img = make_collapsed_chat_image(incoming_boxes=[Box(0.17, 0.72, 0.72, 0.79)])
    ocr.register(
        collapsed_img,
        [
            OcrBox("示例测试群(12)", Box(0.28, 0.06, 0.72, 0.09)),
            OcrBox("测试群友A", Box(0.17, 0.695, 0.28, 0.715)),
            OcrBox("@示例助手怎么团购下单？", Box(0.19, 0.735, 0.70, 0.765)),
        ],
    )

    focused_empty_img = make_keyboard_chat_image(
        blue_send=False,
        with_caret=True,
        incoming_boxes=[Box(0.17, 0.44, 0.72, 0.495)],
    )
    ocr.register(
        focused_empty_img,
        [
            OcrBox("示例测试群(12)", Box(0.28, 0.06, 0.72, 0.09)),
            OcrBox("测试群友A", Box(0.17, 0.418, 0.28, 0.435)),
            OcrBox("@示例助手怎么团购下单？", Box(0.19, 0.455, 0.70, 0.482)),
            OcrBox("空格", Box(0.45, 0.885, 0.53, 0.905)),
            OcrBox("发送", Box(0.84, 0.885, 0.90, 0.905)),
        ],
    )

    expanded_pasted_img = make_keyboard_chat_image(
        blue_send=True,
        with_caret=True,
        input_box=Box(0.132, 0.456, 0.764, 0.576),
        incoming_boxes=[Box(0.17, 0.375, 0.72, 0.425)],
    )
    ocr.register(
        expanded_pasted_img,
        [
            OcrBox("示例测试群(12)", Box(0.28, 0.06, 0.72, 0.09)),
            OcrBox("17:15", Box(0.45, 0.32, 0.55, 0.34)),
            OcrBox("测试群友A", Box(0.17, 0.355, 0.28, 0.372)),
            OcrBox("@示例助手怎么团购下单？", Box(0.19, 0.390, 0.70, 0.415)),
            OcrBox("门", Box(0.050, 0.465, 0.082, 0.478)),
            OcrBox("@测试群友A 群里的特卖品按", Box(0.164, 0.467, 0.708, 0.487)),
            OcrBox("群内官方团购链接或小程序下", Box(0.164, 0.494, 0.706, 0.512)),
            OcrBox("单就行，找不到入口的话直接", Box(0.160, 0.517, 0.710, 0.537)),
            OcrBox("问群主确认。", Box(0.156, 0.541, 0.422, 0.567)),
            OcrBox("空格", Box(0.45, 0.885, 0.53, 0.905)),
            OcrBox("发送", Box(0.84, 0.885, 0.90, 0.905)),
        ],
    )

    driver.observe_queue = [
        Observation(screen_id="scr-init", image_bytes=collapsed_img),
        Observation(screen_id="scr-pasted", image_bytes=expanded_pasted_img),
    ]
    driver.click_handler = lambda *args: ClickResult(
        screen_id="scr-kb",
        clicked=True,
        command_acknowledged=True,
        image_bytes=focused_empty_img,
    )
    driver.paste_handler = lambda text, tb, et: PasteFlowResult(
        ok=True,
        cmd_v_sent=1,
        phase="completed",
        effect_verified=True,
        recovery_state=PasteRecoveryState.COMPLETED,
        next_action="ready",
    )

    cfg = _config(tmp_path, allowed_targets=("示例测试群",), allow_group_count=True)
    rep = SendMessageWorkflow(driver, ocr, cfg).run("示例测试群", expected_text, send=False)
    assert rep.status == SendStatus.DRY_RUN_READY
    assert "post_paste_verified" in rep.steps


def test_for_click_frame_drift_and_pre_send_409_drift_abort_without_clicking(tmp_path: Path) -> None:
    # Part A: Screen drifts to another chat on observe(for_click=True) -> MUST abort with ZERO click_observed
    driver = FakePhoneDriver()
    ocr = MockOcrEngine()
    frames = _register_happy_chat_flow(driver, ocr, message_text="防漂移校验")

    wrong_chat_img = make_keyboard_chat_image(blue_send=True, with_caret=True) + b"wrong_chat"
    ocr.register(
        wrong_chat_img,
        [
            OcrBox("无关会话", Box(0.35, 0.06, 0.65, 0.09)),
            OcrBox("防漂移校验", Box(0.18, 0.53, 0.52, 0.55)),
            OcrBox("空格", Box(0.45, 0.885, 0.53, 0.905)),
            OcrBox("发送", Box(0.84, 0.885, 0.90, 0.905)),
        ],
    )
    driver.observe_queue = [
        Observation(screen_id="scr-kb", image_bytes=frames["focused_empty"]),
        Observation(screen_id="scr-pasted", image_bytes=frames["pasted"]),
    ]
    driver.observe_for_click_queue = [
        Observation(screen_id="scr-fc-wrong", image_bytes=wrong_chat_img),
    ]
    rep = SendMessageWorkflow(driver, ocr, _config(tmp_path)).run("文件传输助手", "防漂移校验", send=True)
    assert rep.status == SendStatus.ABORTED_PASTE_VERIFY
    assert not any(name == "click_observed" for name, _ in driver.calls)
    assert not any(name == "skill_confirm_send" for name, _ in driver.calls)

    # Part B: Pre-send point hits 409 and re-observed frame lost draft -> aborts cleanly with ZERO click_observed
    driver2 = FakePhoneDriver()
    ocr2 = MockOcrEngine()
    frames2 = _register_happy_chat_flow(driver2, ocr2, message_text="重观防漂移")
    driver2.observe_queue = [
        Observation(screen_id="scr-kb2", image_bytes=frames2["focused_empty"]),
        Observation(screen_id="scr-pasted2", image_bytes=frames2["pasted"]),
        # Re-observed frame after 409 has empty input box (draft gone)
        Observation(screen_id="scr-reobs-empty", image_bytes=frames2["focused_empty"]),
    ]
    driver2.point_handler = lambda *args: (_ for _ in ()).throw(
        ConflictError("HTTP 409: caret blinked", status_code=409)
    )
    rep2 = SendMessageWorkflow(driver2, ocr2, _config(tmp_path)).run("文件传输助手", "重观防漂移", send=True)
    assert rep2.status == SendStatus.ABORTED_PASTE_VERIFY
    assert not any(name == "click_observed" for name, _ in driver2.calls)


@pytest.mark.parametrize(
    ("clicked", "acked"),
    [(False, True), (True, False), (False, False)],
)
def test_unacknowledged_send_click_never_settles_as_sent(
    tmp_path: Path,
    clicked: bool,
    acked: bool,
) -> None:
    driver = FakePhoneDriver()
    ocr = MockOcrEngine()
    frames = _register_happy_chat_flow(driver, ocr, message_text="点击未确认禁止结算", bubble_after_send=True)
    driver.observe_queue = [
        Observation(screen_id="scr-kb", image_bytes=frames["focused_empty"]),
        Observation(screen_id="scr-pasted", image_bytes=frames["pasted"]),
    ]
    # Even though returned screenshot has a matching green bubble, clicked/acked is False!
    driver.click_handler = lambda *args: ClickResult(
        screen_id="scr-sent",
        clicked=clicked,
        command_acknowledged=acked,
        image_bytes=frames["sent"],
    )
    rep = SendMessageWorkflow(driver, ocr, _config(tmp_path)).run("文件传输助手", "点击未确认禁止结算", send=True)
    assert rep.status == SendStatus.SEND_UNCONFIRMED
    assert not any(name == "skill_confirm_send" for name, _ in driver.calls)


def test_post_send_wrong_title_never_settles_as_sent(tmp_path: Path) -> None:
    driver = FakePhoneDriver()
    ocr = MockOcrEngine()
    frames = _register_happy_chat_flow(driver, ocr, message_text="发送后验标题", bubble_after_send=True)

    wrong_post_img = make_collapsed_chat_image(outgoing_boxes=[Box(0.55, 0.40, 0.90, 0.46)]) + b"wrong_post"
    ocr.register(
        wrong_post_img,
        [
            OcrBox("另一个会话", Box(0.35, 0.06, 0.65, 0.09)),
            OcrBox("发送后验标题", Box(0.58, 0.415, 0.86, 0.445)),
        ],
    )
    driver.observe_queue = [
        Observation(screen_id="scr-kb", image_bytes=frames["focused_empty"]),
        Observation(screen_id="scr-pasted", image_bytes=frames["pasted"]),
        Observation(screen_id="scr-reobs-wrong", image_bytes=wrong_post_img),
    ]
    driver.click_handler = lambda *args: ClickResult(
        screen_id="scr-sent-wrong",
        clicked=True,
        command_acknowledged=True,
        image_bytes=wrong_post_img,
    )
    rep = SendMessageWorkflow(driver, ocr, _config(tmp_path)).run("文件传输助手", "发送后验标题", send=True)
    assert rep.status == SendStatus.SEND_UNCONFIRMED
    assert "title" in rep.message.lower()
    assert not any(name == "skill_confirm_send" for name, _ in driver.calls)


def test_navigation_single_frame_ocr_miss_recovers_with_passive_reobserve(tmp_path: Path) -> None:
    driver = FakePhoneDriver()
    ocr = MockOcrEngine()
    frames = _register_happy_chat_flow(driver, ocr, message_text="单帧漏检恢复")

    blank_ocr_img = make_collapsed_chat_image() + b"blank_ocr"
    ocr.register(blank_ocr_img, [])  # Single-frame transient OCR miss -> ScreenKind.UNKNOWN

    driver.observe_queue = [
        Observation(screen_id="scr-miss", image_bytes=blank_ocr_img),
        Observation(screen_id="scr-recovered", image_bytes=frames["focused_empty"]),
        Observation(screen_id="scr-pasted", image_bytes=frames["pasted"]),
    ]
    sleeps: list[float] = []
    rep = SendMessageWorkflow(driver, ocr, _config(tmp_path), sleeper=sleeps.append).run(
        "文件传输助手",
        "单帧漏检恢复",
        send=False,
    )
    assert rep.status == SendStatus.DRY_RUN_READY
    assert sleeps == [0.3]

