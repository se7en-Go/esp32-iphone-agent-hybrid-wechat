"""Unit tests for dynamic WeChat UI geometry, blue Send button expansion, drafts, and bubbles."""

from tests.fakes import (
    make_chat_list_image,
    make_collapsed_chat_image,
    make_home_screen_image,
    make_keyboard_chat_image,
)
from wechat_agent.models import Box, OcrBox
from wechat_agent.vision.geometry import (
    canonicalize_chat_title,
    is_in_home_indicator_zone,
    left_biased_input_aim,
)
from wechat_agent.vision.wechat_ui import (
    ScreenKind,
    detect_existing_draft,
    detect_keyboard_and_focused_input,
    detect_screen_context,
    extract_outgoing_bubbles,
    find_chat_row_box,
    find_home_wechat_icon,
    has_exact_chat_title,
    has_new_outgoing_bubble,
    locate_blue_send_button,
    locate_collapsed_input_box,
    verify_pasted_text,
)


def test_home_icon_clicks_icon_body_above_label_never_text_center() -> None:
    img = make_home_screen_image()
    label = OcrBox("微信", Box(0.18, 0.305, 0.24, 0.325))
    icon_box = find_home_wechat_icon([label], img)
    assert icon_box is not None
    assert icon_box.bottom < label.box.top
    assert not icon_box.contains(*label.box.center)


def test_wechat_chat_list_tabs_not_confused_with_home_icon() -> None:
    img = make_chat_list_image()
    boxes = [
        OcrBox("微信", Box(0.45, 0.05, 0.55, 0.08)),
        OcrBox("文件传输助手", Box(0.18, 0.20, 0.48, 0.23)),
        OcrBox("微信", Box(0.08, 0.90, 0.16, 0.925)),
        OcrBox("通讯录", Box(0.32, 0.90, 0.42, 0.925)),
        OcrBox("发现", Box(0.58, 0.90, 0.66, 0.925)),
        OcrBox("我", Box(0.84, 0.90, 0.90, 0.925)),
    ]
    assert detect_screen_context(boxes, "文件传输助手") == ScreenKind.WECHAT_CHAT_LIST
    assert find_home_wechat_icon(boxes, img) is None


def test_collapsed_input_aim_left_biased_and_avoids_home_indicator() -> None:
    img = make_collapsed_chat_image()
    box = locate_collapsed_input_box(img)
    assert box.bottom <= 0.948
    aim_x, aim_y = left_biased_input_aim(box)
    assert aim_x < box.center[0]
    assert aim_y <= 0.948
    assert not is_in_home_indicator_zone(aim_x, aim_y)


def test_locate_blue_send_button_expands_across_blue_body_and_avoids_space_bar() -> None:
    img = make_keyboard_chat_image(blue_send=True)
    # Include a misleading "发送" inside chat area above keyboard (y=0.30) to verify region filtering!
    send_label = OcrBox("发送", Box(0.84, 0.885, 0.90, 0.905))
    space_label = OcrBox("空格", Box(0.45, 0.885, 0.53, 0.905))
    chat_noise = OcrBox("我一会儿发送给你", Box(0.20, 0.30, 0.50, 0.33))
    chat_exact_noise = OcrBox("发送", Box(0.20, 0.35, 0.28, 0.37))

    located = locate_blue_send_button(img, [chat_noise, chat_exact_noise, space_label, send_label])
    assert located is not None
    btn_box, label = located
    assert label == "发送"
    # Expands dynamically across the blue button body (drawn at [0.75, 0.855, 0.985, 0.932]),
    # strictly larger than the OCR text box alone.
    assert btn_box.left < send_label.box.left
    assert btn_box.right > send_label.box.right
    assert btn_box.top < send_label.box.top
    assert btn_box.bottom > send_label.box.bottom
    assert btn_box.left >= 0.74
    # Strictly right of space bar & safe margin
    assert btn_box.left > space_label.box.right + 0.15


def test_locate_blue_send_button_rejects_ambiguous_duplicate_in_region() -> None:
    img = make_keyboard_chat_image(blue_send=True)
    boxes = [
        OcrBox("发送", Box(0.84, 0.885, 0.90, 0.905)),
        OcrBox("Send", Box(0.76, 0.65, 0.84, 0.68)),
    ]
    assert locate_blue_send_button(img, boxes) is None


def test_locate_send_button_fallback_without_blue_pixels_still_avoids_space() -> None:
    img = make_keyboard_chat_image(blue_send=False)
    send_label = OcrBox("Send", Box(0.84, 0.885, 0.91, 0.905))
    space_label = OcrBox("space", Box(0.45, 0.885, 0.55, 0.905))
    located = locate_blue_send_button(img, [space_label, send_label])
    assert located is not None
    btn_box, label = located
    assert label == "Send"
    assert btn_box.left >= 0.72
    assert btn_box.left > space_label.box.right + 0.10
    assert btn_box.bottom <= 0.95


def test_detect_keyboard_draft_and_pasted_text() -> None:
    img = make_keyboard_chat_image(with_caret=True)
    kb_boxes = [
        OcrBox("文件传输助手", Box(0.35, 0.06, 0.65, 0.09)),
        OcrBox("空格", Box(0.45, 0.885, 0.53, 0.905)),
        OcrBox("发送", Box(0.84, 0.885, 0.90, 0.905)),
    ]
    layout = detect_keyboard_and_focused_input(img, kb_boxes)
    assert layout is not None
    assert layout.caret_box is not None
    assert detect_existing_draft(kb_boxes, layout.input_box) is None

    # Existing draft inside input_box must be detected
    draft_boxes = kb_boxes + [OcrBox("未发完的草稿", Box(0.18, 0.53, 0.45, 0.55))]
    assert detect_existing_draft(draft_boxes, layout.input_box) == "未发完的草稿"

    # Verify pasted text exact vs duplicated
    pasted_boxes = kb_boxes + [OcrBox("你好，测试消息", Box(0.18, 0.53, 0.48, 0.55))]
    ok, obs = verify_pasted_text(pasted_boxes, layout.input_box, "你好，测试消息")
    assert ok is True
    assert obs == "你好，测试消息"

    dup_boxes = kb_boxes + [OcrBox("你好，测试消息你好，测试消息", Box(0.18, 0.53, 0.72, 0.55))]
    ok_dup, _ = verify_pasted_text(dup_boxes, layout.input_box, "你好，测试消息")
    assert ok_dup is False


def test_outgoing_green_bubble_distinguished_from_incoming_white_bubble() -> None:
    outgoing_box = Box(0.55, 0.36, 0.90, 0.42)
    incoming_box = Box(0.12, 0.22, 0.58, 0.28)
    img = make_collapsed_chat_image(outgoing_boxes=[outgoing_box], incoming_boxes=[incoming_box])
    ocr = [
        OcrBox("别人发来的消息", Box(0.15, 0.235, 0.45, 0.265)),
        OcrBox("我发出的消息", Box(0.58, 0.375, 0.86, 0.405)),
    ]
    outgoing = extract_outgoing_bubbles(img, ocr, chat_area=Box(0.0, 0.13, 1.0, 0.86))
    assert outgoing == ["我发出的消息"]
    assert has_new_outgoing_bubble([], outgoing, "我发出的消息") is True
    assert has_new_outgoing_bubble(["我发出的消息"], outgoing, "我发出的消息") is False
    assert has_new_outgoing_bubble([], [], "我发出的消息") is False


def test_rapidocr_engine_smoke() -> None:
    import io
    from PIL import Image, ImageDraw, ImageFont
    from wechat_agent.vision.ocr import RapidOcrEngine

    engine = RapidOcrEngine()
    assert engine.read_boxes(b"") == []

    img = Image.new("RGB", (320, 120), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default(size=32)
    draw.text((40, 35), "WeChat", fill=(0, 0, 0), font=font)
    buf = io.BytesIO()
    img.save(buf, format="PNG")

    boxes = engine.read_boxes(buf.getvalue())
    assert any("WeChat" in b.text for b in boxes)


def test_find_chat_row_box_aims_at_right_blank_whitespace_avoiding_title_and_timestamp() -> None:
    title_box = OcrBox("文件传输助手", Box(0.202, 0.281, 0.480, 0.304))
    time_box = OcrBox("14:14", Box(0.878, 0.279, 0.962, 0.296))
    other_row_text = OcrBox("快讯：其他聊天消息", Box(0.204, 0.223, 0.930, 0.242))

    row_box = find_chat_row_box([other_row_text, time_box, title_box], "文件传输助手")
    assert row_box is not None
    aim_x, aim_y = row_box.center
    # Must aim strictly right of the black title text and strictly left of the timestamp
    assert row_box.left > title_box.box.right
    assert row_box.right <= 0.84
    assert aim_x > title_box.box.right + 0.05
    assert aim_x < time_box.box.left - 0.05
    assert title_box.box.top - 0.015 <= aim_y <= title_box.box.bottom + 0.015

    # Long title fallback stays valid and stays clear of timestamp
    long_title = OcrBox("这是一个超级超级超级长的微信群名称测试", Box(0.18, 0.40, 0.82, 0.43))
    long_time = OcrBox("15:20", Box(0.87, 0.40, 0.95, 0.42))
    fallback_box = find_chat_row_box([long_title, long_time], "这是一个超级超级超级长的微信群名称测试")
    assert fallback_box is not None
    assert fallback_box.right <= 0.84
    assert fallback_box.center[0] < long_time.box.left


def test_empty_white_input_box_with_right_mic_and_toolbar_icons_not_truncated() -> None:
    # Focused keyboard state with interior right-side microphone icon + outer voice/emoji icons
    kb_img = make_keyboard_chat_image(with_caret=True, with_right_mic_icon=True, with_side_toolbar_icons=True)
    kb_ocr = [
        OcrBox("文件传输助手", Box(0.35, 0.06, 0.65, 0.09)),
        OcrBox("空格", Box(0.45, 0.885, 0.53, 0.905)),
        OcrBox("发送", Box(0.84, 0.885, 0.90, 0.905)),
    ]
    layout = detect_keyboard_and_focused_input(kb_img, kb_ocr)
    assert layout is not None
    # Complete white box is [0.132..0.764] — must NOT truncate at mic (0.694) or bleed onto emoji (0.795)
    assert 0.125 <= layout.input_box.left <= 0.140
    assert 0.755 <= layout.input_box.right <= 0.775

    # Collapsed input bar with interior right-side microphone icon + outer toolbar icons
    col_img = make_collapsed_chat_image(with_right_mic_icon=True, with_side_toolbar_icons=True)
    col_box = locate_collapsed_input_box(col_img)
    assert 0.125 <= col_box.left <= 0.140
    assert 0.755 <= col_box.right <= 0.775
    assert col_box.bottom <= 0.948


def test_extract_outgoing_bubbles_merges_multiline_wrapped_short_last_line_and_splits_separate_bubbles() -> None:
    # Bubble 1: wide 2-line wrapped green bubble where line 2 ("测试成功") ends at right=0.376 (< 0.42),
    # left-aligned at 0.188 inside the single continuous green bubble [0.16..0.83, 0.17..0.245].
    # Bubble 2: separate green bubble stacked closely below with grey canvas gap between 0.245 and 0.268.
    b1 = Box(0.16, 0.170, 0.83, 0.245)
    b2 = Box(0.16, 0.268, 0.83, 0.312)
    img = make_collapsed_chat_image(outgoing_boxes=[b1, b2])

    ocr = [
        OcrBox("文件传输助手", Box(0.35, 0.06, 0.65, 0.09)),
        OcrBox("iphoneagent自动操控iphone", Box(0.188, 0.182, 0.798, 0.206)),
        OcrBox("测试成功", Box(0.188, 0.208, 0.376, 0.231)),
        OcrBox("示例头像", Box(0.858, 0.185, 0.972, 0.203)),  # right avatar must be excluded
        OcrBox("第二条独立消息", Box(0.188, 0.276, 0.560, 0.300)),
    ]

    bubbles = extract_outgoing_bubbles(img, ocr, chat_area=Box(0.0, 0.13, 1.0, 0.87))
    assert bubbles == [
        "iphoneagent自动操控iphone\n测试成功",
        "第二条独立消息",
    ]
    assert has_new_outgoing_bubble([], bubbles, "iphone agent 自动操控iphone测试成功") is True


def test_canonical_group_title_punctuation_and_parenthesized_suffix_matching() -> None:
    assert canonicalize_chat_title("示例分享•测试3群（198）") == "示例分享·测试3群(198)"
    assert canonicalize_chat_title("示例分享 ・ 测试3群 (198)") == "示例分享·测试3群(198)"

    header_198_12 = [OcrBox("示例分享·测试3群（198）（12）", Box(0.147, 0.082, 0.750, 0.104))]
    header_199_12 = [OcrBox("示例分享·测试3群（199）（12）", Box(0.147, 0.082, 0.750, 0.104))]
    header_no_198 = [OcrBox("示例分享·测试3群（12）", Box(0.147, 0.082, 0.750, 0.104))]

    target = "示例分享•测试3群（198）"
    assert has_exact_chat_title(header_198_12, target, allow_group_count=True) is True
    assert has_exact_chat_title(header_198_12, target, allow_group_count=False) is False
    assert has_exact_chat_title(header_199_12, target, allow_group_count=True) is False
    assert has_exact_chat_title(header_no_198, target, allow_group_count=True) is False

    # Chat list row lookup normalizes middle-dots and fullwidth parens while keeping (198) != (199)
    chat_rows = [
        OcrBox("示例分享·测试3群（199）", Box(0.20, 0.20, 0.65, 0.23)),
        OcrBox("示例分享·测试3群（198）", Box(0.20, 0.30, 0.65, 0.33)),
        OcrBox("14:36", Box(0.87, 0.30, 0.96, 0.32)),
    ]
    row = find_chat_row_box(chat_rows, target)
    assert row is not None
    assert 0.28 <= row.center[1] <= 0.34


def test_verify_pasted_text_4line_draft_strictly_excludes_adjacent_chat_bubbles_and_expand_icon() -> None:
    expected_text = "@测试群友A 群里的特卖品按群内官方团购链接或小程序下单就行，找不到入口的话直接问群主确认。"
    expanded_pill = Box(0.132, 0.456, 0.764, 0.576)
    img = make_keyboard_chat_image(
        blue_send=True,
        with_caret=True,
        input_box=expanded_pill,
        incoming_boxes=[Box(0.17, 0.375, 0.72, 0.425)],
    )
    ocr_boxes = [
        OcrBox("示例测试群(12)", Box(0.28, 0.06, 0.72, 0.09)),
        OcrBox("17:15", Box(0.45, 0.32, 0.55, 0.34)),
        OcrBox("测试群友A", Box(0.17, 0.355, 0.28, 0.372)),
        OcrBox("@示例助手怎么团购下单？", Box(0.19, 0.390, 0.70, 0.415)),
        OcrBox("门", Box(0.050, 0.465, 0.082, 0.478)),  # left expand icon outside pill
        OcrBox("@测试群友A 群里的特卖品按", Box(0.164, 0.467, 0.708, 0.487)),
        OcrBox("群内官方团购链接或小程序下", Box(0.164, 0.494, 0.706, 0.512)),
        OcrBox("单就行，找不到入口的话直接", Box(0.160, 0.517, 0.710, 0.537)),
        OcrBox("问群主确认。", Box(0.156, 0.541, 0.422, 0.567)),
        OcrBox("空格", Box(0.45, 0.885, 0.53, 0.905)),
        OcrBox("发送", Box(0.84, 0.885, 0.90, 0.905)),
    ]

    layout = detect_keyboard_and_focused_input(img, ocr_boxes)
    assert layout is not None
    assert 0.448 <= layout.input_box.top <= 0.464
    assert 0.568 <= layout.input_box.bottom <= 0.584

    ok, obs = verify_pasted_text(ocr_boxes, layout.input_box, expected_text)
    assert ok is True
    assert "17:15" not in obs
    assert "测试群友A" in obs
    assert "@示例助手" not in obs
    assert "门" not in obs

    # Duplicated 4-line paste must still be rejected
    dup_boxes = ocr_boxes + [OcrBox("问群主确认。", Box(0.430, 0.541, 0.690, 0.567))]
    ok_dup, _ = verify_pasted_text(dup_boxes, layout.input_box, expected_text)
    assert ok_dup is False


def test_verify_pasted_text_enforces_hard_newline_structure() -> None:
    pill = Box(0.132, 0.456, 0.764, 0.576)

    # Hard newline lost (merged onto a single visual row) -> MUST fail
    merged_single_row = [OcrBox("第一行第二行", Box(0.16, 0.48, 0.55, 0.505))]
    ok_merged, _ = verify_pasted_text(merged_single_row, pill, "第一行\n第二行")
    assert ok_merged is False

    # Hard newline shifted across visual rows -> MUST fail
    shifted_rows = [
        OcrBox("第一行第二行", Box(0.16, 0.47, 0.58, 0.495)),
        OcrBox("很长补充内容", Box(0.16, 0.505, 0.58, 0.530)),
    ]
    ok_shifted, _ = verify_pasted_text(shifted_rows, pill, "第一行\n第二行很长补充内容")
    assert ok_shifted is False

    # Hard newline preserved across two distinct visual rows -> MUST pass
    preserved_rows = [
        OcrBox("第一行", Box(0.16, 0.47, 0.36, 0.495)),
        OcrBox("第二行", Box(0.16, 0.505, 0.36, 0.530)),
    ]
    ok_preserved, _ = verify_pasted_text(preserved_rows, pill, "第一行\n第二行")
    assert ok_preserved is True

    # Logical line 1 soft-wraps across 2 visual rows + hard newline starts row 3 -> MUST pass
    soft_wrapped_with_newline = [
        OcrBox("这是一段需要自动换", Box(0.16, 0.468, 0.70, 0.490)),
        OcrBox("行的第一行", Box(0.16, 0.498, 0.45, 0.520)),
        OcrBox("第二行总结", Box(0.16, 0.528, 0.45, 0.550)),
    ]
    ok_soft, _ = verify_pasted_text(
        soft_wrapped_with_newline,
        pill,
        "这是一段需要自动换行的第一行\n第二行总结",
    )
    assert ok_soft is True


def test_header_title_unique_primary_box_strict_group_count_and_tight_outgoing_bubble() -> None:
    import pytest

    from wechat_agent.guards.target_policy import PolicyViolationError, TargetPolicy

    # 1. Stray status-bar/banner match at top < 0.045 must NOT pass when nav header shows another chat
    stray_banner = [
        OcrBox("示例测试群(12)", Box(0.28, 0.025, 0.72, 0.042)),
        OcrBox("无关聊天", Box(0.35, 0.06, 0.65, 0.09)),
    ]
    assert has_exact_chat_title(stray_banner, "示例测试群", allow_group_count=True) is False

    # 2. Ambiguous / duplicate nav-bar header boxes must be rejected
    dual_header = [
        OcrBox("示例测试群(12)", Box(0.18, 0.06, 0.48, 0.09)),
        OcrBox("提示横幅", Box(0.54, 0.06, 0.80, 0.09)),
    ]
    assert has_exact_chat_title(dual_header, "示例测试群", allow_group_count=True) is False

    # 3. Strict allow_group_count accepts 1..2000 with no leading zeros and rejects malformed suffixes
    for good in ("示例测试群(1)", "示例测试群(12)", "示例测试群（500）", "示例测试群(2000)"):
        assert has_exact_chat_title([OcrBox(good, Box(0.28, 0.06, 0.72, 0.09))], "示例测试群", allow_group_count=True)
    for bad in ("示例测试群(0)", "示例测试群(01)", "示例测试群(2001)", "示例测试群(99999)", "示例测试群(12人)"):
        assert (
            has_exact_chat_title([OcrBox(bad, Box(0.28, 0.06, 0.72, 0.09))], "示例测试群", allow_group_count=True)
            is False
        )

    # 4. Tightened has_new_outgoing_bubble rejects short substring match while allowing exact / long OCR truncation
    assert has_new_outgoing_bubble([], ["好的，马上处理"], "好的") is False
    assert has_new_outgoing_bubble([], ["好的"], "好的") is True
    assert has_new_outgoing_bubble(["好的"], ["好的", "好的"], "好的") is True
    assert (
        has_new_outgoing_bubble(
            [],
            ["请按群内官方团购链接下单即可"],
            "请按群内官方团购链接下单即可哦",
        )
        is True
    )
    assert (
        has_new_outgoing_bubble(
            [],
            ["补充说明：请按群内官方团购链接下单即可哦，有任何问题欢迎联系群主"],
            "请按群内官方团购链接下单即可哦",
        )
        is False
    )

    # 5. TargetPolicy rejects wildcard / control characters
    for bad_target in ("*", "all", "测试\n群"):
        with pytest.raises(PolicyViolationError):
            TargetPolicy([bad_target])


