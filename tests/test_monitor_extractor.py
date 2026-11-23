"""Offline unit tests for M2 group title matching, clipped-bubble filtering, role extraction, and lock screen."""

from tests.fakes import make_collapsed_chat_image
from wechat_agent.models import Box, OcrBox
from wechat_agent.monitor.extractor import (
    ChatMessageExtractor,
    ViewportAlignment,
    align_viewport_suffix,
    build_viewport_keys,
    classify_bubble_role,
    compute_event_fingerprint,
    compute_message_fingerprint,
    mentions_any_alias,
    merge_same_sender_bursts,
    message_viewport_key,
    normalize_group_title,
)
from wechat_agent.monitor.models import ExtractedMessage, MessageRole
from wechat_agent.vision.wechat_ui import extract_header_title, has_exact_chat_title, is_lock_screen


def test_group_title_matching_with_member_count() -> None:
    boxes = [OcrBox("产品讨论群(18)", Box(0.32, 0.06, 0.68, 0.09))]
    assert extract_header_title(boxes) == "产品讨论群(18)"
    assert normalize_group_title("产品讨论群(18)") == "产品讨论群"

    assert has_exact_chat_title(boxes, "产品讨论群", allow_group_count=True) is True
    assert has_exact_chat_title(boxes, "产品讨论群", allow_group_count=False) is False
    assert has_exact_chat_title(boxes, "产品讨论", allow_group_count=True) is False
    assert has_exact_chat_title(boxes, "产品讨论群2", allow_group_count=True) is False

    real_header = [OcrBox("示例分享·测试3群（198）（12）", Box(0.147, 0.082, 0.750, 0.104))]
    assert has_exact_chat_title(real_header, "示例分享•测试3群（198）", allow_group_count=True) is True
    assert has_exact_chat_title(real_header, "示例分享•测试3群（199）", allow_group_count=True) is False


def test_extractor_distinguishes_white_incoming_green_self_grey_system_and_ignores_clipped() -> None:
    # Top clipped white bubble touching viewport top (0.125..0.165), incoming white bubble,
    # incoming sender nickname on grey canvas, centered system time, and outgoing green bubble.
    clipped_white = Box(0.14, 0.125, 0.56, 0.168)
    incoming_white = Box(0.14, 0.255, 0.68, 0.345)
    outgoing_green = Box(0.46, 0.450, 0.88, 0.510)
    img = make_collapsed_chat_image(
        incoming_boxes=[clipped_white, incoming_white],
        outgoing_boxes=[outgoing_green],
    )

    ocr = [
        OcrBox("产品讨论群(18)", Box(0.32, 0.06, 0.68, 0.09)),
        # Top-clipped partial line (top=0.138 <= 0.145 + continuous white to 0.125)
        OcrBox("被裁切掉一半的旧消息", Box(0.16, 0.138, 0.52, 0.162)),
        # Centered grey system timestamp (must be filtered as SYSTEM)
        OcrBox("21:30", Box(0.44, 0.195, 0.56, 0.215)),
        # Sender nickname on grey canvas right above incoming white bubble
        OcrBox("张三", Box(0.145, 0.224, 0.22, 0.244)),
        # Multi-line incoming white bubble mentioning @运营助手
        OcrBox("@运营助手\u2005请问明天几点", Box(0.16, 0.268, 0.58, 0.294)),
        OcrBox("开始活动？", Box(0.16, 0.304, 0.36, 0.330)),
        # Centered grey system event
        OcrBox("李四加入了群聊", Box(0.32, 0.390, 0.68, 0.415)),
        # Right outgoing green bubble
        OcrBox("明天上午九点开始", Box(0.49, 0.465, 0.85, 0.495)),
    ]

    extractor = ChatMessageExtractor()
    msgs = extractor.extract(
        img,
        ocr,
        group_title="产品讨论群(18)",
        self_aliases=("运营助手",),
        captured_at=1000.0,
    )

    assert len(msgs) == 2
    assert msgs[0].role == MessageRole.INCOMING
    assert msgs[0].sender == "张三"
    assert msgs[0].text == "@运营助手\u2005请问明天几点\n开始活动？"
    assert msgs[0].mentioned_self is True
    assert msgs[1].role == MessageRole.SELF
    assert msgs[1].text == "明天上午九点开始"
    assert all("被裁切" not in m.text for m in msgs)
    assert all("21:30" not in m.text and "加入了群聊" not in m.text for m in msgs)


def test_mentions_and_fingerprint_stability() -> None:
    assert mentions_any_alias("＠运营助手 帮我看下", ("运营助手",)) is True
    assert mentions_any_alias("@助手AI", ("助手AI",)) is True
    assert mentions_any_alias("@助手Al", ("助手AI",)) is True
    assert mentions_any_alias("@助手AI请问营业时间", ("助手AI",)) is True
    assert mentions_any_alias("@助手AIHello请问", ("助手AI",)) is True
    assert mentions_any_alias("@助手AlHello请问", ("助手AI",)) is True
    assert mentions_any_alias("@助手AI，帮看下", ("助手AI",)) is True

    assert mentions_any_alias("@运营助手2 你好", ("运营助手",)) is False
    assert mentions_any_alias("@助手AI2 你好", ("助手AI",)) is False
    assert mentions_any_alias("@helperbot 你好", ("helper",)) is False
    assert mentions_any_alias("@助手AIbot 你好", ("助手AI",)) is False
    assert mentions_any_alias("@助手AI 你好", ("助手",)) is False

    fp1 = compute_message_fingerprint("示例分享•测试3群（198）", MessageRole.INCOMING, "张三", "hello  world", 1)
    fp2 = compute_message_fingerprint("示例分享·测试3群(198)", MessageRole.INCOMING, "张三", "hello world", 1)
    fp3 = compute_message_fingerprint("示例分享·测试3群(199)", MessageRole.INCOMING, "张三", "hello world", 1)
    fp4 = compute_message_fingerprint("示例分享•测试3群（198）", MessageRole.INCOMING, "张三", "hello world", 2)
    assert fp1 == fp2
    assert fp1 != fp3
    assert fp1 != fp4


def test_lock_screen_detection() -> None:
    import io

    from PIL import Image

    black = Image.new("RGB", (120, 260), (0, 0, 0))
    buf = io.BytesIO()
    black.save(buf, format="PNG")
    assert is_lock_screen([], buf.getvalue()) is True

    normal = make_collapsed_chat_image()
    assert is_lock_screen([OcrBox("文件传输助手", Box(0.35, 0.06, 0.65, 0.09))], normal) is False
    assert is_lock_screen([OcrBox("向上轻扫以解锁", Box(0.30, 0.90, 0.70, 0.93))], normal) is True
    assert is_lock_screen([OcrBox("21:45", Box(0.32, 0.14, 0.68, 0.22))], normal) is True


def test_align_viewport_suffix_and_event_fingerprints() -> None:
    a = ("incoming", "张三", "早上好")
    a_var = ("incoming", "张三", "早 上 好")  # placeholder variant for top-edge flicker
    b = ("self", "", "早呀")
    c = ("incoming", "", "在吗")
    d = ("incoming", "", "价格多少")
    x = ("incoming", "路人", "完全无关的消息1")
    y = ("incoming", "路人", "完全无关的消息2")

    # Empty edges
    assert align_viewport_suffix([a, b], []) == ViewportAlignment("empty_curr", 0, 0)
    assert align_viewport_suffix([], [a, b]) == ViewportAlignment("empty_prev", 0, 0)

    # Identical screen -> zero newly arrived
    assert align_viewport_suffix([a, b, c], [a, b, c]) == ViewportAlignment("aligned", 3, 3)

    # Appended tail
    assert align_viewport_suffix([a, b], [a, b, c]) == ViewportAlignment("aligned", 2, 2)

    # Scrolled overlap + repeated identical phrase ("在吗")
    assert align_viewport_suffix([a, b, c], [b, c, d]) == ViewportAlignment("aligned", 2, 2)
    assert align_viewport_suffix([a, c], [a, c, c]) == ViewportAlignment("aligned", 2, 2)
    assert align_viewport_suffix([c, b], [b, c]) == ViewportAlignment("aligned", 1, 1)

    # Top-edge OCR flicker tolerance: A' drifted slightly on line 0, B & C still match prev[-2:]
    assert align_viewport_suffix([a, b, c], [a_var, b, c, d]) == ViewportAlignment("aligned", 2, 3)

    # Scroll-up / keyboard-covered subset
    assert align_viewport_suffix([a, b, c], [a, b]) == ViewportAlignment("scroll_up", 2, 2)
    assert align_viewport_suffix([a, b, c], [x, a, b]) == ViewportAlignment("scroll_up", 2, 3)

    # Zero-overlap screen discontinuity
    assert align_viewport_suffix([a, b], [x, y]) == ViewportAlignment("discontinuity", 0, 2)

    # Helper key & independent event fingerprint across event_seq
    msg = ExtractedMessage(group_title="测试群(5)", role=MessageRole.INCOMING, sender="", text="在吗")
    assert message_viewport_key(msg) == ("incoming", "", "在吗", 1)
    ep1 = compute_event_fingerprint("示例分享•测试3群（198）", 1, MessageRole.INCOMING, "", "在吗")
    ep2 = compute_event_fingerprint("示例分享·测试3群(198)", 1, MessageRole.INCOMING, "", "在吗")
    ep3 = compute_event_fingerprint("示例分享·测试3群(199)", 1, MessageRole.INCOMING, "", "在吗")
    ep4 = compute_event_fingerprint("示例分享•测试3群（198）", 2, MessageRole.INCOMING, "", "在吗")
    assert ep1 == ep2
    assert ep1 != ep3
    assert ep1 != ep4


def test_extractor_merges_multiline_wrapped_self_green_bubble() -> None:
    b1 = Box(0.16, 0.170, 0.83, 0.245)
    b2 = Box(0.16, 0.268, 0.83, 0.312)
    img = make_collapsed_chat_image(outgoing_boxes=[b1, b2])
    ocr = [
        OcrBox("产品讨论群(18)", Box(0.32, 0.06, 0.68, 0.09)),
        OcrBox("iphoneagent自动操控iphone", Box(0.188, 0.182, 0.798, 0.206)),
        OcrBox("测试成功", Box(0.188, 0.208, 0.376, 0.231)),
        OcrBox("示例头像", Box(0.858, 0.185, 0.972, 0.203)),
        OcrBox("另一条单独回复", Box(0.188, 0.276, 0.560, 0.300)),
    ]
    msgs = ChatMessageExtractor().extract(img, ocr, group_title="产品讨论群(18)")
    assert [m.role for m in msgs] == [MessageRole.SELF, MessageRole.SELF]
    assert [m.text for m in msgs] == [
        "iphoneagent自动操控iphone\n测试成功",
        "另一条单独回复",
    ]


def test_extractor_ignores_sender_touching_white_sticker_and_filters_white_bg_sticker_caption() -> None:
    import io

    from PIL import Image, ImageDraw

    w, h = 480, 1040
    base = make_collapsed_chat_image(
        w=w,
        h=h,
        incoming_boxes=[
            Box(0.17, 0.250, 0.52, 0.410),  # white square sticker card
            Box(0.14, 0.480, 0.66, 0.550),  # genuine left-aligned incoming text bubble
        ],
        outgoing_boxes=[
            Box(0.46, 0.585, 0.88, 0.640),  # genuine outgoing green bubble separated by grey gap
        ],
    )
    img = Image.open(io.BytesIO(base)).convert("RGB")
    draw = ImageDraw.Draw(img)
    # Colorful cartoon illustration inside the lower half of the white square sticker
    draw.rectangle(
        [int(0.20 * w), int(0.31 * h), int(0.49 * w), int(0.40 * h)],
        fill=(220, 90, 60),
    )
    draw.rectangle(
        [int(0.24 * w), int(0.32 * h), int(0.40 * w), int(0.38 * h)],
        fill=(240, 190, 70),
    )
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    synthetic_bytes = buf.getvalue()

    ocr = [
        OcrBox("示例测试群(12)", Box(0.32, 0.06, 0.68, 0.09)),
        # Sender A sits just 2px (0.002) above the white sticker edge — must NOT leak white pixels
        OcrBox("测试群友A", Box(0.17, 0.228, 0.30, 0.248)),
        # Centered caption inside the white square sticker -> must be filtered out completely
        OcrBox("HAHA", Box(0.29, 0.262, 0.40, 0.290)),
        # Genuine incoming bubble + sender B directly above it (gap = 0.005)
        OcrBox("测试群友B", Box(0.145, 0.455, 0.28, 0.475)),
        OcrBox("@示例助手 请问今天有活动吗", Box(0.16, 0.495, 0.62, 0.530)),
        # Outgoing green bubble right below (must stay SELF and not falsely trigger sticker Rule B)
        OcrBox("有的，下午三点开始", Box(0.49, 0.598, 0.85, 0.628)),
    ]

    msgs = ChatMessageExtractor().extract(
        synthetic_bytes,
        ocr,
        group_title="示例测试群",
        self_aliases=("示例助手",),
        captured_at=1000.0,
    )
    assert [(m.role, m.sender, m.text, m.mentioned_self) for m in msgs] == [
        (MessageRole.INCOMING, "测试群友B", "@示例助手 请问今天有活动吗", True),
        (MessageRole.SELF, "", "有的，下午三点开始", False),
    ]


def test_merge_same_sender_bursts_behavior() -> None:
    g = "示例测试群"
    m1 = ExtractedMessage(
        g,
        MessageRole.INCOMING,
        "请问营业时间",
        sender="测试群友A",
        mentioned_self=False,
        fingerprint="f1",
        captured_at=100.0,
    )
    m2 = ExtractedMessage(
        g,
        MessageRole.INCOMING,
        "@示例助手",
        sender="测试群友A",
        mentioned_self=True,
        fingerprint="f2",
        captured_at=103.0,
    )
    m3 = ExtractedMessage(
        g,
        MessageRole.INCOMING,
        "有人在吗",
        sender="测试群友B",
        mentioned_self=False,
        fingerprint="f3",
        captured_at=104.0,
    )
    # > 5s gap must NOT merge even for same sender
    m4 = ExtractedMessage(
        g,
        MessageRole.INCOMING,
        "补充一句",
        sender="测试群友B",
        mentioned_self=False,
        fingerprint="f4",
        captured_at=112.0,
    )
    # Empty sender (hidden group nicknames) must NEVER merge across different users' bubbles
    e1 = ExtractedMessage(g, MessageRole.INCOMING, "在吗", sender="", fingerprint="e1", captured_at=120.0)
    e2 = ExtractedMessage(g, MessageRole.INCOMING, "价格多少", sender="", fingerprint="e2", captured_at=121.0)

    out = merge_same_sender_bursts([m1, m2, m3, m4, e1, e2], max_gap_seconds=5.0)
    assert [(x.sender, x.text, x.mentioned_self, x.fingerprint) for x in out] == [
        ("测试群友A", "请问营业时间\n@示例助手", True, "f1"),
        ("测试群友B", "有人在吗", False, "f3"),
        ("测试群友B", "补充一句", False, "f4"),
        ("", "在吗", False, "e1"),
        ("", "价格多少", False, "e2"),
    ]


def test_geometric_role_classification_self_white_card_incoming_green_image_and_centered_ambiguous() -> None:
    # Direct boundary checks on classify_bubble_role
    assert classify_bubble_role(0.16, 0.83, is_green=True, is_white_card=False) == MessageRole.SELF
    assert classify_bubble_role(0.28, 0.86, is_green=False, is_white_card=True) == MessageRole.SELF
    assert classify_bubble_role(0.14, 0.68, is_green=False, is_white_card=True) == MessageRole.INCOMING
    # Incoming green image on the left must NEVER be misclassified as SELF (prevents false 600s takeover)
    assert classify_bubble_role(0.14, 0.62, is_green=True, is_white_card=False) == MessageRole.UNKNOWN
    # Centered ambiguous white/green card must be UNKNOWN and dropped
    assert classify_bubble_role(0.26, 0.74, is_green=False, is_white_card=True) == MessageRole.UNKNOWN
    assert classify_bubble_role(0.26, 0.74, is_green=True, is_white_card=False) == MessageRole.UNKNOWN
    # Right avatar region (> 0.935) must be UNKNOWN
    assert classify_bubble_role(0.86, 0.97, is_green=True, is_white_card=False) == MessageRole.UNKNOWN

    # End-to-end screenshot extraction with:
    # 1. Right-aligned SELF white link/mini-program card (must be SELF, never INCOMING self-loop)
    # 2. Left-aligned INCOMING green-background poster/card (must be UNKNOWN & dropped, never SELF takeover)
    # 3. Centered ambiguous white card (must be UNKNOWN & dropped)
    # 4. Genuine left-aligned INCOMING white text bubble
    self_white_card = Box(0.28, 0.190, 0.86, 0.280)
    incoming_green_poster = Box(0.14, 0.320, 0.62, 0.420)
    centered_card = Box(0.26, 0.460, 0.74, 0.540)
    real_incoming = Box(0.14, 0.590, 0.66, 0.660)

    img = make_collapsed_chat_image(
        incoming_boxes=[self_white_card, centered_card, real_incoming],
        outgoing_boxes=[incoming_green_poster],
    )
    ocr = [
        OcrBox("示例测试群(12)", Box(0.32, 0.06, 0.68, 0.09)),
        OcrBox("官方商品链接卡片", Box(0.31, 0.205, 0.68, 0.230)),
        OcrBox("点击查看详情", Box(0.31, 0.238, 0.54, 0.262)),
        OcrBox("测试群友A", Box(0.145, 0.295, 0.28, 0.315)),
        OcrBox("限时绿色海报文字", Box(0.18, 0.350, 0.56, 0.385)),
        OcrBox("居中模糊卡片", Box(0.36, 0.485, 0.64, 0.515)),
        OcrBox("测试群友B", Box(0.145, 0.562, 0.28, 0.582)),
        OcrBox("请问这个还有库存吗", Box(0.16, 0.608, 0.60, 0.640)),
    ]

    msgs = ChatMessageExtractor().extract(img, ocr, group_title="示例测试群(12)", captured_at=1000.0)
    assert [(m.role, m.sender, m.text) for m in msgs] == [
        (MessageRole.SELF, "", "官方商品链接卡片\n点击查看详情"),
        (MessageRole.INCOMING, "测试群友B", "请问这个还有库存吗"),
    ]


def test_viewport_alignment_with_occurrences_and_repeated_short_messages() -> None:
    # Consecutive identical short messages from the same sender carry distinct occurrences
    b1 = Box(0.14, 0.22, 0.34, 0.28)
    b2 = Box(0.14, 0.34, 0.34, 0.40)
    img = make_collapsed_chat_image(incoming_boxes=[b1, b2])
    ocr = [
        OcrBox("示例测试群(12)", Box(0.32, 0.06, 0.68, 0.09)),
        OcrBox("测试群友A", Box(0.145, 0.195, 0.28, 0.215)),
        OcrBox("1", Box(0.17, 0.235, 0.24, 0.265)),
        OcrBox("测试群友A", Box(0.145, 0.315, 0.28, 0.335)),
        OcrBox("1", Box(0.17, 0.355, 0.24, 0.385)),
    ]
    msgs = ChatMessageExtractor().extract(img, ocr, group_title="示例测试群(12)", captured_at=1000.0)
    assert [m.occurrence for m in msgs] == [1, 2]
    keys = build_viewport_keys(msgs)
    assert keys == [message_viewport_key(m) for m in msgs]
    assert keys == [
        ("incoming", "测试群友A", "1", 1),
        ("incoming", "测试群友A", "1", 2),
    ]

    # Third identical "1" arrives -> overlap=2, new_start=2
    curr3 = keys + [("incoming", "测试群友A", "1", 3)]
    assert align_viewport_suffix(keys, curr3) == ViewportAlignment("aligned", 2, 2)

    # First "1" scrolls off top, leaving the remaining "1" (now occurrence=1 in new frame) + new message
    curr_scrolled = [
        ("incoming", "测试群友A", "1", 1),
        ("incoming", "测试群友A", "收到", 1),
    ]
    assert align_viewport_suffix(keys, curr_scrolled) == ViewportAlignment("aligned", 1, 1)


def test_full_width_incoming_assistivetouch_wide_self_card_and_internal_quote() -> None:
    import io

    from PIL import Image, ImageDraw

    from wechat_agent.monitor.extractor import _decode_rgb, _estimate_canvas_luminance, _measure_bubble_horizontal_span

    w, h = 500, 1080
    # 1. Full-width 2-line INCOMING white bubble [0.166, 0.828] touching right AssistiveTouch ball [0.828, 0.976]
    incoming_touching_ball = Box(0.166, 0.190, 0.828, 0.275)
    # 2. Full-width SELF white quote/link card [0.185, 0.845] with right-side avatar (must be SELF, never INCOMING!)
    wide_self_white_card = Box(0.185, 0.340, 0.845, 0.460)
    # 3. Right-aligned SELF white card [0.240, 0.845] containing internal recessed grey quote block (#E5E5E5)
    self_card_with_quote = Box(0.240, 0.510, 0.845, 0.600)
    # 4. Full-width INCOMING white bubble [0.166, 0.828] with left avatar & internal light-grey subtext/quote (#F2F2F2)
    incoming_with_quote = Box(0.166, 0.660, 0.828, 0.760)

    base = make_collapsed_chat_image(
        w=w,
        h=h,
        incoming_boxes=[
            incoming_touching_ball,
            wide_self_white_card,
            self_card_with_quote,
            incoming_with_quote,
        ],
    )
    img = Image.open(io.BytesIO(base)).convert("RGB")
    draw = ImageDraw.Draw(img)

    # Left colorful avatars for bubbles #1 and #4
    draw.rectangle([int(0.03 * w), int(0.165 * h), int(0.135 * w), int(0.215 * h)], fill=(65, 135, 225))
    draw.rectangle([int(0.03 * w), int(0.635 * h), int(0.135 * w), int(0.685 * h)], fill=(220, 110, 60))
    # Right colorful avatars for SELF white cards #2 and #3
    draw.rectangle([int(0.865 * w), int(0.340 * h), int(0.970 * w), int(0.390 * h)], fill=(70, 175, 110))
    draw.rectangle([int(0.865 * w), int(0.510 * h), int(0.970 * w), int(0.560 * h)], fill=(70, 175, 110))

    # Translucent dark-grey AssistiveTouch floating ball docked on right edge touching bubble #1 (0.828..0.976)
    draw.rectangle([int(0.828 * w), int(0.195 * h), int(0.846 * w), int(0.270 * h)], fill=(227, 226, 228))
    draw.rectangle([int(0.846 * w), int(0.195 * h), int(0.960 * w), int(0.270 * h)], fill=(188, 186, 189))
    draw.rectangle([int(0.960 * w), int(0.195 * h), int(0.976 * w), int(0.270 * h)], fill=(226, 224, 227))

    # Internal recessed grey quote blocks inside cards #3 (#E5E5E5) and #4 (#F2F2F2), keeping outer white margins
    draw.rectangle([int(0.420 * w), int(0.525 * h), int(0.805 * w), int(0.585 * h)], fill=(229, 229, 229))
    draw.rectangle([int(0.450 * w), int(0.675 * h), int(0.790 * w), int(0.745 * h)], fill=(242, 242, 242))

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    raw_bytes = buf.getvalue()

    rgb = _decode_rgb(raw_bytes)
    assert rgb is not None
    canvas_lum = _estimate_canvas_luminance(rgb, 0.875)

    # Verify AssistiveTouch floating ball is cleanly truncated at ~0.828 (never bleeds to 0.96+)
    span1 = _measure_bubble_horizontal_span(rgb, Box(0.206, 0.205, 0.790, 0.228), canvas_lum, is_green=False)
    assert 0.160 <= span1[0] <= 0.172
    assert 0.820 <= span1[1] <= 0.834

    # Verify internal grey quote block (#E5E5E5) inside SELF white card does NOT prematurely truncate right edge
    span3 = _measure_bubble_horizontal_span(rgb, Box(0.260, 0.535, 0.410, 0.565), canvas_lum, is_green=False)
    assert 0.235 <= span3[0] <= 0.245
    assert 0.840 <= span3[1] <= 0.850

    # Verify internal light-grey quote block (#F2F2F2) inside INCOMING white bubble does NOT truncate right edge
    span4 = _measure_bubble_horizontal_span(rgb, Box(0.200, 0.690, 0.440, 0.720), canvas_lum, is_green=False)
    assert 0.160 <= span4[0] <= 0.172
    assert 0.822 <= span4[1] <= 0.834

    ocr = [
        OcrBox("示例测试群(12)", Box(0.28, 0.069, 0.72, 0.094)),
        OcrBox("测试群友A", Box(0.170, 0.160, 0.280, 0.180)),
        OcrBox("@示例助手 这款怎么团购下", Box(0.206, 0.205, 0.790, 0.228)),
        OcrBox("单？", Box(0.204, 0.234, 0.284, 0.258)),
        # Wide SELF white card (2 lines, line 2 below avatar -> must still resolve to SELF via bubble_top!)
        OcrBox("官方商品链接详情标题", Box(0.215, 0.360, 0.680, 0.388)),
        OcrBox("点击卡片查看团购规则说明", Box(0.215, 0.415, 0.700, 0.442)),
        # Right-aligned SELF white card with internal grey quote block
        OcrBox("已核对活动链接", Box(0.260, 0.535, 0.410, 0.565)),
        # Full-width INCOMING white bubble with internal light-grey quote block
        OcrBox("测试群友B", Box(0.170, 0.630, 0.280, 0.650)),
        OcrBox("请问今天还发货吗？", Box(0.200, 0.690, 0.440, 0.720)),
    ]

    msgs = ChatMessageExtractor().extract(
        raw_bytes,
        ocr,
        group_title="示例测试群(12)",
        self_aliases=("示例助手",),
        captured_at=1000.0,
    )
    assert [(m.role, m.sender, m.text, m.mentioned_self) for m in msgs] == [
        (MessageRole.INCOMING, "测试群友A", "@示例助手 这款怎么团购下\n单？", True),
        (MessageRole.SELF, "", "官方商品链接详情标题\n点击卡片查看团购规则说明", False),
        (MessageRole.SELF, "", "已核对活动链接", False),
        (MessageRole.INCOMING, "测试群友B", "请问今天还发货吗？", False),
    ]
