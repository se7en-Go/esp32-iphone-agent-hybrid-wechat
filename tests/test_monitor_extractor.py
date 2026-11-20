"""Offline unit tests for M2 group title matching, clipped-bubble filtering, role extraction, and lock screen."""

from tests.fakes import make_collapsed_chat_image
from wechat_agent.models import Box, OcrBox
from wechat_agent.monitor.extractor import (
    ChatMessageExtractor,
    ViewportAlignment,
    align_viewport_suffix,
    compute_event_fingerprint,
    compute_message_fingerprint,
    mentions_any_alias,
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
    from PIL import Image
    import io

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
    assert message_viewport_key(msg) == ("incoming", "", "在吗")
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
