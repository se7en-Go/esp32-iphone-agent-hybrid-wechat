"""WeChat group chat message extraction with bubble-color role detection and clipped-bubble filtering."""

from __future__ import annotations

import hashlib
import io
import re
import time
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from PIL import Image

from wechat_agent.models import Box, OcrBox
from wechat_agent.vision.geometry import canonicalize_chat_title, normalize_text
from wechat_agent.vision.wechat_ui import has_continuous_green_bridge

from .models import ExtractedMessage, MessageRole

ViewportItem = tuple[str, str, str, int]
LegacyViewportItem = tuple[str, str, str]


@dataclass(frozen=True)
class ViewportAlignment:
    """Result of aligning the current viewport message sequence with the previous viewport tail."""

    kind: str  # "empty_curr" | "empty_prev" | "aligned" | "scroll_up" | "discontinuity"
    overlap_len: int
    new_start: int


_VIEWPORT_TOP = 0.125
_TOP_SAFE_TEXT_Y = 0.145
_SELF_MIN_RIGHT_EDGE = 0.80
_SELF_MAX_RIGHT_EDGE = 0.935
_SELF_MIN_LEFT_EDGE = 0.13
_SELF_WHITE_CARD_MIN_LEFT = 0.22
_INCOMING_MIN_LEFT_EDGE = 0.10
_INCOMING_MAX_LEFT_EDGE = 0.215
_INCOMING_MAX_RIGHT_EDGE = 0.850
_INCOMING_NARROW_MAX_RIGHT_EDGE = 0.799
_INCOMING_FULL_WIDTH_MAX_LEFT = 0.185
_INCOMING_FULL_WIDTH_MAX_RIGHT = 0.840
_GROUP_COUNT_RE = re.compile(r"\s*\(\d+\)$")
_SYSTEM_MSG_RE = re.compile(
    r"^(?:"
    r"\d{1,2}:\d{2}"
    r"|(?:昨天|前天|今天)\s*\d{1,2}:\d{2}"
    r"|(?:星期|周)[一二三四五六日天]\s*\d{1,2}:\d{2}"
    r"|\d{4}年\d{1,2}月\d{1,2}日.*"
    r"|.*(?:加入了群聊|邀请.+加入了群聊|撤回了一条消息|拍了拍|修改群名为|开启了群聊邀请确认|以下为新消息).*"
    r")$"
)


def normalize_group_title(title: str) -> str:
    """Canonicalize a raw WeChat group header title, stripping one trailing `(N)` live member count."""
    return _GROUP_COUNT_RE.sub("", canonicalize_chat_title(title)).strip()


def _canonical_for_hash(text: str) -> str:
    return re.sub(r"\s+", " ", normalize_text(text))


def compute_message_fingerprint(
    group_title: str,
    role: MessageRole,
    sender: str,
    text: str,
    occurrence: int = 1,
) -> str:
    """Compute a deterministic cross-restart fingerprint for a visible message bubble."""
    norm_group = canonicalize_chat_title(group_title)
    norm_sender = normalize_text(sender)
    canon_text = _canonical_for_hash(text)
    payload = f"{norm_group}\x1f{role.value}\x1f{norm_sender}\x1f{canon_text}\x1f{occurrence}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


def message_viewport_key(msg: ExtractedMessage, occurrence: int | None = None) -> ViewportItem:
    """Build the ordered canonical tuple (including per-viewport occurrence) for alignment."""
    occ = int(msg.occurrence if occurrence is None else occurrence)
    return (msg.role.value, normalize_text(msg.sender), _canonical_for_hash(msg.text), max(1, occ))


def build_viewport_keys(messages: Sequence[ExtractedMessage]) -> list[ViewportItem]:
    """Compute ordered ViewportItems with per-viewport occurrence counts for repeated identical texts."""
    counts: Counter[tuple[str, str, str]] = Counter()
    keys: list[ViewportItem] = []
    for msg in messages:
        base = (msg.role.value, normalize_text(msg.sender), _canonical_for_hash(msg.text))
        counts[base] += 1
        keys.append((base[0], base[1], base[2], counts[base]))
    return keys


def _coerce_viewport_items(items: Sequence[ViewportItem | LegacyViewportItem]) -> list[ViewportItem]:
    counts: Counter[tuple[str, str, str]] = Counter()
    out: list[ViewportItem] = []
    for item in items:
        if len(item) == 4:
            r, s, t, occ = item
            out.append((str(r), str(s), str(t), max(1, int(occ))))
        else:
            r, s, t = item
            base = (str(r), str(s), str(t))
            counts[base] += 1
            out.append((base[0], base[1], base[2], counts[base]))
    return out


def _adjust_window_occurrences(
    prefix_scrolled: Sequence[ViewportItem],
    window: Sequence[ViewportItem],
) -> list[ViewportItem]:
    """Shift window occurrences downward by the exact count of scrolled-off identical items."""
    scrolled_counts = Counter((r, s, t) for r, s, t, _ in prefix_scrolled)
    return [(r, s, t, max(1, occ - scrolled_counts[(r, s, t)])) for r, s, t, occ in window]


def _viewport_slice_matches(
    prev: Sequence[ViewportItem],
    prev_start: int,
    curr_slice: Sequence[ViewportItem],
) -> bool:
    prev_slice = prev[prev_start : prev_start + len(curr_slice)]
    if len(prev_slice) != len(curr_slice):
        return False
    if list(prev_slice) == list(curr_slice):
        return True
    if prev_start > 0:
        adjusted = _adjust_window_occurrences(prev[:prev_start], prev_slice)
        return adjusted == list(curr_slice)
    return False


def compute_event_fingerprint(
    group_title: str,
    event_seq: int,
    role: MessageRole,
    sender: str,
    text: str,
) -> str:
    """Compute an independent idempotency fingerprint for a newly arrived message event."""
    norm_group = canonicalize_chat_title(group_title)
    norm_sender = normalize_text(sender)
    canon_text = _canonical_for_hash(text)
    payload = f"evt\x1f{norm_group}\x1f{int(event_seq)}\x1f{role.value}\x1f{norm_sender}\x1f{canon_text}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


def align_viewport_suffix(
    prev_items: Sequence[ViewportItem | LegacyViewportItem],
    curr_items: Sequence[ViewportItem | LegacyViewportItem],
) -> ViewportAlignment:
    """Align current visible viewport items against the tail of the previous viewport."""
    prev = _coerce_viewport_items(prev_items)
    curr = _coerce_viewport_items(curr_items)
    if not curr:
        return ViewportAlignment("empty_curr", 0, 0)
    if not prev:
        return ViewportAlignment("empty_prev", 0, 0)

    # 1. Primary longest suffix-prefix overlap: prev[-k:] == curr[:k]
    for k in range(min(len(prev), len(curr)), 0, -1):
        if _viewport_slice_matches(prev, len(prev) - k, curr[:k]):
            return ViewportAlignment("aligned", k, k)

    # 2. Top-edge OCR flicker tolerance: skip leading items if prev[-k:] == curr[j : j + k] with k >= 2
    if len(prev) >= 2 and len(curr) >= 2:
        for j in range(1, len(curr) - 1):
            for k in range(min(len(prev), len(curr) - j), 1, -1):
                if _viewport_slice_matches(prev, len(prev) - k, curr[j : j + k]):
                    return ViewportAlignment("aligned", k, j + k)

    # 3. Scroll-up / keyboard-obscured subset detection (never lose viewport tail anchor)
    if len(curr) < len(prev):
        for i in range(len(prev) - len(curr) + 1):
            if _viewport_slice_matches(prev, i, curr):
                return ViewportAlignment("scroll_up", len(curr), len(curr))

    for m in range(min(len(prev), len(curr)), 1, -1):
        curr_tail = _adjust_window_occurrences(curr[:-m], curr[-m:])
        if list(prev[:m]) == list(curr[-m:]) or list(prev[:m]) == curr_tail:
            return ViewportAlignment("scroll_up", m, len(curr))

    return ViewportAlignment("discontinuity", 0, len(curr))


def merge_same_sender_bursts(
    messages: Sequence[ExtractedMessage],
    *,
    max_gap_seconds: float = 5.0,
) -> list[ExtractedMessage]:
    """Merge adjacent INCOMING messages from the same non-empty sender within `max_gap_seconds`."""
    merged: list[ExtractedMessage] = []
    for msg in messages:
        if not merged or msg.role != MessageRole.INCOMING or merged[-1].role != MessageRole.INCOMING:
            merged.append(msg)
            continue
        prev = merged[-1]
        prev_sender = normalize_text(prev.sender)
        curr_sender = normalize_text(msg.sender)
        gap = abs(float(msg.captured_at) - float(prev.captured_at))
        if prev_sender and prev_sender == curr_sender and gap <= max_gap_seconds:
            combined_text = f"{prev.text}\n{msg.text}" if prev.text and msg.text else (prev.text or msg.text)
            merged[-1] = ExtractedMessage(
                group_title=prev.group_title,
                role=MessageRole.INCOMING,
                text=combined_text,
                sender=prev.sender,
                box=msg.box or prev.box,
                mentioned_self=prev.mentioned_self or msg.mentioned_self,
                fingerprint=prev.fingerprint,
                captured_at=max(float(prev.captured_at), float(msg.captured_at)),
            )
        else:
            merged.append(msg)
    return merged


def _alias_ocr_pattern(alias: str) -> str:
    """Build regex for `alias`, tolerating sans-serif `I`/`l` OCR confusion."""
    return "".join("[Il]" if ch in ("I", "l") else re.escape(ch) for ch in alias)


def mentions_any_alias(text: str, self_aliases: tuple[str, ...] | list[str]) -> bool:
    """Return True if `text` contains `@alias` (supports `＠`, dropped `\\u2005`, and `I`/`l` OCR confusion)."""
    norm_text = normalize_text(text)
    if not norm_text or not self_aliases:
        return False
    for raw_alias in self_aliases:
        alias = normalize_text(raw_alias)
        if not alias:
            continue
        pattern = re.compile(
            r"[@＠]"
            + _alias_ocr_pattern(alias)
            + r"(?:$|[\s\u2005,，。!！?？:：;；、)~～]|[\u4e00-\u9fff]|[A-Z][a-z])"
        )
        if pattern.search(norm_text):
            return True
    return False


def _decode_rgb(image_bytes: bytes) -> np.ndarray | None:
    if not image_bytes:
        return None
    try:
        with Image.open(io.BytesIO(image_bytes)) as img:
            return np.asarray(img.convert("RGB"))
    except (OSError, ValueError):
        return None


def _norm_box_to_pixels(box: Box, width: int, height: int) -> tuple[int, int, int, int]:
    x0 = max(0, min(width - 1, int(round(box.left * width))))
    y0 = max(0, min(height - 1, int(round(box.top * height))))
    x1 = max(x0 + 1, min(width, int(round(box.right * width))))
    y1 = max(y0 + 1, min(height, int(round(box.bottom * height))))
    return x0, y0, x1, y1


def _estimate_canvas_luminance(rgb: np.ndarray, bottom_limit: float) -> float:
    h, w, _ = rgb.shape
    y0 = max(0, int(round(_VIEWPORT_TOP * h)))
    y1 = max(y0 + 1, int(round(bottom_limit * h)))
    left_strip = rgb[y0:y1, 0 : max(2, int(0.03 * w))]
    right_strip = rgb[y0:y1, min(w - 2, int(0.97 * w)) : w]
    samples = np.concatenate([left_strip.reshape(-1, 3), right_strip.reshape(-1, 3)], axis=0)
    return float(np.median(samples.mean(axis=1))) if samples.size else 237.0


def _bubble_ratios(rgb: np.ndarray, box: Box, canvas_lum: float) -> tuple[float, float]:
    """Return `(green_ratio, incoming_white_ratio)` measured tightly within `box` (no vertical bleed)."""
    h, w, _ = rgb.shape
    probe = Box.clamped(
        max(0.0, box.left - 0.004),
        max(_VIEWPORT_TOP, box.top),
        min(1.0, box.right + 0.004),
        min(1.0, box.bottom),
    )
    x0, y0, x1, y1 = _norm_box_to_pixels(probe, w, h)
    patch = rgb[y0:y1, x0:x1].astype(np.int16)
    if patch.size == 0:
        return 0.0, 0.0
    green_mask = (patch[:, :, 1] - patch[:, :, 0] >= 12) & (patch[:, :, 1] - patch[:, :, 2] >= 12)
    green_ratio = float(green_mask.mean())

    if canvas_lum < 90.0:
        # Dark mode incoming bubble is a raised card brighter than dark canvas, without strong green tint.
        lum = patch.mean(axis=2)
        white_mask = (lum >= canvas_lum + 8.0) & (~green_mask)
    else:
        white_thresh = max(246, int(round(canvas_lum + 6)))
        white_mask = (
            (patch[:, :, 0] >= white_thresh)
            & (patch[:, :, 1] >= white_thresh)
            & (patch[:, :, 2] >= white_thresh)
            & (~green_mask)
        )
    return green_ratio, float(white_mask.mean())


def _has_white_above(rgb: np.ndarray, box: Box, canvas_lum: float) -> bool:
    """Confirm the thin band immediately above `box.top` is also inside the white/raised bubble."""
    h, w, _ = rgb.shape
    strip = Box.clamped(
        box.left,
        max(_VIEWPORT_TOP, box.top - 0.005),
        box.right,
        box.top,
    )
    x0, y0, x1, y1 = _norm_box_to_pixels(strip, w, h)
    if y1 <= y0:
        return False
    patch = rgb[y0:y1, x0:x1].astype(np.int16)
    green_mask = (patch[:, :, 1] - patch[:, :, 0] >= 12) & (patch[:, :, 1] - patch[:, :, 2] >= 12)
    if canvas_lum < 90.0:
        above_mask = (patch.mean(axis=2) >= canvas_lum + 8.0) & (~green_mask)
    else:
        white_thresh = max(246, int(round(canvas_lum + 6)))
        above_mask = (
            (patch[:, :, 0] >= white_thresh)
            & (patch[:, :, 1] >= white_thresh)
            & (patch[:, :, 2] >= white_thresh)
            & (~green_mask)
        )
    return float(above_mask.mean()) >= 0.40


def _is_white_sticker_caption(
    rgb: np.ndarray,
    box: Box,
    canvas_lum: float,
    bottom_limit: float,
) -> bool:
    """Detect zero-margin or centered decorative captions inside white-background image stickers."""
    if canvas_lum < 90.0:
        return False
    h, w, _ = rgb.shape
    x0, y0, x1, y1 = _norm_box_to_pixels(box, w, h)
    white_thresh = max(246, int(round(canvas_lum + 6)))

    band = rgb[y0:y1, :, :].astype(np.int16)
    if band.size == 0:
        return False
    col_white = (
        (band[:, :, 0] >= white_thresh)
        & (band[:, :, 1] >= white_thresh)
        & (band[:, :, 2] >= white_thresh)
    ).mean(axis=0)

    run_left = x0
    while run_left > 0 and col_white[run_left - 1] >= 0.45:
        run_left -= 1
    run_right = x1
    while run_right < w and col_white[run_right] >= 0.45:
        run_right += 1

    pad_l = (x0 - run_left) / w
    pad_r = (run_right - x1) / w
    # Rule A: centered caption inside a wide white sticker card (real chat bubbles left-align text).
    if pad_l >= 0.075 and pad_r >= 0.05:
        return True

    # Rule B: colorful illustration directly attached below the text inside the same card (no grey canvas gap).
    if run_right <= run_left + 4:
        return False
    scan_end = min(int(round(bottom_limit * h)), y1 + int(round(0.14 * h)))
    if scan_end <= y1:
        return False
    sub = rgb[y1:scan_end, run_left:run_right].astype(np.int16)
    lum = sub.mean(axis=2)
    sat = sub.max(axis=2) - sub.min(axis=2)
    canvas_rows = ((np.abs(lum - canvas_lum) <= 6.0) & (sat <= 12)).mean(axis=1) >= 0.70
    hit_canvas = np.where(canvas_rows)[0]
    card_rows = int(hit_canvas[0]) if hit_canvas.size > 0 else sub.shape[0]
    if card_rows < max(12, int(round(0.055 * h))):
        return False
    colorful_frac = float((sat[:card_rows] >= 28).mean())
    return colorful_frac >= 0.14


def _is_top_clipped_bubble(rgb: np.ndarray | None, box: Box, canvas_lum: float) -> bool:
    """Detect bubbles clipped by the top header bar so partial scrolled-off messages are ignored."""
    if box.top <= _TOP_SAFE_TEXT_Y:
        return True
    if rgb is None or box.top > 0.195:
        return False
    h, w, _ = rgb.shape
    # Sample from the chat viewport top edge down to the OCR box top to see if bubble fill touches the header.
    strip = Box.clamped(box.left, _VIEWPORT_TOP, box.right, box.top)
    x0, y0, x1, y1 = _norm_box_to_pixels(strip, w, h)
    if y1 <= y0 + 1:
        return True
    patch = rgb[y0:y1, x0:x1].astype(np.int16)
    green_mask = (patch[:, :, 1] - patch[:, :, 0] >= 12) & (patch[:, :, 1] - patch[:, :, 2] >= 12)
    if canvas_lum < 90.0:
        bubble_mask = green_mask | (patch.mean(axis=2) >= canvas_lum + 8.0)
    else:
        white_thresh = max(246, int(round(canvas_lum + 6)))
        bubble_mask = green_mask | (
            (patch[:, :, 0] >= white_thresh)
            & (patch[:, :, 1] >= white_thresh)
            & (patch[:, :, 2] >= white_thresh)
        )
    row_bubble_frac = bubble_mask.mean(axis=1)
    # If the very top rows against the header are already bubble-colored and stay continuous down to the text,
    # this bubble's upper section is clipped off-screen.
    return bool((row_bubble_frac[:2].mean() >= 0.30) and (row_bubble_frac.min() >= 0.25))


def _measure_bubble_horizontal_span(
    rgb: np.ndarray,
    box: Box,
    canvas_lum: float,
    *,
    is_green: bool = False,
) -> tuple[float, float]:
    """Measure the horizontal extent `(left, right)` of the bubble containing `box` by surface cues.

    Uses surface-matched columns (green for green bubbles; white/raised or colorful card pixels
    for white cards) with interior lookahead so:
    - Internal light-grey subtext or recessed grey quote blocks (`#E5E5E5` / `#F2F2F2`) inside a
      white bubble/card are bridged cleanly without premature truncation whenever bubble surface
      resumes at the card border.
    - External grey chat canvas (`|lum - canvas_lum| <= 5`, `sat <= 12`) and right-docked
      translucent grey iOS AssistiveTouch floating balls (`lum < canvas_lum - 20`, `sat <= 14`,
      with no white/colorful card surface beyond them) terminate the scan accurately at the true
      bubble edge.
    """
    h, w, _ = rgb.shape
    x0, y0, x1, y1 = _norm_box_to_pixels(box, w, h)
    band = rgb[y0:y1, :, :].astype(np.int16)
    if band.size == 0 or w <= 0:
        return box.left, box.right
    lum = band.mean(axis=2)
    sat = band.max(axis=2) - band.min(axis=2)
    col_canvas = ((np.abs(lum - canvas_lum) <= 3.5) & (sat <= 10)).mean(axis=0)

    green_mask = (band[:, :, 1] - band[:, :, 0] >= 12) & (band[:, :, 1] - band[:, :, 2] >= 12)
    col_green = green_mask.mean(axis=0)

    if canvas_lum < 90.0:
        col_white = ((lum >= canvas_lum + 8.0) & (~green_mask)).mean(axis=0)
        col_dark_barrier = np.zeros(w, dtype=np.float32)
    else:
        white_thresh = max(246, int(round(canvas_lum + 6)))
        col_white = (
            (band[:, :, 0] >= white_thresh)
            & (band[:, :, 1] >= white_thresh)
            & (band[:, :, 2] >= white_thresh)
            & (~green_mask)
        ).mean(axis=0)
        # Smooth dark-grey translucent AssistiveTouch outer ring in light mode (zero white/color).
        col_dark_barrier = ((lum < canvas_lum - 20.0) & (sat <= 14)).mean(axis=0)

    col_colorful = (sat >= 20).mean(axis=0)

    if is_green:
        col_surface = col_green >= 0.18
    else:
        col_surface = (col_white >= 0.15) | (col_colorful >= 0.20)

    def _scan(start: int, step: int, limit: int) -> int:
        last_surface = start
        c = start
        while c > limit if step < 0 else c < limit:
            idx = c - 1 if step < 0 else c
            if col_canvas[idx] >= 0.65:
                break
            in_outer_dock = (idx / w >= 0.80) or (idx / w <= 0.18)
            if in_outer_dock and col_dark_barrier[idx] >= 0.55 and not col_surface[idx]:
                break
            if col_surface[idx]:
                last_surface = idx if step < 0 else idx + 1
            c += step
        return last_surface

    run_left = _scan(x0, -1, 0)
    run_right = _scan(x1, 1, w)
    return run_left / w, run_right / w


def _measure_bubble_top(
    rgb: np.ndarray,
    box: Box,
    canvas_lum: float,
    *,
    is_green: bool = False,
) -> float:
    """Scan upward from `box.top` through contiguous bubble surface to find the container's top edge."""
    h, w, _ = rgb.shape
    x0, y0, x1, _ = _norm_box_to_pixels(box, w, h)
    top_limit = max(0, int(round(_VIEWPORT_TOP * h)))
    if y0 <= top_limit or x1 <= x0:
        return box.top
    strip = rgb[top_limit:y0, x0:x1].astype(np.int16)
    if strip.size == 0:
        return box.top
    lum = strip.mean(axis=2)
    sat = strip.max(axis=2) - strip.min(axis=2)
    row_canvas = ((np.abs(lum - canvas_lum) <= 5.0) & (sat <= 12)).mean(axis=1)
    row = y0 - 1
    while row >= top_limit:
        if row_canvas[row - top_limit] >= 0.65:
            break
        row -= 1
    return (row + 1) / h


def _detect_avatar_side(
    rgb: np.ndarray,
    bubble_top: float,
    canvas_lum: float,
    bottom_limit: float,
) -> str:
    """Detect whether an avatar sits in the left or right avatar column near `bubble_top`.

    Ignores smooth low-saturation AssistiveTouch translucent grey (`lum` ~ 180..228, `sat <= 14`)
    by requiring color saturation (`sat >= 16`) or strong contrast vs canvas.
    Returns `"left"`, `"right"`, `"both"`, or `"none"`.
    """
    h, w, _ = rgb.shape
    y_l0 = max(0, int(round(max(_VIEWPORT_TOP, bubble_top - 0.035) * h)))
    y_l1 = min(h, int(round(min(bottom_limit, bubble_top + 0.038) * h)))
    y_r0 = max(0, int(round(max(_VIEWPORT_TOP, bubble_top - 0.015) * h)))
    y_r1 = min(h, int(round(min(bottom_limit, bubble_top + 0.048) * h)))

    def _has_avatar(y0: int, y1: int, xa: float, xb: float) -> bool:
        if y1 <= y0 + 2:
            return False
        x0 = max(0, int(round(xa * w)))
        x1 = min(w, int(round(xb * w)))
        if x1 <= x0 + 2:
            return False
        patch = rgb[y0:y1, x0:x1].astype(np.int16)
        lum = patch.mean(axis=2)
        sat = patch.max(axis=2) - patch.min(axis=2)
        if canvas_lum < 90.0:
            avatar_px = (sat >= 16) | (lum >= canvas_lum + 30.0)
        else:
            # Exclude AssistiveTouch grey (sat<=14, lum ~ 180..228); match colorful or crisp dark/white avatars.
            avatar_px = (sat >= 16) | (lum <= canvas_lum - 65.0) | (lum >= canvas_lum + 10.0)
        return float(avatar_px.mean()) >= 0.14

    has_left = _has_avatar(y_l0, y_l1, 0.025, 0.140)
    has_right = _has_avatar(y_r0, y_r1, 0.860, 0.975)
    if has_left and has_right:
        return "both"
    if has_left:
        return "left"
    if has_right:
        return "right"
    return "none"


def _has_continuous_white_bridge(rgb: np.ndarray, upper: Box, lower: Box, canvas_lum: float) -> bool:
    """Return True when the vertical gap between `upper` and `lower` stays inside a white/raised card."""
    h, w, _ = rgb.shape
    y0 = max(0, min(h - 1, int(round(upper.bottom * h))))
    y1 = max(0, min(h, int(round(lower.top * h))))
    if y1 <= y0:
        return True
    x_left = max(upper.left, lower.left)
    x_right = min(upper.right, lower.right)
    if x_right <= x_left:
        x_left = min(upper.left, lower.left)
        x_right = max(upper.right, lower.right)
    x0 = max(0, min(w - 1, int(round(x_left * w))))
    x1 = max(x0 + 1, min(w, int(round(x_right * w))))
    patch = rgb[y0:y1, x0:x1].astype(np.int16)
    if patch.size == 0:
        return True
    lum = patch.mean(axis=2)
    sat = patch.max(axis=2) - patch.min(axis=2)
    canvas_rows = ((np.abs(lum - canvas_lum) <= 5.0) & (sat <= 12)).mean(axis=1)
    return bool(float(canvas_rows.max()) < 0.55)


def classify_bubble_role(
    bubble_left: float,
    bubble_right: float,
    *,
    is_green: bool,
    is_white_card: bool,
    avatar_side: str = "none",
) -> MessageRole:
    """Classify a detected bubble strictly by surface type + container geometry + avatar-side cue.

    - Green bubbles (`is_green=True`): `SELF` iff hugging the right bubble gutter
      (`0.80 <= right <= 0.935`, `left >= 0.13`); otherwise `UNKNOWN` (drops incoming green
      posters/images on the left so they never trigger false human takeover).
    - White/raised bubbles (`is_white_card=True`, not green):
      * Right-only white cards (`left >= 0.22`, `0.80 <= right <= 0.935`) -> `SELF` (prevents
        self-reply loop on own shared link/mini-program cards).
      * Left-only narrow/medium white bubbles (`0.10 <= left <= 0.215`, `right < 0.80`) -> `INCOMING`.
      * Full-width white bubbles/cards touching BOTH left (`0.10 <= left <= 0.215`) and right
        (`0.80 <= right <= 0.850`) gutters disambiguate via `avatar_side`:
        - `"right"` -> `SELF` (protects wide SELF white/quote cards from misclassifying as INCOMING)
        - `"left"` -> `INCOMING` (real multi-line full-width incoming WeChat bubble)
        - `"both"` -> `UNKNOWN`
        - `"none"` (e.g. avatarless synthetic frame) -> `INCOMING` iff strictly hugging the
          incoming left margin (`left <= 0.185` and `right <= 0.840`), else `UNKNOWN`.
    - Centered, full-screen, or avatar-column boxes return `MessageRole.UNKNOWN`.
    """
    if is_green:
        if _SELF_MIN_RIGHT_EDGE <= bubble_right <= _SELF_MAX_RIGHT_EDGE and bubble_left >= _SELF_MIN_LEFT_EDGE:
            return MessageRole.SELF
        return MessageRole.UNKNOWN

    if not is_white_card:
        return MessageRole.UNKNOWN

    # Right-aligned white/mini-program card (clearly detached from left incoming gutter)
    if _SELF_MIN_RIGHT_EDGE <= bubble_right <= _SELF_MAX_RIGHT_EDGE and bubble_left >= _SELF_WHITE_CARD_MIN_LEFT:
        return MessageRole.SELF

    # Left-hugging white bubble (narrow/medium or full-width multi-line)
    if _INCOMING_MIN_LEFT_EDGE <= bubble_left <= _INCOMING_MAX_LEFT_EDGE and bubble_right <= _INCOMING_MAX_RIGHT_EDGE:
        if bubble_right <= _INCOMING_NARROW_MAX_RIGHT_EDGE:
            return MessageRole.INCOMING
        # Full-width white bubble/card spanning into 0.80..0.850: disambiguate via avatar column.
        if avatar_side == "right":
            return MessageRole.SELF
        if avatar_side == "left":
            return MessageRole.INCOMING
        if avatar_side == "both":
            return MessageRole.UNKNOWN
        if bubble_left <= _INCOMING_FULL_WIDTH_MAX_LEFT and bubble_right <= _INCOMING_FULL_WIDTH_MAX_RIGHT:
            return MessageRole.INCOMING
        return MessageRole.UNKNOWN

    return MessageRole.UNKNOWN


class ChatMessageExtractor:
    """Extracts structured group chat messages from screenshots + OCR detections."""

    def extract(
        self,
        image_bytes: bytes,
        ocr_boxes: list[OcrBox],
        *,
        group_title: str,
        self_aliases: tuple[str, ...] = (),
        bottom_limit: float = 0.875,
        captured_at: float | None = None,
    ) -> list[ExtractedMessage]:
        ts = time.time() if captured_at is None else float(captured_at)
        norm_group = canonicalize_chat_title(group_title)
        rgb = _decode_rgb(image_bytes)
        canvas_lum = _estimate_canvas_luminance(rgb, bottom_limit) if rgb is not None else 237.0

        self_items: list[OcrBox] = []
        self_is_green: dict[int, bool] = {}
        incoming_items: list[OcrBox] = []
        canvas_labels: list[OcrBox] = []
        clipped_regions: list[Box] = []

        for item in ocr_boxes:
            text = normalize_text(item.text)
            if not text:
                continue
            cy = item.box.center[1]
            if cy <= _VIEWPORT_TOP or cy >= bottom_limit:
                continue
            if item.box.bottom >= bottom_limit - 0.008:
                continue
            if _is_top_clipped_bubble(rgb, item.box, canvas_lum):
                clipped_regions.append(item.box)
                continue

            if rgb is not None:
                green_r, white_r = _bubble_ratios(rgb, item.box, canvas_lum)
                is_green = green_r >= 0.15
                is_white = white_r >= 0.30 and _has_white_above(rgb, item.box, canvas_lum)
                if is_green or is_white:
                    b_left, b_right = _measure_bubble_horizontal_span(
                        rgb,
                        item.box,
                        canvas_lum,
                        is_green=is_green,
                    )
                    avatar_side = "none"
                    if (
                        not is_green
                        and is_white
                        and _INCOMING_MIN_LEFT_EDGE <= b_left <= _INCOMING_MAX_LEFT_EDGE
                        and _INCOMING_NARROW_MAX_RIGHT_EDGE < b_right <= _INCOMING_MAX_RIGHT_EDGE
                    ):
                        b_top = _measure_bubble_top(rgb, item.box, canvas_lum, is_green=False)
                        avatar_side = _detect_avatar_side(rgb, b_top, canvas_lum, bottom_limit)
                    role = classify_bubble_role(
                        b_left,
                        b_right,
                        is_green=is_green,
                        is_white_card=is_white,
                        avatar_side=avatar_side,
                    )
                    if role == MessageRole.SELF and item.box.left <= 0.84:
                        self_items.append(item)
                        self_is_green[id(item)] = is_green
                    elif role == MessageRole.INCOMING:
                        if not _is_white_sticker_caption(rgb, item.box, canvas_lum, bottom_limit):
                            incoming_items.append(item)
                    # Ambiguous/UNKNOWN bubble or card is intentionally dropped (never leaks into canvas_labels).
                    continue
                if not _SYSTEM_MSG_RE.match(text):
                    canvas_labels.append(item)
            else:
                if _SYSTEM_MSG_RE.match(text):
                    continue
                if (
                    _SELF_MIN_RIGHT_EDGE <= item.box.right <= _SELF_MAX_RIGHT_EDGE
                    and item.box.left >= _SELF_MIN_LEFT_EDGE
                ):
                    self_items.append(item)
                    self_is_green[id(item)] = True
                elif (
                    _INCOMING_MIN_LEFT_EDGE <= item.box.left <= _INCOMING_MAX_LEFT_EDGE
                    and item.box.right <= _INCOMING_MAX_RIGHT_EDGE
                ):
                    incoming_items.append(item)

        def _overlaps_clipped_group(box: Box) -> bool:
            for cbox in clipped_regions:
                if 0 <= (box.top - cbox.bottom) <= 0.032 and abs(box.left - cbox.left) <= 0.08:
                    return True
            return False

        self_items = [b for b in self_items if not _overlaps_clipped_group(b.box)]
        incoming_items = [b for b in incoming_items if not _overlaps_clipped_group(b.box)]

        messages: list[ExtractedMessage] = []
        for role, items in ((MessageRole.SELF, self_items), (MessageRole.INCOMING, incoming_items)):
            if not items:
                continue
            items.sort(key=lambda b: (round(b.box.top, 3), round(b.box.left, 3)))
            groups: list[list[OcrBox]] = [[items[0]]]
            for item in items[1:]:
                prev = groups[-1][-1]
                vgap = item.box.top - prev.box.bottom
                if role == MessageRole.SELF:
                    aligned = abs(item.box.left - prev.box.left) <= 0.09 or abs(item.box.right - prev.box.right) <= 0.12
                    same_group = -0.005 <= vgap <= 0.032 and aligned
                    if same_group and rgb is not None:
                        prev_green = self_is_green.get(id(prev), True)
                        curr_green = self_is_green.get(id(item), True)
                        if prev_green != curr_green:
                            same_group = False
                        elif prev_green:
                            same_group = has_continuous_green_bridge(rgb, prev.box, item.box)
                        else:
                            same_group = _has_continuous_white_bridge(rgb, prev.box, item.box, canvas_lum)
                else:
                    aligned = abs(item.box.left - prev.box.left) <= 0.09
                    same_group = -0.005 <= vgap <= 0.032 and aligned
                if same_group:
                    groups[-1].append(item)
                else:
                    groups.append([item])

            for grp in groups:
                combined_box = Box.clamped(
                    min(b.box.left for b in grp),
                    min(b.box.top for b in grp),
                    max(b.box.right for b in grp),
                    max(b.box.bottom for b in grp),
                )
                joined = "\n".join(b.text.strip() for b in grp if b.text.strip())
                sender = ""
                if role == MessageRole.INCOMING:
                    sender = _find_sender_label(combined_box, canvas_labels)
                mentioned = role == MessageRole.INCOMING and mentions_any_alias(joined, self_aliases)
                messages.append(
                    ExtractedMessage(
                        group_title=norm_group,
                        role=role,
                        text=joined,
                        sender=sender,
                        box=combined_box,
                        mentioned_self=mentioned,
                        fingerprint="",
                        captured_at=ts,
                    )
                )

        messages.sort(key=lambda m: (round(m.box.top if m.box else 0.0, 4), round(m.box.left if m.box else 0.0, 4)))

        occ_counts: Counter[tuple[str, str, str, str]] = Counter()
        finalized: list[ExtractedMessage] = []
        for msg in messages:
            key = (norm_group, msg.role.value, normalize_text(msg.sender), _canonical_for_hash(msg.text))
            occ_counts[key] += 1
            fp = compute_message_fingerprint(
                norm_group,
                msg.role,
                msg.sender,
                msg.text,
                occurrence=occ_counts[key],
            )
            finalized.append(
                ExtractedMessage(
                    group_title=msg.group_title,
                    role=msg.role,
                    text=msg.text,
                    sender=msg.sender,
                    box=msg.box,
                    mentioned_self=msg.mentioned_self,
                    fingerprint=fp,
                    captured_at=msg.captured_at,
                    occurrence=occ_counts[key],
                )
            )
        return finalized


def _find_sender_label(bubble_box: Box, canvas_labels: list[OcrBox]) -> str:
    """Attach a grey-canvas sender nickname sitting directly above an incoming white bubble."""
    best: OcrBox | None = None
    best_gap = 1.0
    for lbl in canvas_labels:
        gap = bubble_box.top - lbl.box.bottom
        if not (0.002 <= gap <= 0.044):
            continue
        if abs(lbl.box.left - bubble_box.left) > 0.08:
            continue
        if lbl.box.left > 0.45:
            continue
        if gap < best_gap:
            best_gap = gap
            best = lbl
    return normalize_text(best.text) if best is not None else ""
