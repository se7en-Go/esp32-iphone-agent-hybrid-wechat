"""Dynamic WeChat iOS screen recognition, UI geometry, blue Send button, and bubble analysis."""

from __future__ import annotations

import io
import re
from collections import Counter, deque
from dataclasses import dataclass
from enum import Enum

import numpy as np
from PIL import Image

from wechat_agent.models import Box, OcrBox

from .geometry import (
    HOME_INDICATOR_MAX_SAFE_Y,
    canonicalize_chat_title,
    clamp_avoiding_home_indicator,
    derive_home_icon_body_box,
    normalize_text,
)

SEND_LABELS = ("发送", "發送", "Send")
SEND_SEARCH_REGION = Box(0.65, 0.50, 1.0, 0.96)
SPACE_LABELS = ("空格", "space", "Space")
WECHAT_TAB_LABELS = {"微信", "通讯录", "发现", "我"}
KEYBOARD_CUE_TOKENS = {"发送", "發送", "Send", "空格", "space", "Space", "换行", "return", "拼音", "选词"}


class ScreenKind(str, Enum):
    TARGET_CHAT = "target_chat"
    WECHAT_CHAT_LIST = "wechat_chat_list"
    HOME_SCREEN = "home_screen"
    SPOTLIGHT_RESULTS = "spotlight_results"
    OTHER_CHAT = "other_chat"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class FocusedInputLayout:
    input_box: Box
    caret_box: Box | None
    keyboard_top: float


def _decode_rgb(image_bytes: bytes) -> np.ndarray | None:
    if not image_bytes:
        return None
    with Image.open(io.BytesIO(image_bytes)) as img:
        return np.asarray(img.convert("RGB"))


def _image_aspect(image_bytes: bytes, default: float = 0.463) -> float:
    if not image_bytes:
        return default
    with Image.open(io.BytesIO(image_bytes)) as img:
        w, h = img.size
        if w > 0 and h > 0:
            return w / h
    return default


def _norm_box_to_pixels(box: Box, width: int, height: int) -> tuple[int, int, int, int]:
    x0 = max(0, min(width - 1, int(round(box.left * width))))
    y0 = max(0, min(height - 1, int(round(box.top * height))))
    x1 = max(x0 + 1, min(width, int(round(box.right * width))))
    y1 = max(y0 + 1, min(height, int(round(box.bottom * height))))
    return x0, y0, x1, y1


def _has_bottom_wechat_tabs(ocr_boxes: list[OcrBox]) -> bool:
    bottom_tokens = {
        normalize_text(b.text)
        for b in ocr_boxes
        if b.box.top >= 0.86 and normalize_text(b.text) in WECHAT_TAB_LABELS
    }
    return len(bottom_tokens) >= 2


_STRICT_GROUP_COUNT_RE = re.compile(r"^\(([1-9]\d{0,3})\)$")
_MAX_GROUP_MEMBER_COUNT = 2000


def _header_title_candidates(ocr_boxes: list[OcrBox]) -> list[OcrBox]:
    """Return non-empty OCR boxes inside the WeChat iOS top navigation title slot (excluding status bar)."""
    return [
        b
        for b in ocr_boxes
        if 0.045 <= b.box.top
        and b.box.bottom <= 0.14
        and 0.14 <= b.box.center[0] <= 0.86
        and normalize_text(b.text)
    ]


def extract_header_title(ocr_boxes: list[OcrBox]) -> str | None:
    """Return the normalized primary top navigation header title if present."""
    candidates = _header_title_candidates(ocr_boxes)
    if not candidates:
        return None
    candidates.sort(key=lambda b: abs(b.box.center[0] - 0.5))
    title = normalize_text(candidates[0].text)
    return title or None


def has_exact_chat_title(
    ocr_boxes: list[OcrBox],
    target_title: str,
    *,
    allow_group_count: bool = False,
) -> bool:
    """Return True iff the unique primary top-bar header title strictly matches `target_title`.

    When `allow_group_count=True`, accepts ONLY a single trailing standard member count `(N)`
    with `1 <= N <= 2000` and no leading zeros or extra characters. Rejects ambiguous screens
    with multiple conflicting title boxes or stray non-header matches (e.g. status-bar / banner).
    """
    target_canon = canonicalize_chat_title(target_title)
    if not target_canon:
        return False

    candidates = _header_title_candidates(ocr_boxes)
    if len(candidates) != 1:
        return False

    item_canon = canonicalize_chat_title(candidates[0].text)
    if item_canon == target_canon:
        return True
    if allow_group_count and item_canon.startswith(target_canon):
        suffix = item_canon[len(target_canon) :]
        m = _STRICT_GROUP_COUNT_RE.fullmatch(suffix)
        if m is not None and 1 <= int(m.group(1)) <= _MAX_GROUP_MEMBER_COUNT:
            return True
    return False


_LOCK_SCREEN_PHRASES = (
    "向上轻扫以解锁",
    "向上轻扫以打开",
    "向上轻扫以开启",
    "Swipe up to unlock",
    "Swipe up to open",
    "输入密码",
    "Enter Passcode",
    "Touch ID",
    "触控 ID",
    "面容 ID",
    "Face ID",
)


def is_lock_screen(ocr_boxes: list[OcrBox], image_bytes: bytes = b"") -> bool:
    """Return True if the screen appears asleep/black or displays the iOS Lock Screen."""
    rgb = _decode_rgb(image_bytes)
    if rgb is not None and float(rgb.mean()) < 12.0:
        return True

    for item in ocr_boxes:
        norm = normalize_text(item.text)
        if any(phrase in norm for phrase in _LOCK_SCREEN_PHRASES):
            return True

    # Large centered lock-screen clock HH:MM in upper-middle band without WeChat header
    for item in ocr_boxes:
        norm = normalize_text(item.text)
        if (
            0.08 <= item.box.top <= 0.30
            and 0.25 <= item.box.center[0] <= 0.75
            and item.box.height >= 0.045
            and re.fullmatch(r"\d{1,2}:\d{2}", norm)
        ):
            return True
    return False


def detect_screen_context(
    ocr_boxes: list[OcrBox],
    target_title: str,
    *,
    allow_group_count: bool = False,
) -> ScreenKind:
    """Classify the current iOS / WeChat screen based on OCR detections."""
    if _has_bottom_wechat_tabs(ocr_boxes):
        return ScreenKind.WECHAT_CHAT_LIST

    top_tokens = [b for b in ocr_boxes if b.box.top <= 0.18]
    has_cancel = any(normalize_text(b.text) in {"取消", "Cancel"} and b.box.left >= 0.72 for b in top_tokens)
    has_wechat_query = any(normalize_text(b.text).lower() == "wechat" for b in top_tokens)
    if has_cancel and (has_wechat_query or any(normalize_text(b.text) in {"微信", "WeChat"} for b in ocr_boxes)):
        return ScreenKind.SPOTLIGHT_RESULTS

    if has_exact_chat_title(ocr_boxes, target_title, allow_group_count=allow_group_count):
        return ScreenKind.TARGET_CHAT

    # If a WeChat icon/label appears on screen without WeChat tabs or chat nav bar, assume Home Screen.
    for item in ocr_boxes:
        norm = normalize_text(item.text)
        if norm in {"微信", "WeChat"} and 0.08 <= item.box.top <= 0.88:
            return ScreenKind.HOME_SCREEN

    # Any other non-clock title in the top navigation bar without bottom tabs indicates another chat/page.
    header_texts = [
        normalize_text(b.text)
        for b in ocr_boxes
        if 0.045 <= b.box.top <= 0.13 and 0.18 <= b.box.center[0] <= 0.82
    ]
    if header_texts:
        return ScreenKind.OTHER_CHAT

    if ocr_boxes:
        return ScreenKind.HOME_SCREEN
    return ScreenKind.UNKNOWN


def find_home_wechat_icon(ocr_boxes: list[OcrBox], image_bytes: bytes = b"") -> Box | None:
    """Locate the clickable iOS Home Screen WeChat icon BODY directly above the visible '微信' label."""
    if _has_bottom_wechat_tabs(ocr_boxes):
        return None
    aspect = _image_aspect(image_bytes)
    candidates = [
        b
        for b in ocr_boxes
        if normalize_text(b.text) in {"微信", "WeChat"} and 0.10 <= b.box.top <= 0.86
    ]
    if not candidates:
        return None
    label = candidates[0]
    return derive_home_icon_body_box(label.box, image_aspect=aspect)


def find_spotlight_wechat_target(ocr_boxes: list[OcrBox], image_bytes: bytes = b"") -> Box | None:
    """Locate WeChat in Spotlight search results (either Open/row target or icon body)."""
    aspect = _image_aspect(image_bytes)
    results = [
        b
        for b in ocr_boxes
        if normalize_text(b.text) in {"微信", "WeChat"} and 0.14 <= b.box.top <= 0.85
    ]
    if not results:
        return None
    label = results[0]

    # Check if there is an inline "打开"/"Open" button on the same row.
    row_cy = label.box.center[1]
    for item in ocr_boxes:
        if normalize_text(item.text) in {"打开", "Open"} and abs(item.box.center[1] - row_cy) <= 0.045:
            return Box.clamped(
                item.box.left - 0.02,
                item.box.top - 0.015,
                item.box.right + 0.02,
                item.box.bottom + 0.015,
            )

    # If the label sits in a horizontal list row (left ~0.16..0.55) rather than a narrow grid column,
    # check whether the icon is to its left or directly above.
    if label.box.left >= 0.16 and label.box.top <= 0.45:
        # Try icon body above when label sits under a grid icon; if too close to search bar (<0.18), click row.
        if label.box.top >= 0.22:
            return derive_home_icon_body_box(label.box, image_aspect=aspect)
    return Box.clamped(
        max(0.05, label.box.left - 0.04),
        max(0.14, label.box.top - 0.015),
        min(0.85, label.box.right + 0.18),
        min(0.85, label.box.bottom + 0.015),
    )


def find_chat_row_box(ocr_boxes: list[OcrBox], target_title: str) -> Box | None:
    """Locate a clickable row box for `target_title`, aiming at right-side blank row whitespace.

    Avoids both the black title characters on the left and the far-right timestamp text so the grey
    AssistiveTouch pointer dot stays clearly detectable during `point` and `observe --for-click`.
    """
    target_canon = canonicalize_chat_title(target_title)
    if not target_canon:
        return None
    for item in ocr_boxes:
        if item.box.top < 0.12 or item.box.bottom > 0.88:
            continue
        if canonicalize_chat_title(item.text) == target_canon:
            cy = item.box.center[1]
            right_items = [
                other
                for other in ocr_boxes
                if other is not item
                and other.box.left >= item.box.right
                and abs(other.box.center[1] - cy) <= 0.035
            ]
            right_bound = min([0.84] + [other.box.left - 0.02 for other in right_items])
            blank_left = item.box.right + 0.03
            top = max(0.12, item.box.top - 0.010)
            bottom = min(0.88, item.box.bottom + 0.010)
            if right_bound - blank_left >= 0.08:
                return Box.clamped(blank_left, top, right_bound, bottom)
            # Fallback if an unusually long title crowds the timestamp area.
            fallback_right = max(0.28, right_bound)
            fallback_left = max(0.16, fallback_right - 0.12)
            return Box.clamped(fallback_left, top, fallback_right, bottom)
    return None


def _extract_input_pill_box(rgb: np.ndarray, scan_box: Box) -> Box | None:
    """Detect the full white/pill input field inside `scan_box` without truncating at interior icons.

    Bridges small interior dark features (left green caret, right voice/mic icon inside the white
    pill) while rejecting outer toolbar icons (voice/emoji) whose column coverage stays low.
    """
    h, w, _ = rgb.shape
    x0, y0, x1, y1 = _norm_box_to_pixels(scan_box, w, h)
    patch = rgb[y0:y1, x0:x1].astype(np.int16)
    if patch.size == 0:
        return None
    lum = patch.mean(axis=2)
    bg_lum = float(np.percentile(lum, 25))
    white_thresh = max(250, int(round(bg_lum + 4)))
    mask = (patch[:, :, 0] >= white_thresh) & (patch[:, :, 1] >= white_thresh) & (patch[:, :, 2] >= white_thresh)

    if int(mask.sum()) < max(20, int(0.08 * patch.shape[0] * patch.shape[1])) and bg_lum < 80:
        mask = lum >= bg_lum + 8

    row_counts = mask.sum(axis=1)
    active_rows = np.where(row_counts >= max(10, int(0.28 * patch.shape[1])))[0]
    if len(active_rows) == 0:
        return None

    active_set = set(active_rows.tolist())
    end_r = int(active_rows[-1])
    start_r = end_r
    while start_r > 0 and ((start_r - 1) in active_set or (start_r - 2) in active_set):
        start_r -= 1

    sub = mask[start_r : end_r + 1, :]
    col_frac = sub.mean(axis=0)
    col_on = col_frac >= 0.40

    runs: list[list[int]] = []
    run_start: int | None = None
    for idx, on in enumerate(col_on):
        if on and run_start is None:
            run_start = idx
        elif not on and run_start is not None:
            runs.append([run_start, idx])
            run_start = None
    if run_start is not None:
        runs.append([run_start, len(col_on)])
    if not runs:
        return None

    # Bridge interior caret / right-side microphone icon gaps ONLY when pill border rows remain
    # white across the gap (col_frac >= 0.12). A true toolbar gutter drops near 0 and never merges.
    max_gap = max(2, int(round(0.055 * w)))
    merged: list[list[int]] = []
    for run in runs:
        if (
            merged
            and (run[0] - merged[-1][1]) <= max_gap
            and float(col_frac[merged[-1][1] : run[0]].min()) >= 0.12
        ):
            merged[-1][1] = run[1]
        else:
            merged.append(run)

    best = max(merged, key=lambda r: r[1] - r[0])
    left = (x0 + best[0]) / w
    right = (x0 + best[1]) / w
    top = (y0 + start_r) / h
    bottom = (y0 + end_r + 1) / h
    if (right - left) >= 0.25 and 0.02 <= (bottom - top) <= 0.26:
        return Box.clamped(left, top, right, bottom)
    return None


def locate_collapsed_input_box(image_bytes: bytes, ocr_boxes: list[OcrBox] | None = None) -> Box:
    """Locate the collapsed bottom input field pill, strictly clamped above y > 0.95 Home Indicator."""
    rgb = _decode_rgb(image_bytes)
    if rgb is not None:
        pill = _extract_input_pill_box(rgb, Box(0.10, 0.855, 0.82, HOME_INDICATOR_MAX_SAFE_Y))
        if pill is not None:
            return clamp_avoiding_home_indicator(pill)

    # Safe fallback box avoiding y > 0.95 system indicator bar.
    return clamp_avoiding_home_indicator(Box(0.16, 0.885, 0.76, 0.942))


def _has_open_keyboard(ocr_boxes: list[OcrBox]) -> tuple[bool, float]:
    kb_boxes = [b for b in ocr_boxes if b.box.top >= 0.54]
    if not kb_boxes:
        return False, 0.0
    cues = [b for b in kb_boxes if normalize_text(b.text) in KEYBOARD_CUE_TOKENS]
    if cues:
        upper_keys = [b for b in kb_boxes if 0.56 <= b.box.top <= 0.74]
        min_top = min(b.box.top for b in upper_keys) if upper_keys else 0.58
        return True, max(0.50, min(0.72, min_top - 0.015))
    short_keys = [b for b in kb_boxes if len(normalize_text(b.text)) <= 3 and b.box.top >= 0.58]
    if len(short_keys) >= 6:
        min_top = min(b.box.top for b in short_keys)
        return True, max(0.50, min(0.72, min_top - 0.015))
    return False, 0.0


def _find_green_caret_box(rgb: np.ndarray, input_box: Box) -> Box | None:
    h, w, _ = rgb.shape
    x0, y0, x1, y1 = _norm_box_to_pixels(input_box, w, h)
    patch = rgb[y0:y1, x0:x1].astype(np.int16)
    green_mask = (patch[:, :, 1] - patch[:, :, 0] >= 55) & (patch[:, :, 1] >= 120)
    if green_mask.sum() < 2:
        return None
    ys, xs = np.where(green_mask)
    left = max(input_box.left, (x0 + int(xs.min())) / w - 0.01)
    right = min(input_box.right, (x0 + int(xs.max()) + 1) / w + 0.015)
    top = max(input_box.top, (y0 + int(ys.min())) / h - 0.004)
    bottom = min(input_box.bottom, (y0 + int(ys.max()) + 1) / h + 0.004)
    return Box.clamped(left, top, max(left + 0.02, right), max(top + 0.015, bottom))


def detect_keyboard_and_focused_input(
    image_bytes: bytes,
    ocr_boxes: list[OcrBox],
) -> FocusedInputLayout | None:
    """Detect whether the native keyboard is open and locate the focused input field & caret box."""
    kb_open, kb_top_est = _has_open_keyboard(ocr_boxes)
    if not kb_open:
        return None

    rgb = _decode_rgb(image_bytes)
    search_top = 0.34
    search_bottom = min(0.72, max(0.60, kb_top_est + 0.04))
    input_box: Box | None = None
    caret_box: Box | None = None

    if rgb is not None:
        input_box = _extract_input_pill_box(rgb, Box(0.10, search_top, 0.84, search_bottom))

    if input_box is None:
        top = max(0.38, kb_top_est - 0.068)
        bottom = max(top + 0.04, kb_top_est - 0.008)
        input_box = Box.clamped(0.13, top, 0.78, bottom)

    if rgb is not None:
        caret_box = _find_green_caret_box(rgb, input_box)
    if caret_box is None:
        caret_box = Box.clamped(
            input_box.left + 0.004,
            input_box.top + 0.004,
            min(input_box.right, input_box.left + 0.045),
            input_box.bottom - 0.004,
        )

    return FocusedInputLayout(input_box=input_box, caret_box=caret_box, keyboard_top=max(kb_top_est, input_box.bottom + 0.008))


def detect_existing_draft(ocr_boxes: list[OcrBox], input_box: Box) -> str | None:
    """Return existing non-empty text inside `input_box` (preserving user drafts), or None if empty."""
    expanded = Box.clamped(
        input_box.left - 0.01,
        max(0.18, input_box.top - 0.01),
        input_box.right + 0.01,
        input_box.bottom + 0.01,
    )
    found: list[OcrBox] = []
    for item in ocr_boxes:
        cx, cy = item.box.center
        if expanded.contains(cx, cy):
            norm = normalize_text(item.text)
            if norm and norm not in {"按住说话", "按住 说话"}:
                found.append(item)
    if not found:
        return None
    found.sort(key=lambda b: (round(b.box.top, 3), round(b.box.left, 3)))
    return "\n".join(item.text.strip() for item in found if item.text.strip())


def _canonical_text(text: str) -> str:
    return re.sub(r"\s+", "", normalize_text(text))


def _group_visual_rows(items: list[OcrBox]) -> list[list[OcrBox]]:
    """Group OCR boxes inside an input region into visual rows ordered top-to-bottom, left-to-right."""
    if not items:
        return []
    ordered = sorted(items, key=lambda b: (round(b.box.center[1], 4), round(b.box.left, 4)))
    rows: list[list[OcrBox]] = []
    for item in ordered:
        cy = item.box.center[1]
        if rows:
            prev_row = rows[-1]
            row_cy = sum(b.box.center[1] for b in prev_row) / len(prev_row)
            row_h = max(b.box.height for b in prev_row)
            if abs(cy - row_cy) <= max(0.010, row_h * 0.55):
                prev_row.append(item)
                continue
        rows.append([item])
    for row in rows:
        row.sort(key=lambda b: round(b.box.left, 4))
    return rows


def _matches_logical_lines(exp_lines: list[str], obs_rows: list[str]) -> bool:
    """Require each logical newline-separated line to begin on a fresh visual row (allowing soft-wrap)."""
    if len(obs_rows) < len(exp_lines):
        return False
    idx = 0
    for exp_line in exp_lines:
        if idx >= len(obs_rows):
            return False
        acc = ""
        while idx < len(obs_rows) and len(acc) < len(exp_line):
            acc += obs_rows[idx]
            idx += 1
        if acc != exp_line:
            return False
    return idx == len(obs_rows)


def verify_pasted_text(
    ocr_boxes: list[OcrBox],
    input_box: Box,
    expected_text: str,
    *,
    allow_upward_growth: bool = False,
) -> tuple[bool, str]:
    """Verify pasted text inside `input_box` matches `expected_text` once, preserving hard newlines."""
    # When callers already pass the freshly detected post-paste white input box, keep a tight
    # margin (±0.015) so adjacent incoming chat bubbles immediately above never leak in.
    # Only widen upward when falling back to a pre-paste single-line input box.
    top_margin = 0.14 if allow_upward_growth else 0.015
    paste_region = Box.clamped(
        input_box.left - 0.015,
        max(0.18, input_box.top - top_margin),
        input_box.right + 0.015,
        min(0.72, input_box.bottom + 0.015),
    )
    items = [
        item
        for item in ocr_boxes
        if paste_region.contains(*item.box.center)
        and normalize_text(item.text) not in KEYBOARD_CUE_TOKENS
    ]
    visual_rows = _group_visual_rows(items)
    row_texts = [" ".join(b.text.strip() for b in row if b.text.strip()) for row in visual_rows]
    observed_raw = "\n".join(rt for rt in row_texts if rt)

    canon_obs = _canonical_text(observed_raw)
    canon_exp = _canonical_text(expected_text)
    if not canon_obs or not canon_exp:
        return False, observed_raw

    # Reject duplicated paste (e.g., pasted twice).
    if len(canon_exp) >= 2 and canon_obs.count(canon_exp) > 1:
        return False, observed_raw
    if len(canon_obs) >= int(1.85 * len(canon_exp)) and len(canon_exp) >= 2:
        return False, observed_raw

    if canon_obs != canon_exp:
        return False, observed_raw

    exp_lines = [_canonical_text(line) for line in expected_text.splitlines() if _canonical_text(line)]
    obs_rows = [_canonical_text(rt) for rt in row_texts if _canonical_text(rt)]
    if len(exp_lines) > 1 and not _matches_logical_lines(exp_lines, obs_rows):
        return False, observed_raw

    return True, observed_raw


def _find_space_bar_right_edge(ocr_boxes: list[OcrBox]) -> float:
    """Locate the right edge of the space bar OCR text if visible in the bottom keyboard row."""
    candidates = [
        b
        for b in ocr_boxes
        if b.box.top >= 0.72 and b.box.right <= 0.78 and normalize_text(b.text) in SPACE_LABELS
    ]
    if not candidates:
        return 0.0
    return max(b.box.right for b in candidates)


def _find_connected_blue_button(rgb: np.ndarray, label_box: Box) -> Box | None:
    """Measure the contiguous iOS blue Send button around `label_box` via bounded BFS."""
    h, w, _ = rgb.shape
    search_box = Box.clamped(
        max(0.60, label_box.left - 0.16),
        max(0.52, label_box.top - 0.09),
        min(1.0, label_box.right + 0.16),
        min(0.97, label_box.bottom + 0.09),
    )
    x0, y0, x1, y1 = _norm_box_to_pixels(search_box, w, h)
    patch = rgb[y0:y1, x0:x1].astype(np.int16)
    # Match iOS blue Send button (#007AFF light, #0A84FF dark) while ignoring white label text inside.
    blue_mask = (
        (patch[:, :, 2] >= 135)
        & ((patch[:, :, 2] - patch[:, :, 0]) >= 35)
        & (patch[:, :, 2] >= patch[:, :, 1] - 15)
    )
    ph, pw = blue_mask.shape
    if int(blue_mask.sum()) < 25:
        return None

    # Seed from pixels inside/near the label box (jumping off white label strokes onto surrounding blue fill).
    lx0, ly0, lx1, ly1 = _norm_box_to_pixels(label_box, w, h)
    slx0 = max(0, lx0 - x0 - 6)
    sly0 = max(0, ly0 - y0 - 6)
    slx1 = min(pw, lx1 - x0 + 6)
    sly1 = min(ph, ly1 - y0 + 6)

    visited = np.zeros((ph, pw), dtype=bool)
    queue: deque[tuple[int, int]] = deque()
    for ry in range(sly0, sly1):
        for rx in range(slx0, slx1):
            if blue_mask[ry, rx]:
                visited[ry, rx] = True
                queue.append((ry, rx))

    if not queue:
        return None

    min_rx, max_rx = pw, -1
    min_ry, max_ry = ph, -1
    count = 0
    while queue:
        cy, cx = queue.popleft()
        count += 1
        if cx < min_rx:
            min_rx = cx
        if cx > max_rx:
            max_rx = cx
        if cy < min_ry:
            min_ry = cy
        if cy > max_ry:
            max_ry = cy
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            ny, nx = cy + dy, cx + dx
            if 0 <= ny < ph and 0 <= nx < pw and not visited[ny, nx]:
                # Allow traversing either blue pixels or interior white label pixels inside the label box.
                in_label = slx0 <= nx < slx1 and sly0 <= ny < sly1
                if blue_mask[ny, nx] or in_label:
                    visited[ny, nx] = True
                    queue.append((ny, nx))

    if count < 25 or max_rx <= min_rx or max_ry <= min_ry:
        return None

    left = (x0 + min_rx) / w
    right = (x0 + max_rx + 1) / w
    top = (y0 + min_ry) / h
    bottom = (y0 + max_ry + 1) / h

    # Slight inner inset (~0.003) keeps pointer strictly inside rounded button corners while covering body.
    inset_x = min(0.004, (right - left) * 0.05)
    inset_y = min(0.004, (bottom - top) * 0.08)
    measured = Box.clamped(
        min(left + inset_x, label_box.left),
        min(top + inset_y, label_box.top),
        max(right - inset_x, label_box.right),
        max(bottom - inset_y, label_box.bottom),
    )
    # Sanity check: ensure we didn't accidentally capture an unrelated huge region.
    if measured.width > 0.34 or measured.height > 0.22:
        return None
    return measured


def locate_blue_send_button(
    image_bytes: bytes,
    ocr_boxes: list[OcrBox],
) -> tuple[Box, str] | None:
    """Locate the lower-right keyboard Send button and expand target_box across the full blue button body.

    Strictly avoids overlapping the left space bar key and restricts OCR matching to the lower-right
    keyboard region (`x >= 0.65, y >= 0.50`). Returns `(button_box, matched_label)` or `None`.
    """
    matches = [
        b
        for b in ocr_boxes
        if normalize_text(b.text) in SEND_LABELS
        and SEND_SEARCH_REGION.contains(*b.box.center)
    ]
    if len(matches) != 1:
        return None

    label_item = matches[0]
    matched_label = normalize_text(label_item.text)
    space_right = _find_space_bar_right_edge(ocr_boxes)
    # Hard minimum left bound keeps the button strictly right of the space bar text/keycap.
    min_safe_left = max(0.68, space_right + 0.015) if space_right > 0 else 0.68
    min_safe_left = min(min_safe_left, label_item.box.left)

    rgb = _decode_rgb(image_bytes)
    blue_box = _find_connected_blue_button(rgb, label_item.box) if rgb is not None else None

    if blue_box is not None:
        button_box = Box.clamped(
            max(min_safe_left, blue_box.left),
            blue_box.top,
            min(0.995, blue_box.right),
            min(0.955, blue_box.bottom),
        )
    else:
        # Fallback dynamic expansion around OCR label when blue pixels are absent.
        button_box = Box.clamped(
            max(min_safe_left, label_item.box.left - 0.035),
            max(0.55, label_item.box.top - 0.018),
            min(0.99, label_item.box.right + 0.035),
            min(0.95, label_item.box.bottom + 0.018),
        )
    return button_box, matched_label


def _is_green_bubble_box(rgb: np.ndarray, box: Box) -> bool:
    h, w, _ = rgb.shape
    probe = Box.clamped(
        max(0.0, box.left - 0.02),
        max(0.0, box.top - 0.01),
        min(1.0, box.right + 0.02),
        min(1.0, box.bottom + 0.01),
    )
    x0, y0, x1, y1 = _norm_box_to_pixels(probe, w, h)
    patch = rgb[y0:y1, x0:x1].astype(np.int16)
    green_mask = (patch[:, :, 1] - patch[:, :, 0] >= 12) & (patch[:, :, 1] - patch[:, :, 2] >= 12)
    return float(green_mask.mean()) >= 0.15


def has_continuous_green_bridge(rgb: np.ndarray, top_box: Box, bottom_box: Box) -> bool:
    """Verify vertical inter-line gap stays inside the same green bubble (no grey canvas gap)."""
    if bottom_box.top - top_box.bottom <= 0.003:
        return True
    h, w, _ = rgb.shape
    x_left = max(top_box.left, bottom_box.left)
    x_right = min(top_box.right, bottom_box.right)
    if x_right - x_left < 0.01:
        x_left, x_right = bottom_box.left, bottom_box.right
    strip = Box.clamped(x_left, top_box.bottom, x_right, bottom_box.top)
    x0, y0, x1, y1 = _norm_box_to_pixels(strip, w, h)
    if y1 <= y0:
        return True
    patch = rgb[y0:y1, x0:x1].astype(np.int16)
    green_mask = (patch[:, :, 1] - patch[:, :, 0] >= 12) & (patch[:, :, 1] - patch[:, :, 2] >= 12)
    return float(green_mask.mean(axis=1).min()) >= 0.20


def extract_outgoing_bubbles(
    image_bytes: bytes,
    ocr_boxes: list[OcrBox],
    chat_area: Box | None = None,
) -> list[str]:
    """Extract outgoing (right-side green bubble) message texts visible inside `chat_area`."""
    area = chat_area or Box(0.0, 0.13, 1.0, 0.87)
    rgb = _decode_rgb(image_bytes)

    candidates: list[OcrBox] = []
    for item in ocr_boxes:
        if not area.contains(*item.box.center):
            continue
        if not normalize_text(item.text):
            continue
        if rgb is not None:
            # Exclude far-left/right avatar columns while keeping short wrapped tail lines inside wide green bubbles.
            if not (0.13 <= item.box.left <= 0.83):
                continue
            if not _is_green_bubble_box(rgb, item.box):
                continue
        else:
            if item.box.right < 0.42 or item.box.center[0] < 0.50:
                continue
        candidates.append(item)

    if not candidates:
        return []

    candidates.sort(key=lambda b: (round(b.box.top, 3), round(b.box.left, 3)))
    grouped: list[list[OcrBox]] = [[candidates[0]]]
    for item in candidates[1:]:
        prev = grouped[-1][-1]
        vgap = item.box.top - prev.box.bottom
        aligned = abs(item.box.left - prev.box.left) <= 0.12 or abs(item.box.right - prev.box.right) <= 0.25
        same_bubble = -0.005 <= vgap <= 0.032 and aligned
        if same_bubble and rgb is not None:
            same_bubble = has_continuous_green_bridge(rgb, prev.box, item.box)
        if same_bubble:
            grouped[-1].append(item)
        else:
            grouped.append([item])

    return ["\n".join(part.text.strip() for part in grp if part.text.strip()) for grp in grouped]


def has_new_outgoing_bubble(
    baseline_bubbles: list[str],
    current_bubbles: list[str],
    expected_text: str,
) -> bool:
    """Confirm that a genuinely NEW outgoing bubble matching `expected_text` appeared after Send."""
    canon_expected = _canonical_text(expected_text)
    if not canon_expected or not current_bubbles:
        return False

    base_counts = Counter(_canonical_text(b) for b in baseline_bubbles)
    curr_counts = Counter(_canonical_text(b) for b in current_bubbles)

    # Check if an exact (or long-text >=85% OCR-edge-truncated) bubble increased in count over baseline.
    for canon_msg, count in curr_counts.items():
        if count > base_counts.get(canon_msg, 0):
            if canon_msg == canon_expected:
                return True
            if len(canon_expected) >= 10 and len(canon_msg) >= 10:
                ratio = min(len(canon_expected), len(canon_msg)) / max(len(canon_expected), len(canon_msg))
                if ratio >= 0.85 and (canon_expected in canon_msg or canon_msg in canon_expected):
                    return True
    return False
