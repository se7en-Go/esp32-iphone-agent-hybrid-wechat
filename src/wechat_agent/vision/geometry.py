"""Normalized UI geometry helpers and iOS safety bounds."""

from __future__ import annotations

import re
import unicodedata

from wechat_agent.models import Box

# iOS Home Indicator adsorption zone near the bottom edge (y > 0.95, x in ~0.32..0.68).
HOME_INDICATOR_Y = 0.95
HOME_INDICATOR_MAX_SAFE_Y = 0.948
HOME_INDICATOR_X_RANGE = (0.32, 0.68)
_MIDDLE_DOT_RE = re.compile(r"[•·・‧∙●◦]")
_PUNCT_SPACE_RE = re.compile(r"\s*([·()])\s*")


def normalize_text(text: str) -> str:
    """Normalize NFKC full/half-width characters and collapse outer whitespace."""
    return unicodedata.normalize("NFKC", text or "").strip()


def canonicalize_chat_title(title: str) -> str:
    """Canonicalize a WeChat chat/group title without stripping group-owned `(N)` suffixes.

    - Converts fullwidth characters (such as `（198）`) via NFKC -> `(198)`
    - Unifies middle-dot punctuation variants (`•`, `·`, `・`, `‧`, `∙`, `●`, `◦`) -> `·`
    - Strips cosmetic whitespace around `·`, `(`, `)` and collapses repeated inner spaces
    """
    norm = normalize_text(title)
    if not norm:
        return ""
    dotted = _MIDDLE_DOT_RE.sub("·", norm)
    compact = _PUNCT_SPACE_RE.sub(r"\1", dotted)
    return re.sub(r"\s+", " ", compact).strip()


def is_in_home_indicator_zone(x: float, y: float) -> bool:
    """Return True if point is in or dangerously close to the bottom iOS Home Indicator zone."""
    if y > HOME_INDICATOR_Y and HOME_INDICATOR_X_RANGE[0] <= x <= HOME_INDICATOR_X_RANGE[1]:
        return True
    return y > 0.965


def clamp_avoiding_home_indicator(box: Box) -> Box:
    """Clamp bottom of a box to <= 0.948 when it overlaps the bottom Home Indicator x-band."""
    if box.right >= HOME_INDICATOR_X_RANGE[0] and box.left <= HOME_INDICATOR_X_RANGE[1]:
        safe_bottom = min(box.bottom, HOME_INDICATOR_MAX_SAFE_Y)
        safe_top = min(box.top, safe_bottom - 0.01)
        return Box.clamped(box.left, safe_top, box.right, safe_bottom)
    return box


def left_biased_input_aim(input_box: Box, *, avoid_home_indicator: bool = True) -> tuple[float, float]:
    """Aim at the left-biased text-start area inside a WeChat input box."""
    safe_box = clamp_avoiding_home_indicator(input_box) if avoid_home_indicator else input_box
    aim_x = safe_box.left + min(0.24 * safe_box.width, max(0.06, 0.16 * safe_box.width))
    aim_x = max(safe_box.left + 0.02, min(safe_box.right - 0.02, aim_x))
    if avoid_home_indicator:
        aim_y = min(safe_box.center[1], HOME_INDICATOR_MAX_SAFE_Y - 0.006)
        aim_y = max(safe_box.top + 0.004, aim_y)
        if is_in_home_indicator_zone(aim_x, aim_y):
            aim_y = min(aim_y, HOME_INDICATOR_MAX_SAFE_Y - 0.01)
    else:
        aim_y = min(safe_box.center[1], min(0.992, safe_box.bottom - 0.004))
        aim_y = max(safe_box.top + 0.004, aim_y)
    return (aim_x, aim_y)


def derive_home_icon_body_box(label_box: Box, *, image_aspect: float = 0.463) -> Box:
    """Derive the iOS app icon body box strictly ABOVE the app title label.

    Clicking the label text center fails to open an iOS Home Screen app; we must click
    the square app icon body situated directly above the label.
    """
    center_x = label_box.center[0]
    # iOS app icon is ~0.125-0.145 screen-width wide on 4-column grids; scale height by aspect ratio.
    half_w = max(0.055, min(0.075, max(label_box.width * 0.85, 0.062)))
    icon_height = (2.0 * half_w) * image_aspect
    icon_height = max(0.048, min(0.075, icon_height))
    gap = 0.008
    bottom = max(0.03, label_box.top - gap)
    top = max(0.02, bottom - icon_height)
    left = max(0.02, center_x - half_w)
    right = min(0.98, center_x + half_w)
    return Box.clamped(left, top, right, bottom)
