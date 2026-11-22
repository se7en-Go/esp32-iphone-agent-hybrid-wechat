"""Test fakes and synthetic WeChat iOS image builders."""

from __future__ import annotations

import hashlib
import io
from collections.abc import Callable
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from wechat_agent.adapters.driver import ConflictError, DriverError
from wechat_agent.models import (
    Box,
    ClickResult,
    MeasuredPointer,
    Observation,
    OcrBox,
    PasteFlowResult,
    PasteRecoveryState,
    PointResult,
    SkillCheckResult,
    SkillConfirmResult,
    SkillPrepareResult,
)


def _encode_png(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _box_px(box: Box, w: int, h: int) -> tuple[int, int, int, int]:
    return (
        int(round(box.left * w)),
        int(round(box.top * h)),
        int(round(box.right * w)),
        int(round(box.bottom * h)),
    )


def make_home_screen_image(w: int = 480, h: int = 1040) -> bytes:
    img = Image.new("RGB", (w, h), (230, 236, 245))
    draw = ImageDraw.Draw(img)
    # Green WeChat icon square above label around (0.14..0.28, 0.23..0.30)
    draw.rounded_rectangle(_box_px(Box(0.14, 0.23, 0.28, 0.295), w, h), radius=12, fill=(7, 193, 96))
    return _encode_png(img)


def make_chat_list_image(w: int = 480, h: int = 1040) -> bytes:
    img = Image.new("RGB", (w, h), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    # Top bar + bottom tab bar
    draw.rectangle([0, 0, w, int(0.12 * h)], fill=(237, 237, 237))
    draw.rectangle([0, int(0.88 * h), w, h], fill=(247, 247, 247))
    return _encode_png(img)


def make_collapsed_chat_image(
    *,
    w: int = 480,
    h: int = 1040,
    outgoing_boxes: list[Box] | None = None,
    incoming_boxes: list[Box] | None = None,
    with_right_mic_icon: bool = False,
    with_side_toolbar_icons: bool = False,
) -> bytes:
    img = Image.new("RGB", (w, h), (237, 237, 237))
    draw = ImageDraw.Draw(img)
    # Top nav bar
    draw.rectangle([0, 0, w, int(0.12 * h)], fill=(237, 237, 237))
    # Bottom collapsed input toolbar
    draw.rectangle([0, int(0.875 * h), w, h], fill=(245, 245, 245))
    # White input pill inside safe zone above y=0.95
    draw.rounded_rectangle(_box_px(Box(0.132, 0.900, 0.764, 0.947), w, h), radius=8, fill=(255, 255, 255))
    if with_right_mic_icon:
        # Voice/microphone icon inside right end of white pill (~0.692..0.720)
        draw.rounded_rectangle(_box_px(Box(0.694, 0.908, 0.718, 0.938), w, h), radius=4, fill=(75, 75, 80))
    if with_side_toolbar_icons:
        # Outer voice icon (left) and emoji / plus icons (right) must NOT bleed into pill bounds
        draw.ellipse(_box_px(Box(0.03, 0.904, 0.10, 0.942), w, h), outline=(60, 60, 65), width=3)
        draw.ellipse(_box_px(Box(0.79, 0.904, 0.86, 0.942), w, h), outline=(60, 60, 65), width=3)
        draw.ellipse(_box_px(Box(0.89, 0.904, 0.96, 0.942), w, h), outline=(60, 60, 65), width=3)
    for ob in outgoing_boxes or []:
        draw.rounded_rectangle(_box_px(ob, w, h), radius=8, fill=(149, 236, 105))
    for ib in incoming_boxes or []:
        draw.rounded_rectangle(_box_px(ib, w, h), radius=8, fill=(255, 255, 255))
    return _encode_png(img)


def make_keyboard_chat_image(
    *,
    w: int = 480,
    h: int = 1040,
    blue_send: bool = True,
    with_caret: bool = True,
    with_right_mic_icon: bool = False,
    with_side_toolbar_icons: bool = False,
    outgoing_boxes: list[Box] | None = None,
    incoming_boxes: list[Box] | None = None,
    input_box: Box | None = None,
) -> bytes:
    img = Image.new("RGB", (w, h), (237, 237, 237))
    draw = ImageDraw.Draw(img)
    pill = input_box or Box(0.132, 0.528, 0.764, 0.576)
    toolbar_top = min(0.50, max(0.20, pill.top - 0.014))
    # Focused toolbar directly above keyboard (keyboard starts ~0.58)
    draw.rectangle([0, int(toolbar_top * h), w, int(0.58 * h)], fill=(245, 245, 245))
    # Focused white input box (matches real WeChat 0.132..0.764)
    draw.rounded_rectangle(_box_px(pill, w, h), radius=8, fill=(255, 255, 255))
    if with_caret:
        # Bright WeChat green caret bar near left side of bottom line inside input box
        caret_top = max(pill.top + 0.006, pill.bottom - 0.041)
        caret_bottom = max(caret_top + 0.015, pill.bottom - 0.008)
        draw.rectangle(
            _box_px(Box(pill.left + 0.023, caret_top, pill.left + 0.031, caret_bottom), w, h),
            fill=(7, 193, 96),
        )
    if with_right_mic_icon:
        # Interior dark grey microphone icon inside right side of white input box (~0.692..0.720)
        draw.rounded_rectangle(_box_px(Box(0.694, 0.536, 0.718, 0.568), w, h), radius=4, fill=(77, 75, 78))
    if with_side_toolbar_icons:
        # Outer voice & emoji icons outside pill bounds (with white inner ring, must NOT be merged into pill)
        draw.ellipse(_box_px(Box(0.03, 0.532, 0.10, 0.570), w, h), fill=(255, 255, 255), outline=(60, 60, 65), width=3)
        draw.ellipse(_box_px(Box(0.795, 0.532, 0.865, 0.570), w, h), fill=(255, 255, 255), outline=(60, 60, 65), width=3)
    # Keyboard background
    draw.rectangle([0, int(0.58 * h), w, h], fill=(209, 212, 217))
    # Bottom row keys: grey space bar + optional blue Send button
    draw.rounded_rectangle(_box_px(Box(0.25, 0.855, 0.73, 0.932), w, h), radius=8, fill=(252, 252, 252))
    send_fill = (0, 122, 255) if blue_send else (180, 185, 195)
    draw.rounded_rectangle(_box_px(Box(0.75, 0.855, 0.985, 0.932), w, h), radius=8, fill=send_fill)

    for ob in outgoing_boxes or []:
        draw.rounded_rectangle(_box_px(ob, w, h), radius=8, fill=(149, 236, 105))
    for ib in incoming_boxes or []:
        draw.rounded_rectangle(_box_px(ib, w, h), radius=8, fill=(255, 255, 255))
    return _encode_png(img)


class MockOcrEngine:
    """Deterministic OCR stub keyed by image byte digest."""

    def __init__(self) -> None:
        self._map: dict[str, list[OcrBox]] = {}

    def register(self, image_bytes: bytes, boxes: list[OcrBox]) -> None:
        key = hashlib.sha256(image_bytes).hexdigest()
        self._map[key] = list(boxes)

    def read_boxes(self, image_bytes: bytes) -> list[OcrBox]:
        key = hashlib.sha256(image_bytes).hexdigest()
        return list(self._map.get(key, []))


class FakePhoneDriver:
    """Scriptable fake PhoneDriver recording every call for strict offline assertion."""

    def __init__(
        self,
        *,
        status_payload: dict[str, Any] | None = None,
        capabilities_payload: dict[str, Any] | None = None,
        skills: list[str] | None = None,
    ) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self._status = status_payload or {
            "device": True,
            "screen": True,
            "calibrated": True,
            "ready": True,
            "guard": {"tripped": False, "reason": ""},
        }
        self._capabilities = capabilities_payload or {
            "features": {
                "clipboard": {
                    "configured": True,
                    "staged": True,
                    "protocol": "staged-v1",
                    "staged_available": True,
                }
            }
        }
        self._skills = skills if skills is not None else ["wechat-send", "wechat-group-reply"]

        self.observe_queue: list[Observation | Exception] = []
        self.observe_for_click_queue: list[Observation | Exception] = []
        self.point_handler: Callable[[float, float, str, Box | None, bool], PointResult | Exception] | None = None
        self.click_handler: Callable[[float, float, str, Box, str], ClickResult | Exception] | None = None
        self.search_app_handler: Callable[[str, str], Observation] | None = None
        self.scroll_handler: Callable[[float, float, int, str], Observation] | None = None
        self.paste_flow_handler: Callable[..., PasteFlowResult] | None = None
        self.skill_check_handler: Callable[[str, str], SkillCheckResult] | None = None
        self.skill_prepare_handler: Callable[[str], SkillPrepareResult] | None = None
        self.skill_confirm_handler: Callable[[bool, str, str], SkillConfirmResult] | None = None

        self._obs_counter = 0

    def status(self) -> dict[str, Any]:
        self.calls.append(("status", {}))
        return dict(self._status)

    def capabilities(self) -> dict[str, Any]:
        self.calls.append(("capabilities", {}))
        return dict(self._capabilities)

    def skill_list(self) -> list[str]:
        self.calls.append(("skill_list", {}))
        return list(self._skills)

    def skill_show(self, name: str) -> dict[str, Any]:
        self.calls.append(("skill_show", {"name": name}))
        return {"name": name}

    def skill_check(self, observed_text: str, skill: str = "") -> SkillCheckResult:
        self.calls.append(("skill_check", {"observed_text": observed_text, "skill": skill}))
        if self.skill_check_handler:
            return self.skill_check_handler(observed_text, skill)
        for bad in ("操作频繁", "风控", "验证码", "账号异常", "功能受限"):
            if bad in observed_text:
                return SkillCheckResult(stop=True, counted=False, allowed=False, reason=bad)
        return SkillCheckResult(stop=False, counted=False, allowed=False, reason="ok")

    def skill_prepare_send(self, skill: str = "") -> SkillPrepareResult:
        self.calls.append(("skill_prepare_send", {"skill": skill}))
        if self.skill_prepare_handler:
            return self.skill_prepare_handler(skill)
        return SkillPrepareResult(stop=False, counted=False, allowed=True, reason="", reservation="res-1")

    def skill_confirm_send(
        self,
        bubble_visible: bool,
        observed_text: str = "",
        skill: str = "",
    ) -> SkillConfirmResult:
        self.calls.append(
            ("skill_confirm_send", {"bubble_visible": bubble_visible, "observed_text": observed_text, "skill": skill})
        )
        if self.skill_confirm_handler:
            return self.skill_confirm_handler(bubble_visible, observed_text, skill)
        if not bubble_visible:
            return SkillConfirmResult(stop=False, counted=False, allowed=False, reason="no bubble")
        return SkillConfirmResult(stop=False, counted=True, allowed=True, reason="")

    def observe(
        self,
        *,
        for_click: bool = False,
        image_output: Path | None = None,
    ) -> Observation:
        self.calls.append(("observe", {"for_click": for_click}))
        self._obs_counter += 1
        if for_click:
            if self.observe_for_click_queue:
                nxt = self.observe_for_click_queue.pop(0)
                if isinstance(nxt, Exception):
                    raise nxt
                return nxt
            return Observation(screen_id=f"scr-fc-{self._obs_counter}", image_bytes=b"")
        if self.observe_queue:
            nxt = self.observe_queue.pop(0)
            if isinstance(nxt, Exception):
                raise nxt
            return nxt
        return Observation(screen_id=f"scr-{self._obs_counter}", image_bytes=b"")

    def point(
        self,
        x: float,
        y: float,
        *,
        screen_id: str,
        target_box: Box | None = None,
        rehome: bool = False,
        caret_scope: Box | None = None,
        image_output: Path | None = None,
    ) -> PointResult:
        self.calls.append(
            (
                "point",
                {
                    "x": x,
                    "y": y,
                    "screen_id": screen_id,
                    "target_box": target_box,
                    "rehome": rehome,
                    "caret_scope": caret_scope,
                },
            )
        )
        if self.point_handler:
            res = self.point_handler(x, y, screen_id, target_box, rehome)
            if isinstance(res, Exception):
                raise res
            return res
        return PointResult(screen_id=f"{screen_id}-pt", measured_pointer=MeasuredPointer(x, y), pointer_seen=True)

    def click_observed(
        self,
        pointer_x: float,
        pointer_y: float,
        *,
        screen_id: str,
        target_box: Box,
        target_label: str = "",
        duration: float | None = None,
        caret_scope: Box | None = None,
        image_output: Path | None = None,
    ) -> ClickResult:
        self.calls.append(
            (
                "click_observed",
                {
                    "pointer_x": pointer_x,
                    "pointer_y": pointer_y,
                    "screen_id": screen_id,
                    "target_box": target_box,
                    "target_label": target_label,
                    "caret_scope": caret_scope,
                },
            )
        )
        if self.click_handler:
            res = self.click_handler(pointer_x, pointer_y, screen_id, target_box, target_label)
            if isinstance(res, Exception):
                raise res
            return res
        return ClickResult(screen_id=f"{screen_id}-clk", clicked=True, command_acknowledged=True, effect_verified=True)

    def find_text(
        self,
        text: str,
        *,
        screen_id: str,
        nth: int = 1,
        exact: bool = False,
        region: Box | None = None,
    ) -> dict[str, Any]:
        self.calls.append(("find_text", {"text": text, "screen_id": screen_id}))
        return {"matches": [], "match": None}

    def search_app(
        self,
        text: str,
        *,
        screen_id: str,
        image_output: Path | None = None,
    ) -> Observation:
        self.calls.append(("search_app", {"text": text, "screen_id": screen_id}))
        if self.search_app_handler:
            return self.search_app_handler(text, screen_id)
        return Observation(screen_id=f"{screen_id}-spot", image_bytes=b"")

    def scroll(
        self,
        x: float,
        y: float,
        lines: int,
        *,
        screen_id: str,
        image_output: Path | None = None,
    ) -> Observation:
        self.calls.append(("scroll", {"x": x, "y": y, "lines": lines, "screen_id": screen_id}))
        if self.scroll_handler:
            return self.scroll_handler(x, y, lines, screen_id)
        return Observation(screen_id=f"{screen_id}-scr", image_bytes=b"")

    def paste_flow(
        self,
        payload_path: Path,
        *,
        target_box: Box,
        expect_title: str,
        caret_box: Box | None = None,
        evidence_dir: Path | None = None,
        image_output: Path | None = None,
    ) -> PasteFlowResult:
        self.calls.append(
            (
                "paste_flow",
                {
                    "payload_path": str(payload_path),
                    "payload_text": payload_path.read_text(encoding="utf-8"),
                    "target_box": target_box,
                    "expect_title": expect_title,
                    "caret_box": caret_box,
                },
            )
        )
        if self.paste_flow_handler:
            return self.paste_flow_handler(
                payload_path,
                target_box=target_box,
                expect_title=expect_title,
                caret_box=caret_box,
            )
        return PasteFlowResult(
            ok=True,
            cmd_v_sent=1,
            phase="completed",
            effect_verified=True,
            recovery_state=PasteRecoveryState.COMPLETED,
            next_action="Paste confirmed.",
            copy_trigger="assistive_touch_hold",
            raw={
                "cmd_v_sent": 1,
                "phase": "completed",
                "effect_verified": True,
                "recovery_state": "completed",
                "transaction_id": "tx-secret-123",
            },
        )
