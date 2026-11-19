"""Domain models and strict response parsers for WeChat automation."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


@dataclass(frozen=True)
class Box:
    """Normalized bounding box in [0, 1] coordinates (left, top, right, bottom)."""

    left: float
    top: float
    right: float
    bottom: float

    def __post_init__(self) -> None:
        values = (self.left, self.top, self.right, self.bottom)
        if not all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in values):
            raise ValueError(f"Box coordinates must be finite numbers, got {values!r}")
        if not (0.0 <= self.left < self.right <= 1.0 and 0.0 <= self.top < self.bottom <= 1.0):
            raise ValueError(
                f"Box must satisfy 0 <= left < right <= 1 and 0 <= top < bottom <= 1, got {values!r}"
            )

    @property
    def width(self) -> float:
        return self.right - self.left

    @property
    def height(self) -> float:
        return self.bottom - self.top

    @property
    def center(self) -> tuple[float, float]:
        return ((self.left + self.right) / 2.0, (self.top + self.bottom) / 2.0)

    def contains(self, x: float, y: float, margin: float = 0.0) -> bool:
        if margin < 0:
            raise ValueError("margin must be non-negative")
        if self.left + margin > self.right - margin or self.top + margin > self.bottom - margin:
            return False
        return (self.left + margin) <= x <= (self.right - margin) and (self.top + margin) <= y <= (
            self.bottom - margin
        )

    def intersects(self, other: Box) -> bool:
        return not (
            self.right <= other.left
            or other.right <= self.left
            or self.bottom <= other.top
            or other.bottom <= self.top
        )

    def clamp(
        self,
        *,
        min_left: float = 0.0,
        min_top: float = 0.0,
        max_right: float = 1.0,
        max_bottom: float = 1.0,
    ) -> Box:
        return Box.clamped(
            max(self.left, min_left),
            max(self.top, min_top),
            min(self.right, max_right),
            min(self.bottom, max_bottom),
        )

    @classmethod
    def clamped(cls, left: float, top: float, right: float, bottom: float) -> Box:
        cl = max(0.0, min(1.0, float(left)))
        ct = max(0.0, min(1.0, float(top)))
        cr = max(0.0, min(1.0, float(right)))
        cb = max(0.0, min(1.0, float(bottom)))
        if cr <= cl:
            cr = min(1.0, cl + 1e-4)
            if cr <= cl:
                cl = max(0.0, cr - 1e-4)
        if cb <= ct:
            cb = min(1.0, ct + 1e-4)
            if cb <= ct:
                ct = max(0.0, cb - 1e-4)
        return cls(cl, ct, cr, cb)

    def as_tuple(self) -> tuple[float, float, float, float]:
        return (self.left, self.top, self.right, self.bottom)

    def as_list(self) -> list[float]:
        return [self.left, self.top, self.right, self.bottom]


@dataclass(frozen=True)
class OcrBox:
    """Single OCR detection in normalized screen coordinates."""

    text: str
    box: Box
    confidence: float = 1.0


@dataclass(frozen=True)
class MeasuredPointer:
    """Pointer position reported by the base server as a JSON object {"x": ..., "y": ...}."""

    x: float
    y: float

    @classmethod
    def from_point_response(cls, payload: dict[str, Any]) -> MeasuredPointer | None:
        if not isinstance(payload, dict):
            raise ValueError(f"Expected dict point response, got {type(payload).__name__}")
        raw_mp = payload.get("measured_pointer")
        if raw_mp is None:
            return None
        if not isinstance(raw_mp, dict):
            raise ValueError(
                f"measured_pointer must be a JSON object or null, got {type(raw_mp).__name__}: {raw_mp!r}"
            )
        if "x" not in raw_mp or "y" not in raw_mp:
            raise ValueError(f"measured_pointer object missing 'x' or 'y': {raw_mp!r}")
        raw_x, raw_y = raw_mp["x"], raw_mp["y"]
        if (
            isinstance(raw_x, bool)
            or isinstance(raw_y, bool)
            or not isinstance(raw_x, (int, float))
            or not isinstance(raw_y, (int, float))
        ):
            raise ValueError(f"measured_pointer coordinates must be numbers: {raw_mp!r}")
        x, y = float(raw_x), float(raw_y)
        if not (math.isfinite(x) and math.isfinite(y)):
            raise ValueError(f"measured_pointer coordinates must be finite: {raw_mp!r}")
        return cls(x=x, y=y)


@dataclass
class Observation:
    screen_id: str
    image_bytes: bytes = b""
    image_path: str | None = None
    width: int = 0
    height: int = 0
    fresh: bool = True
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class PointResult:
    screen_id: str | None
    measured_pointer: MeasuredPointer | None
    pointer_seen: bool = False
    image_bytes: bytes = b""
    image_path: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class ClickResult:
    screen_id: str | None
    clicked: bool = False
    command_acknowledged: bool = False
    effect_verified: bool = False
    changed: bool | None = None
    image_bytes: bytes = b""
    image_path: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)


class PasteRecoveryState(str, Enum):
    COMPLETED = "completed"
    NOT_COPIED = "not_copied"
    CLAIMED_NOT_CONFIRMED = "claimed_not_confirmed"
    COPIED_NOT_PASTED = "copied_not_pasted"
    PASTE_SENT_UNCONFIRMED = "paste_sent_unconfirmed"
    UNKNOWN = "unknown"

    @classmethod
    def from_raw(cls, value: str | None) -> PasteRecoveryState:
        if not value:
            return cls.UNKNOWN
        try:
            return cls(str(value))
        except ValueError:
            return cls.UNKNOWN


@dataclass
class PasteFlowResult:
    ok: bool
    cmd_v_sent: int
    phase: str
    effect_verified: bool
    recovery_state: PasteRecoveryState
    next_action: str = ""
    copy_trigger: str = ""
    permission_sheet_seen: bool = False
    error: str = ""
    image_path: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def is_strictly_completed(self) -> bool:
        return (
            self.ok
            and self.cmd_v_sent == 1
            and self.phase == "completed"
            and self.effect_verified is True
            and self.recovery_state == PasteRecoveryState.COMPLETED
        )


@dataclass(frozen=True)
class SkillCheckResult:
    stop: bool
    counted: bool
    allowed: bool
    reason: str
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SkillPrepareResult:
    stop: bool
    counted: bool
    allowed: bool
    reason: str
    reservation: str = ""
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SkillConfirmResult:
    stop: bool
    counted: bool
    allowed: bool
    reason: str
    raw: dict[str, Any] = field(default_factory=dict)


class SendStatus(str, Enum):
    DRY_RUN_READY = "dry_run_ready"
    SENT_CONFIRMED = "sent_confirmed"
    SEND_UNCONFIRMED = "send_unconfirmed"
    ABORTED_POLICY = "aborted_policy"
    ABORTED_PREFLIGHT = "aborted_preflight"
    ABORTED_NAVIGATION = "aborted_navigation"
    ABORTED_EXISTING_DRAFT = "aborted_existing_draft"
    ABORTED_PASTE_RECOVERY = "aborted_paste_recovery"
    ABORTED_PASTE_VERIFY = "aborted_paste_verify"
    ABORTED_RISK_PHRASE = "aborted_risk_phrase"
    ABORTED_RATE_LIMIT = "aborted_rate_limit"
    ABORTED_SEND_BUTTON = "aborted_send_button"
    ABORTED_POINTER = "aborted_pointer"


@dataclass
class SendReport:
    status: SendStatus
    target: str
    text: str
    dry_run: bool
    message: str
    run_id: str = ""
    evidence_dir: str = ""
    recovery_state: PasteRecoveryState | None = None
    next_action: str = ""
    observed_draft: str | None = None
    paste_receipt: dict[str, Any] | None = None
    confirm_result: dict[str, Any] | None = None
    steps: list[str] = field(default_factory=list)

    @property
    def succeeded(self) -> bool:
        if self.dry_run:
            return self.status == SendStatus.DRY_RUN_READY
        return self.status == SendStatus.SENT_CONFIRMED

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "succeeded": self.succeeded,
            "target": self.target,
            "dry_run": self.dry_run,
            "message": self.message,
            "run_id": self.run_id,
            "evidence_dir": self.evidence_dir,
            "recovery_state": self.recovery_state.value if self.recovery_state else None,
            "next_action": self.next_action,
            "observed_draft": self.observed_draft,
            "paste_receipt": self.paste_receipt,
            "confirm_result": self.confirm_result,
            "steps": list(self.steps),
        }
