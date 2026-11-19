"""Persistent dual-layer rate limiter, quiet hours, human takeover cooldown, and reply content guard."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from wechat_agent.monitor.store import MonitorStore

DEFAULT_FORBIDDEN_PATTERNS: tuple[str, ...] = (
    r"(?:验证码|动态码|校验码|短信码|OTP)",
    r"(?:密码|口令|PIN码|支付码)",
    r"(?:红包|转账|打款|汇款|收款码|付款码|支付宝口令)",
    r"(?<!\d)1[3-9]\d{9}(?!\d)",
    r"(?<!\d)\d{17}[\dXx](?!\d)",
    r"(?:稳赚不赔|保本收益|包治百病|确诊为|法律担保)",
)


@dataclass(frozen=True)
class RateLimitConfig:
    global_min_interval_seconds: float = 15.0
    global_max_per_hour: int = 15
    per_group_cooldown_seconds: float = 60.0
    per_group_max_per_hour: int = 6
    human_takeover_cooldown_seconds: float = 600.0
    quiet_hours: str = "23:00-07:00"
    max_reply_chars: int = 240
    forbidden_reply_patterns: tuple[str, ...] = DEFAULT_FORBIDDEN_PATTERNS

    def __post_init__(self) -> None:
        if self.global_min_interval_seconds < 3.0:
            raise ValueError("global_min_interval_seconds cannot be lower than base floor 3.0s")
        if not (1 <= self.global_max_per_hour <= 30):
            raise ValueError("global_max_per_hour must be within 1..30")
        if self.per_group_cooldown_seconds < 0 or self.per_group_max_per_hour < 1:
            raise ValueError("invalid per-group rate limit configuration")


@dataclass(frozen=True)
class RateLimitDecision:
    allowed: bool
    reason: str = ""
    block_kind: str = ""


def _parse_quiet_range(spec: str) -> tuple[tuple[int, int], tuple[int, int]] | None:
    clean = (spec or "").strip()
    if not clean or clean.lower() == "off":
        return None
    if "-" not in clean:
        raise ValueError(f"Invalid quiet_hours format: {spec!r}")
    left, right = clean.split("-", 1)
    sh, _, sm = left.partition(":")
    eh, _, em = right.partition(":")
    return (int(sh), int(sm or "0")), (int(eh), int(em or "0"))


def _in_quiet_window(moment: datetime, start: tuple[int, int], end: tuple[int, int]) -> bool:
    cur = moment.hour * 60 + moment.minute
    begin = start[0] * 60 + start[1]
    stop = end[0] * 60 + end[1]
    if begin == stop:
        return False
    if begin < stop:
        return begin <= cur < stop
    return cur >= begin or cur < stop


class MonitorRateLimiter:
    """Enforces quiet hours, human takeover silence, and persistent per-group/global rate limits."""

    def __init__(
        self,
        store: MonitorStore,
        config: RateLimitConfig | None = None,
        *,
        clock=time.time,
    ) -> None:
        self.store = store
        self.config = config or RateLimitConfig()
        self._clock = clock
        self._quiet_bounds = _parse_quiet_range(self.config.quiet_hours)
        self._compiled_forbidden = tuple(
            re.compile(pat, re.IGNORECASE) for pat in self.config.forbidden_reply_patterns
        )

    def is_quiet_hours(self, now: float | None = None) -> bool:
        if self._quiet_bounds is None:
            return False
        ts = self._clock() if now is None else float(now)
        moment = datetime.fromtimestamp(ts)
        return _in_quiet_window(moment, *self._quiet_bounds)

    def activate_human_takeover(self, group_title: str, now: float | None = None) -> float:
        ts = self._clock() if now is None else float(now)
        until_ts = ts + self.config.human_takeover_cooldown_seconds
        self.store.set_human_takeover(group_title, until_ts)
        return until_ts

    def check_can_reply(
        self,
        group_title: str,
        now: float | None = None,
        *,
        per_group_cooldown_override: float | None = None,
        per_group_max_per_hour_override: int | None = None,
    ) -> RateLimitDecision:
        ts = self._clock() if now is None else float(now)
        if self.is_quiet_hours(ts):
            return RateLimitDecision(
                allowed=False,
                reason=f"Quiet hours active ({self.config.quiet_hours})",
                block_kind="quiet_hours",
            )

        takeover_until = self.store.get_takeover_until(group_title)
        if takeover_until > ts:
            remaining = int(round(takeover_until - ts))
            return RateLimitDecision(
                allowed=False,
                reason=f"Human takeover cooldown active ({remaining}s remaining)",
                block_kind="human_takeover",
            )

        group_cooldown = (
            float(per_group_cooldown_override)
            if per_group_cooldown_override is not None
            else self.config.per_group_cooldown_seconds
        )
        group_max_hour = (
            int(per_group_max_per_hour_override)
            if per_group_max_per_hour_override is not None
            else self.config.per_group_max_per_hour
        )

        hour_ago = ts - 3600.0
        group_sends = self.store.recent_send_timestamps(hour_ago, group_title=group_title)
        if group_sends and (ts - group_sends[-1]) < group_cooldown:
            return RateLimitDecision(
                allowed=False,
                reason=f"Per-group cooldown active (< {group_cooldown:.0f}s)",
                block_kind="group_cooldown",
            )
        if len(group_sends) >= group_max_hour:
            return RateLimitDecision(
                allowed=False,
                reason=f"Per-group hourly limit reached ({group_max_hour}/h)",
                block_kind="group_hourly_cap",
            )

        global_sends = self.store.recent_send_timestamps(hour_ago, group_title=None)
        if global_sends and (ts - global_sends[-1]) < self.config.global_min_interval_seconds:
            return RateLimitDecision(
                allowed=False,
                reason=f"Global minimum interval active (< {self.config.global_min_interval_seconds:.0f}s)",
                block_kind="global_interval",
            )
        if len(global_sends) >= self.config.global_max_per_hour:
            return RateLimitDecision(
                allowed=False,
                reason=f"Global hourly limit reached ({self.config.global_max_per_hour}/h)",
                block_kind="global_hourly_cap",
            )

        return RateLimitDecision(allowed=True)

    def check_reply_content(self, text: str) -> tuple[bool, str]:
        if not isinstance(text, str) or not text.strip():
            return False, "Reply text is empty"
        stripped = text.strip()
        if len(stripped) > self.config.max_reply_chars:
            return False, f"Reply exceeds max_reply_chars ({len(stripped)} > {self.config.max_reply_chars})"
        for pat in self._compiled_forbidden:
            if pat.search(stripped):
                return False, f"Reply matched forbidden pattern: {pat.pattern}"
        return True, ""
