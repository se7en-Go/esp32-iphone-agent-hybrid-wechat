"""WeChat iOS navigation workflow: desktop icon body direct click, fallback Spotlight, chat list click."""

from __future__ import annotations

import time
from collections.abc import Callable

from wechat_agent.adapters.driver import PhoneDriver
from wechat_agent.evidence import RunEvidence
from wechat_agent.models import Box, ClickResult, Observation, OcrBox
from wechat_agent.vision.geometry import normalize_text
from wechat_agent.vision.ocr import OcrEngine
from wechat_agent.vision.wechat_ui import (
    ScreenKind,
    detect_screen_context,
    find_chat_row_box,
    find_home_wechat_icon,
    find_spotlight_wechat_target,
    has_exact_chat_title,
)

from .click_chain import ClickChain, PointerConflictError

_SYNC_HEADER_TOKENS = ("收取中", "连接中", "receiving", "connecting")
DEFAULT_SYNC_POLL_INTERVAL = 1.2
DEFAULT_MAX_SYNC_WAIT_ATTEMPTS = 12
DEFAULT_CONFLICT_BACKOFF_SECONDS: tuple[float, ...] = (0.8, 1.5, 2.2, 2.5, 3.0)


def find_chat_list_sync_label(ocr_boxes: list[OcrBox]) -> str | None:
    """Return the normalized WeChat sync/connecting header label if the chat list is still syncing."""
    for item in ocr_boxes:
        if not (0.03 <= item.box.top and item.box.bottom <= 0.14 and 0.14 <= item.box.center[0] <= 0.86):
            continue
        norm = normalize_text(item.text)
        lowered = norm.lower()
        if any(tok in lowered for tok in _SYNC_HEADER_TOKENS):
            return norm
    return None


class NavigationError(RuntimeError):
    """Raised when WeChat or target conversation cannot be opened reliably."""


class WeChatNavigator:
    def __init__(
        self,
        driver: PhoneDriver,
        ocr: OcrEngine,
        click_chain: ClickChain,
        evidence: RunEvidence,
        *,
        allow_group_count: bool = False,
        sleeper: Callable[[float], None] | None = None,
        sync_poll_interval: float = DEFAULT_SYNC_POLL_INTERVAL,
        max_sync_wait_attempts: int = DEFAULT_MAX_SYNC_WAIT_ATTEMPTS,
        conflict_backoff_seconds: tuple[float, ...] = DEFAULT_CONFLICT_BACKOFF_SECONDS,
        max_ocr_miss_reobserves: int = 1,
    ) -> None:
        self.driver = driver
        self.ocr = ocr
        self.click_chain = click_chain
        self.evidence = evidence
        self.allow_group_count = allow_group_count
        self._sleeper = sleeper
        self.sync_poll_interval = float(sync_poll_interval)
        self.max_sync_wait_attempts = max(0, int(max_sync_wait_attempts))
        self.conflict_backoff_seconds = tuple(float(s) for s in conflict_backoff_seconds)
        self.max_ocr_miss_reobserves = max(0, int(max_ocr_miss_reobserves))

    def _sleep(self, seconds: float) -> None:
        if seconds <= 0:
            return
        if self._sleeper is not None:
            self._sleeper(seconds)
        else:
            time.sleep(seconds)

    def _next_observation(self, click_res: ClickResult, step_name: str) -> Observation:
        if click_res.screen_id and click_res.image_bytes:
            return Observation(
                screen_id=click_res.screen_id,
                image_bytes=click_res.image_bytes,
                image_path=click_res.image_path,
                raw=click_res.raw,
            )
        return self.driver.observe(image_output=self.evidence.image_output_path(step_name))

    def ensure_target_chat(
        self,
        target_title: str,
        initial_obs: Observation,
        initial_ocr: list[OcrBox],
    ) -> tuple[Observation, list[OcrBox]]:
        obs = initial_obs
        ocr_boxes = initial_ocr
        scroll_attempts = 0
        back_attempts = 0
        conflict_retries = 0
        ocr_miss_reobserves = 0
        max_conflict_retries = len(self.conflict_backoff_seconds)
        max_steps = max(12, 7 + max_conflict_retries + self.max_ocr_miss_reobserves)

        for step in range(max_steps):
            kind = detect_screen_context(
                ocr_boxes,
                target_title,
                allow_group_count=self.allow_group_count,
            )
            self.evidence.log_event("nav_screen_context", {"step": step, "kind": kind.value})

            if kind == ScreenKind.TARGET_CHAT and has_exact_chat_title(
                ocr_boxes,
                target_title,
                allow_group_count=self.allow_group_count,
            ):
                return obs, ocr_boxes

            try:
                if kind == ScreenKind.HOME_SCREEN:
                    icon_box = find_home_wechat_icon(ocr_boxes, obs.image_bytes)
                    if icon_box is not None:
                        # Direct icon body click above label; NEVER invoke search_app when icon is visible.
                        click_res = self.click_chain.point_and_click(
                            obs.screen_id,
                            icon_box.center,
                            icon_box,
                            step_name=f"nav_home_wechat_{step}",
                        )
                        obs = self._next_observation(click_res, f"nav_after_home_{step}")
                        ocr_boxes = self.ocr.read_boxes(obs.image_bytes)
                        continue

                    # Fallback only when WeChat icon is NOT visible on the current Home screen.
                    spot_obs = self.driver.search_app(
                        "WeChat",
                        screen_id=obs.screen_id,
                        image_output=self.evidence.image_output_path(f"nav_spotlight_{step}"),
                    )
                    obs = spot_obs
                    ocr_boxes = self.ocr.read_boxes(obs.image_bytes)
                    spot_box = find_spotlight_wechat_target(ocr_boxes, obs.image_bytes)
                    if spot_box is None:
                        raise NavigationError("WeChat not found on current Home screen or in Spotlight results")
                    click_res = self.click_chain.point_and_click(
                        obs.screen_id,
                        spot_box.center,
                        spot_box,
                        step_name=f"nav_spotlight_click_{step}",
                    )
                    obs = self._next_observation(click_res, f"nav_after_spotlight_{step}")
                    ocr_boxes = self.ocr.read_boxes(obs.image_bytes)
                    continue

                if kind == ScreenKind.SPOTLIGHT_RESULTS:
                    spot_box = find_spotlight_wechat_target(ocr_boxes, obs.image_bytes)
                    if spot_box is None:
                        raise NavigationError("WeChat missing in Spotlight results")
                    click_res = self.click_chain.point_and_click(
                        obs.screen_id,
                        spot_box.center,
                        spot_box,
                        step_name=f"nav_spotlight_click_{step}",
                    )
                    obs = self._next_observation(click_res, f"nav_after_spotlight_{step}")
                    ocr_boxes = self.ocr.read_boxes(obs.image_bytes)
                    continue

                if kind == ScreenKind.WECHAT_CHAT_LIST:
                    sync_waits = 0
                    left_chat_list = False
                    while sync_waits < self.max_sync_wait_attempts:
                        sync_label = find_chat_list_sync_label(ocr_boxes)
                        if sync_label is None:
                            break
                        sync_waits += 1
                        self.evidence.log_event(
                            "nav_chat_list_sync_wait",
                            {
                                "step": step,
                                "attempt": sync_waits,
                                "header": sync_label,
                                "sleep_seconds": self.sync_poll_interval,
                            },
                        )
                        self._sleep(self.sync_poll_interval)
                        obs = self.driver.observe(
                            image_output=self.evidence.image_output_path(f"nav_sync_wait_{step}_{sync_waits}")
                        )
                        ocr_boxes = self.ocr.read_boxes(obs.image_bytes)
                        recheck_kind = detect_screen_context(
                            ocr_boxes,
                            target_title,
                            allow_group_count=self.allow_group_count,
                        )
                        if recheck_kind != ScreenKind.WECHAT_CHAT_LIST:
                            left_chat_list = True
                            break

                    if left_chat_list:
                        continue
                    remaining_sync = find_chat_list_sync_label(ocr_boxes)
                    if remaining_sync is not None:
                        self.evidence.log_event(
                            "nav_chat_list_sync_timeout",
                            {"step": step, "attempts": sync_waits, "header": remaining_sync},
                        )

                    row_box = find_chat_row_box(ocr_boxes, target_title)
                    if row_box is not None:
                        click_res = self.click_chain.point_and_click(
                            obs.screen_id,
                            row_box.center,
                            row_box,
                            step_name=f"nav_chat_row_{step}",
                        )
                        obs = self._next_observation(click_res, f"nav_after_chat_row_{step}")
                        ocr_boxes = self.ocr.read_boxes(obs.image_bytes)
                        continue

                    if scroll_attempts < 2:
                        scroll_attempts += 1
                        obs = self.driver.scroll(
                            0.5,
                            0.45,
                            -5,
                            screen_id=obs.screen_id,
                            image_output=self.evidence.image_output_path(f"nav_scroll_{step}"),
                        )
                        ocr_boxes = self.ocr.read_boxes(obs.image_bytes)
                        continue
                    if ocr_miss_reobserves < self.max_ocr_miss_reobserves:
                        ocr_miss_reobserves += 1
                        self.evidence.log_event(
                            "nav_ocr_miss_reobserve",
                            {"step": step, "attempt": ocr_miss_reobserves, "kind": kind.value},
                        )
                        self._sleep(0.3)
                        obs = self.driver.observe(
                            image_output=self.evidence.image_output_path(
                                f"nav_chat_list_reobs_{step}_{ocr_miss_reobserves}"
                            )
                        )
                        ocr_boxes = self.ocr.read_boxes(obs.image_bytes)
                        continue
                    raise NavigationError(f"Conversation {target_title!r} not found in WeChat chat list")

                if kind == ScreenKind.OTHER_CHAT and back_attempts < 1:
                    back_attempts += 1
                    back_box = Box(0.02, 0.038, 0.13, 0.115)
                    click_res = self.click_chain.point_and_click(
                        obs.screen_id,
                        back_box.center,
                        back_box,
                        step_name=f"nav_back_{step}",
                    )
                    obs = self._next_observation(click_res, f"nav_after_back_{step}")
                    ocr_boxes = self.ocr.read_boxes(obs.image_bytes)
                    continue

                if kind == ScreenKind.UNKNOWN and ocr_miss_reobserves < self.max_ocr_miss_reobserves:
                    ocr_miss_reobserves += 1
                    self.evidence.log_event(
                        "nav_ocr_miss_reobserve",
                        {"step": step, "attempt": ocr_miss_reobserves, "kind": kind.value},
                    )
                    self._sleep(0.3)
                    obs = self.driver.observe(
                        image_output=self.evidence.image_output_path(f"nav_ocr_reobs_{step}_{ocr_miss_reobserves}")
                    )
                    ocr_boxes = self.ocr.read_boxes(obs.image_bytes)
                    continue

                raise NavigationError(f"Cannot navigate to {target_title!r} from screen state {kind.value}")
            except PointerConflictError as exc:
                conflict_retries += 1
                if conflict_retries > max_conflict_retries:
                    self.evidence.log_event(
                        "nav_pre_click_409_reobserve",
                        {"step": step, "retry": conflict_retries, "detail": exc.detail},
                    )
                    raise NavigationError(
                        f"Navigation exceeded {max_conflict_retries} pre-click 409 re-observations: {exc}"
                    ) from exc
                delay = self.conflict_backoff_seconds[conflict_retries - 1]
                self.evidence.log_event(
                    "nav_pre_click_409_reobserve",
                    {
                        "step": step,
                        "retry": conflict_retries,
                        "sleep_seconds": delay,
                        "detail": exc.detail,
                    },
                )
                self._sleep(delay)
                obs = self.driver.observe(
                    image_output=self.evidence.image_output_path(f"nav_reobs_409_{step}")
                )
                ocr_boxes = self.ocr.read_boxes(obs.image_bytes)
                continue

        raise NavigationError(f"Exceeded max navigation steps trying to enter {target_title!r}")
