"""WeChat iOS navigation workflow: desktop icon body direct click, fallback Spotlight, chat list click."""

from __future__ import annotations

import time

from wechat_agent.adapters.driver import PhoneDriver
from wechat_agent.evidence import RunEvidence
from wechat_agent.models import Box, ClickResult, Observation, OcrBox
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


class NavigationError(RuntimeError):
    """Raised when WeChat or target conversation cannot be opened reliably."""


class WeChatNavigator:
    def __init__(
        self,
        driver: PhoneDriver,
        ocr: OcrEngine,
        click_chain: ClickChain,
        evidence: RunEvidence,
    ) -> None:
        self.driver = driver
        self.ocr = ocr
        self.click_chain = click_chain
        self.evidence = evidence

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
        max_conflict_retries = 3

        for step in range(9):
            kind = detect_screen_context(ocr_boxes, target_title)
            self.evidence.log_event("nav_screen_context", {"step": step, "kind": kind.value})

            if kind == ScreenKind.TARGET_CHAT and has_exact_chat_title(ocr_boxes, target_title):
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
                    raise NavigationError(f"Conversation {target_title!r} not found in WeChat chat list")

                if kind == ScreenKind.OTHER_CHAT and back_attempts < 1:
                    back_attempts += 1
                    back_box = Box(0.02, 0.045, 0.13, 0.115)
                    click_res = self.click_chain.point_and_click(
                        obs.screen_id,
                        back_box.center,
                        back_box,
                        step_name=f"nav_back_{step}",
                    )
                    obs = self._next_observation(click_res, f"nav_after_back_{step}")
                    ocr_boxes = self.ocr.read_boxes(obs.image_bytes)
                    continue

                raise NavigationError(f"Cannot navigate to {target_title!r} from screen state {kind.value}")
            except PointerConflictError as exc:
                conflict_retries += 1
                self.evidence.log_event(
                    "nav_pre_click_409_reobserve",
                    {"step": step, "retry": conflict_retries, "detail": exc.detail},
                )
                if conflict_retries > max_conflict_retries:
                    raise NavigationError(
                        f"Navigation exceeded {max_conflict_retries} pre-click 409 re-observations: {exc}"
                    ) from exc
                time.sleep(0.25)
                obs = self.driver.observe(
                    image_output=self.evidence.image_output_path(f"nav_reobs_409_{step}")
                )
                ocr_boxes = self.ocr.read_boxes(obs.image_bytes)
                continue

        raise NavigationError(f"Exceeded max navigation steps trying to enter {target_title!r}")
