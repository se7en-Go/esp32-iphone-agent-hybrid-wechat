"""Atomic pointer-and-click execution chains obeying iOS 3-second pointer auto-hide rules."""

from __future__ import annotations

from collections.abc import Callable

from wechat_agent.adapters.driver import ConflictError, PhoneDriver
from wechat_agent.evidence import RunEvidence
from wechat_agent.models import Box, ClickResult, MeasuredPointer, Observation, PointResult, SendStatus


class PointerNotConfirmedError(RuntimeError):
    """Raised when measured_pointer stays null or falls outside the target_box."""


class SendPreconditionError(PointerNotConfirmedError):
    """Raised BEFORE click dispatch when a fresh frame fails title/draft/send-button verification."""

    def __init__(self, message: str, *, status: SendStatus = SendStatus.ABORTED_POINTER) -> None:
        super().__init__(message)
        self.status = status


class PointerConflictError(ConflictError):
    """Raised when HTTP 409 occurs BEFORE `click_observed` is dispatched (zero clicks sent)."""


class ClickChain:
    """Executes tightly coupled `point -> observe --for-click -> click-observed` sequences.

    Never inserts OCR or arbitrary delays between `point` and `click_observed`, and strictly uses
    `measured_pointer` (JSON object) rather than requested coordinates or OCR text centers.
    """

    def __init__(
        self,
        driver: PhoneDriver,
        evidence: RunEvidence,
        *,
        rehome_retries: int = 1,
    ) -> None:
        self.driver = driver
        self.evidence = evidence
        self.rehome_retries = max(0, int(rehome_retries))
        self.last_verified_send_box: Box | None = None
        self.last_verified_send_label: str | None = None
        self.last_verified_caret_scope: Box | None = None

    def _confirm_pointer(
        self,
        screen_id: str,
        aim_xy: tuple[float, float],
        target_box: Box,
        *,
        step_name: str,
        rehome: bool = False,
        caret_scope: Box | None = None,
    ) -> tuple[MeasuredPointer, PointResult]:
        aim_x, aim_y = aim_xy
        if not target_box.contains(aim_x, aim_y):
            raise ValueError(f"Aim coordinates {aim_xy} are outside target_box {target_box.as_tuple()}")

        current_sid = screen_id
        for attempt in range(self.rehome_retries + 1):
            use_rehome = rehome or (attempt > 0)
            try:
                pt_res = self.driver.point(
                    aim_x,
                    aim_y,
                    screen_id=current_sid,
                    target_box=target_box,
                    rehome=use_rehome,
                    caret_scope=caret_scope,
                    image_output=self.evidence.image_output_path(f"{step_name}_point_{attempt}"),
                )
            except ConflictError as exc:
                raise PointerConflictError(
                    str(exc),
                    status_code=exc.status_code,
                    detail=exc.detail,
                    raw=exc.raw,
                    command=exc.command,
                ) from exc
            mp = pt_res.measured_pointer
            if mp is not None and target_box.contains(mp.x, mp.y):
                self.evidence.log_event(
                    f"{step_name}_pointer_confirmed",
                    {"attempt": attempt, "rehome": use_rehome, "measured_pointer": {"x": mp.x, "y": mp.y}},
                )
                return mp, pt_res

            self.evidence.log_event(
                f"{step_name}_pointer_unconfirmed",
                {
                    "attempt": attempt,
                    "rehome": use_rehome,
                    "measured_pointer": {"x": mp.x, "y": mp.y} if mp else None,
                    "target_box": target_box.as_list(),
                },
            )
            if attempt < self.rehome_retries:
                fresh = self.driver.observe(
                    image_output=self.evidence.image_output_path(f"{step_name}_reobs_{attempt}")
                )
                current_sid = fresh.screen_id

        raise PointerNotConfirmedError(
            f"measured_pointer could not be confirmed inside target_box {target_box.as_tuple()} for {step_name}"
        )

    def point_and_click(
        self,
        screen_id: str,
        aim_xy: tuple[float, float],
        target_box: Box,
        *,
        step_name: str,
        target_label: str = "",
        rehome: bool = False,
        caret_scope: Box | None = None,
    ) -> ClickResult:
        """Atomically move pointer, observe --for-click, and click at `measured_pointer`."""
        mp, _ = self._confirm_pointer(
            screen_id,
            aim_xy,
            target_box,
            step_name=step_name,
            rehome=rehome,
            caret_scope=caret_scope,
        )
        # Execute observe(--for-click) -> click-observed immediately with zero intervening OCR/sleep.
        try:
            obs_click = self.driver.observe(
                for_click=True,
                image_output=self.evidence.image_output_path(f"{step_name}_for_click"),
            )
        except ConflictError as exc:
            raise PointerConflictError(
                str(exc),
                status_code=exc.status_code,
                detail=exc.detail,
                raw=exc.raw,
                command=exc.command,
            ) from exc
        click_res = self.driver.click_observed(
            mp.x,
            mp.y,
            screen_id=obs_click.screen_id,
            target_box=target_box,
            target_label=target_label,
            caret_scope=caret_scope,
            image_output=self.evidence.image_output_path(f"{step_name}_clicked"),
        )
        self.evidence.log_event(
            f"{step_name}_clicked",
            {
                "pointer": {"x": mp.x, "y": mp.y},
                "target_box": target_box.as_list(),
                "caret_scope": caret_scope.as_list() if caret_scope else None,
                "clicked": click_res.clicked,
                "command_acknowledged": click_res.command_acknowledged,
            },
        )
        return click_res

    def pre_send_point_with_caret_retry(
        self,
        aim_xy: tuple[float, float],
        target_box: Box,
        *,
        step_name: str,
        max_attempts: int = 3,
        initial_screen_id: str | None = None,
        caret_scope: Box | None = None,
        reverify_observation: Callable[[Observation], tuple[Box, str, Box | None]] | None = None,
    ) -> tuple[MeasuredPointer, PointResult]:
        """Move pointer to the Send button before clicking; retries ONLY pointer move on HTTP 409 caret blink.

        When a fresh observation is taken (e.g. after 409), optional `reverify_observation(obs)`
        re-verifies title/draft and dynamically relocates the Send button and caret_scope on that
        fresh frame so stale coordinates are never reused. NEVER invokes `click_observed`.
        """
        aim_x, aim_y = aim_xy
        if not target_box.contains(aim_x, aim_y):
            raise ValueError(f"Send aim {aim_xy} is outside target_box {target_box.as_tuple()}")

        last_err: Exception | None = None
        sid = initial_screen_id
        current_box = target_box
        current_caret = caret_scope
        self.last_verified_send_box = target_box
        self.last_verified_send_label = None
        self.last_verified_caret_scope = caret_scope
        for attempt in range(max(1, max_attempts)):
            try:
                if not sid:
                    obs = self.driver.observe(
                        image_output=self.evidence.image_output_path(f"{step_name}_obs_{attempt}")
                    )
                    sid = obs.screen_id
                    if reverify_observation is not None:
                        fresh_box, fresh_label, fresh_caret = reverify_observation(obs)
                        current_box = fresh_box
                        current_caret = fresh_caret
                        aim_x, aim_y = fresh_box.center
                        self.last_verified_send_box = fresh_box
                        self.last_verified_send_label = fresh_label
                        self.last_verified_caret_scope = fresh_caret
                        self.evidence.log_event(
                            f"{step_name}_reverified_frame",
                            {
                                "attempt": attempt,
                                "screen_id": sid,
                                "send_box": fresh_box.as_list(),
                                "label": fresh_label,
                            },
                        )
                pt_res = self.driver.point(
                    aim_x,
                    aim_y,
                    screen_id=sid,
                    target_box=current_box,
                    rehome=(attempt > 0),
                    caret_scope=current_caret,
                    image_output=self.evidence.image_output_path(f"{step_name}_point_{attempt}"),
                )
                mp = pt_res.measured_pointer
                if mp is not None and current_box.contains(mp.x, mp.y):
                    self.last_verified_send_box = current_box
                    self.last_verified_caret_scope = current_caret
                    self.evidence.log_event(
                        f"{step_name}_pointer_ready",
                        {
                            "attempt": attempt,
                            "measured_pointer": {"x": mp.x, "y": mp.y},
                            "caret_scope": current_caret.as_list() if current_caret else None,
                        },
                    )
                    return mp, pt_res
                self.evidence.log_event(
                    f"{step_name}_pointer_retry_misaligned",
                    {"attempt": attempt, "measured_pointer": {"x": mp.x, "y": mp.y} if mp else None},
                )
            except ConflictError as exc:
                last_err = exc
                self.evidence.log_event(
                    f"{step_name}_409_caret_retry_pointer_only",
                    {"attempt": attempt, "detail": exc.detail},
                )
            sid = None

        if last_err is not None:
            raise PointerNotConfirmedError(
                f"Pre-send point failed after {max_attempts} pointer-only attempts: {last_err}"
            ) from last_err
        raise PointerNotConfirmedError(
            f"Pre-send measured_pointer not confirmed in {current_box.as_tuple()} after {max_attempts} attempts"
        )

    def single_shot_send_click(
        self,
        measured_pointer: MeasuredPointer,
        target_box: Box,
        target_label: str,
        *,
        step_name: str = "send_click",
        caret_scope: Box | None = None,
        verify_for_click: Callable[[Observation], tuple[Box, str, Box | None]] | None = None,
    ) -> ClickResult:
        """Execute ONE observe(--for-click) -> click-observed with target_label. NEVER retries on failure.

        When `verify_for_click` is supplied, validates the fresh `for_click=True` frame (title,
        pasted draft, and fresh Send button bounds) BEFORE dispatching `click_observed`. Any
        conflict/precondition failure prior to `click_observed` raises `PointerConflictError` or
        `SendPreconditionError` (zero click sent); any failure from `click_observed` itself stays
        a plain `DriverError`/`ConflictError` so callers treat it as unconfirmed and never retry.
        """
        if not target_box.contains(measured_pointer.x, measured_pointer.y):
            raise PointerNotConfirmedError(
                f"measured_pointer {(measured_pointer.x, measured_pointer.y)} outside {target_box.as_tuple()}"
            )
        if not target_label:
            raise ValueError("Sending control click requires non-empty target_label")

        try:
            obs_click = self.driver.observe(
                for_click=True,
                image_output=self.evidence.image_output_path(f"{step_name}_for_click"),
            )
        except ConflictError as exc:
            raise PointerConflictError(
                str(exc),
                status_code=exc.status_code,
                detail=exc.detail,
                raw=exc.raw,
                command=exc.command,
            ) from exc

        click_box = target_box
        click_label = target_label
        click_caret = caret_scope
        if verify_for_click is not None:
            fresh_box, fresh_label, fresh_caret = verify_for_click(obs_click)
            if not fresh_box.contains(measured_pointer.x, measured_pointer.y):
                raise SendPreconditionError(
                    f"Blue Send button shifted on for_click frame: pointer {(measured_pointer.x, measured_pointer.y)} "
                    f"is outside fresh box {fresh_box.as_tuple()}",
                    status=SendStatus.ABORTED_POINTER,
                )
            click_box = fresh_box
            click_label = fresh_label
            click_caret = fresh_caret

        click_res = self.driver.click_observed(
            measured_pointer.x,
            measured_pointer.y,
            screen_id=obs_click.screen_id,
            target_box=click_box,
            target_label=click_label,
            caret_scope=click_caret,
            image_output=self.evidence.image_output_path(f"{step_name}_clicked"),
        )
        self.evidence.log_event(
            f"{step_name}_executed",
            {
                "pointer": {"x": measured_pointer.x, "y": measured_pointer.y},
                "target_label": click_label,
                "target_box": click_box.as_list(),
                "caret_scope": click_caret.as_list() if click_caret else None,
                "clicked": click_res.clicked,
                "command_acknowledged": click_res.command_acknowledged,
            },
        )
        return click_res
