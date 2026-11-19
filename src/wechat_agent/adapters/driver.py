"""Abstract PhoneDriver protocol and exception hierarchy."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from wechat_agent.models import (
    Box,
    ClickResult,
    Observation,
    PasteFlowResult,
    PointResult,
    SkillCheckResult,
    SkillConfirmResult,
    SkillPrepareResult,
)


class DriverError(RuntimeError):
    """Error returned by the base iPhone control CLI."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        detail: str = "",
        raw: dict[str, Any] | None = None,
        command: list[str] | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.detail = detail or message
        self.raw = raw or {}
        self.command = command or []


class ConflictError(DriverError):
    """HTTP 409 conflict (e.g., screen changed / blinking caret rejection)."""


@runtime_checkable
class PhoneDriver(Protocol):
    def status(self) -> dict[str, Any]: ...

    def capabilities(self) -> dict[str, Any]: ...

    def skill_list(self) -> list[str]: ...

    def skill_show(self, name: str) -> dict[str, Any]: ...

    def skill_check(self, observed_text: str, skill: str = "") -> SkillCheckResult: ...

    def skill_prepare_send(self, skill: str = "") -> SkillPrepareResult: ...

    def skill_confirm_send(
        self,
        bubble_visible: bool,
        observed_text: str = "",
        skill: str = "",
    ) -> SkillConfirmResult: ...

    def observe(
        self,
        *,
        for_click: bool = False,
        image_output: Path | None = None,
    ) -> Observation: ...

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
    ) -> PointResult: ...

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
    ) -> ClickResult: ...

    def find_text(
        self,
        text: str,
        *,
        screen_id: str,
        nth: int = 1,
        exact: bool = False,
        region: Box | None = None,
    ) -> dict[str, Any]: ...

    def search_app(
        self,
        text: str,
        *,
        screen_id: str,
        image_output: Path | None = None,
    ) -> Observation: ...

    def scroll(
        self,
        x: float,
        y: float,
        lines: int,
        *,
        screen_id: str,
        image_output: Path | None = None,
    ) -> Observation: ...

    def paste_flow(
        self,
        payload_path: Path,
        *,
        target_box: Box,
        expect_title: str,
        caret_box: Box | None = None,
        evidence_dir: Path | None = None,
        image_output: Path | None = None,
    ) -> PasteFlowResult: ...
