"""Skill frontmatter contract validation and base skill-check/prepare/confirm wrappers."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from wechat_agent.adapters.driver import PhoneDriver
from wechat_agent.models import OcrBox, SkillCheckResult, SkillConfirmResult, SkillPrepareResult

_QUIET_RE = re.compile(r"^\d{1,2}:\d{2}-\d{1,2}:\d{2}$")


def parse_skill_frontmatter(markdown: str) -> dict[str, Any]:
    """Parse SKILL.md frontmatter strictly following the base server flat key: value contract."""
    if not markdown.startswith("---"):
        raise ValueError("SKILL.md must start with '---' frontmatter fence")
    end = markdown.find("\n---", 3)
    if end < 0:
        raise ValueError("SKILL.md missing closing '---' frontmatter fence")

    raw: dict[str, str] = {}
    for line_no, line in enumerate(markdown[3:end].splitlines(), start=2):
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("- "):
            raise ValueError(
                f"Line {line_no}: base frontmatter parser does not support YAML lists; use comma-separated strings"
            )
        if ":" not in line:
            raise ValueError(f"Line {line_no}: expected 'key: value', got {line!r}")
        key, value = line.split(":", 1)
        clean_val = value.strip().strip('"').strip("'")
        if clean_val.startswith("[") and clean_val.endswith("]"):
            raise ValueError(
                f"Line {line_no}: base frontmatter parser does not support bracket lists; use comma-separated strings"
            )
        raw[key.strip()] = clean_val

    parsed: dict[str, Any] = {}
    if "min_interval_seconds" in raw:
        val = float(raw["min_interval_seconds"])
        if val < 3.0:
            raise ValueError(f"min_interval_seconds ({val}) cannot be stricter/lower than base floor 3.0s")
        parsed["min_interval_seconds"] = val
    if "max_per_hour" in raw:
        val_int = int(float(raw["max_per_hour"]))
        if not (1 <= val_int <= 30):
            raise ValueError(f"max_per_hour ({val_int}) must be within 1..30")
        parsed["max_per_hour"] = val_int
    if "quiet_hours" in raw:
        qh = raw["quiet_hours"]
        if qh != "off" and not _QUIET_RE.match(qh):
            raise ValueError(f"quiet_hours must be 'off' or 'HH:MM-HH:MM', got {qh!r}")
        parsed["quiet_hours"] = qh
    if "extra_stop" in raw:
        parsed["extra_stop"] = [part.strip() for part in raw["extra_stop"].split(",") if part.strip()]
    else:
        parsed["extra_stop"] = []

    return parsed


def validate_skills_dir(skills_dir: Path) -> list[str]:
    """Validate all SKILL.md files under `skills_dir` and enforce zero static layout.json files."""
    if not skills_dir.is_dir():
        raise FileNotFoundError(f"Skills directory not found: {skills_dir}")
    static_layouts = sorted(skills_dir.glob("*/layout.json"))
    if static_layouts:
        names = ", ".join(str(p.relative_to(skills_dir)) for p in static_layouts)
        raise ValueError(f"Static layout.json is forbidden in this repo, found: {names}")

    skill_files = sorted(skills_dir.glob("*/SKILL.md"))
    if not skill_files:
        raise ValueError(f"No */SKILL.md files found under {skills_dir}")

    validated: list[str] = []
    for path in skill_files:
        parse_skill_frontmatter(path.read_text(encoding="utf-8"))
        validated.append(path.parent.name)
    return validated


class SkillGuard:
    def __init__(self, driver: PhoneDriver, skill_name: str = "wechat-send") -> None:
        self.driver = driver
        self.skill_name = skill_name

    @staticmethod
    def validate_message_text(text: str) -> str:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Message text must not be empty or whitespace-only")
        return text

    @staticmethod
    def join_observed_text(ocr_boxes: list[OcrBox]) -> str:
        return "\n".join(b.text.strip() for b in ocr_boxes if b.text.strip())

    def check_screen_text(self, ocr_boxes: list[OcrBox]) -> SkillCheckResult:
        observed_text = self.join_observed_text(ocr_boxes)
        return self.driver.skill_check(observed_text, skill=self.skill_name)

    def prepare_send(self) -> SkillPrepareResult:
        return self.driver.skill_prepare_send(skill=self.skill_name)

    def confirm_send(self, *, bubble_visible: bool, ocr_boxes: list[OcrBox]) -> SkillConfirmResult:
        observed_text = self.join_observed_text(ocr_boxes)
        return self.driver.skill_confirm_send(
            bubble_visible=bubble_visible,
            observed_text=observed_text,
            skill=self.skill_name,
        )
