"""Configuration loader for WeChat automation CLI."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

DEFAULT_IPHONE_AGENT_CLI = "/Users/seven/Documents/esp32-iphone-agent-hybrid-new/server/.venv/bin/iphone-agent"
DEFAULT_IPHONE_AGENT_URL = "http://127.0.0.1:8890"
DEFAULT_ALLOWED_TARGETS = ("文件传输助手",)


def require_bool(value: Any, field_name: str, *, default: bool | None = None) -> bool:
    """Strictly validate a YAML/config boolean flag (rejects truthy strings like 'false')."""
    if value is None:
        if default is not None:
            return default
        raise ValueError(f"{field_name} must be a boolean (true/false), got None")
    if not isinstance(value, bool):
        raise ValueError(
            f"{field_name} must be a YAML boolean (true/false), got {type(value).__name__}: {value!r}"
        )
    return value


def clamp_timeout(
    value: Any,
    field_name: str,
    *,
    default: float,
    min_seconds: float,
    max_seconds: float | None = None,
) -> float:
    """Parse a finite numeric timeout (rejecting booleans/NaN/Inf) and clamp to safe bounds."""
    if value is None:
        return default
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be a number, got bool: {value!r}")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"{field_name} must be finite, got {value!r}")
    clamped = max(min_seconds, parsed)
    if max_seconds is not None:
        clamped = min(max_seconds, clamped)
    return clamped


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class WeChatSendConfig:
    iphone_agent_cli: str = DEFAULT_IPHONE_AGENT_CLI
    iphone_agent_url: str = DEFAULT_IPHONE_AGENT_URL
    skill_name: str = "wechat-send"
    allowed_targets: tuple[str, ...] = DEFAULT_ALLOWED_TARGETS
    ocr_backend: str = "rapidocr"
    evidence_root: Path = Path(".local-data/runs")
    cli_timeout_seconds: float = 90.0
    paste_flow_timeout_seconds: float = 180.0
    point_rehome_retries: int = 1
    pre_send_point_retries: int = 3
    require_mounted_skill: bool = True
    allow_group_count: bool = False

    def resolved_evidence_root(self) -> Path:
        if self.evidence_root.is_absolute():
            return self.evidence_root
        return (repo_root() / self.evidence_root).resolve()

    def skills_dir(self) -> Path:
        return (repo_root() / "skills").resolve()


def _parse_yaml_config(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError(f"Config file {path} must contain a YAML mapping")
    return raw


def load_config(
    config_path: str | Path | None = None,
    *,
    allowed_targets_override: list[str] | tuple[str, ...] | None = None,
    skill_name_override: str | None = None,
    ocr_backend_override: str | None = None,
    cli_override: str | None = None,
    url_override: str | None = None,
) -> WeChatSendConfig:
    """Load config from optional YAML file, environment variables, and explicit overrides."""
    data: dict[str, Any] = {}
    if config_path is not None:
        path = Path(config_path)
        if not path.is_file():
            raise FileNotFoundError(f"Config file not found: {path}")
        data = _parse_yaml_config(path)

    cli_path = (
        cli_override
        or os.environ.get("IPHONE_AGENT_CLI")
        or str(data.get("iphone_agent_cli") or DEFAULT_IPHONE_AGENT_CLI)
    )
    url = (
        url_override
        or os.environ.get("IPHONE_AGENT_URL")
        or str(data.get("iphone_agent_url") or DEFAULT_IPHONE_AGENT_URL)
    )
    skill_name = skill_name_override or str(data.get("skill_name") or "wechat-send")
    ocr_backend = ocr_backend_override or str(data.get("ocr_backend") or "rapidocr")
    evidence_root = Path(str(data.get("evidence_root") or ".local-data/runs"))
    cli_timeout = clamp_timeout(data.get("cli_timeout_seconds"), "cli_timeout_seconds", default=90.0, min_seconds=5.0)
    paste_flow_timeout = max(
        cli_timeout,
        clamp_timeout(
            data.get("paste_flow_timeout_seconds"),
            "paste_flow_timeout_seconds",
            default=180.0,
            min_seconds=15.0,
        ),
    )
    point_rehome_retries = max(0, int(data.get("point_rehome_retries", 1)))
    pre_send_point_retries = max(1, int(data.get("pre_send_point_retries", 3)))
    require_mounted_skill = require_bool(
        data.get("require_mounted_skill"), "require_mounted_skill", default=True
    )
    allow_group_count = require_bool(data.get("allow_group_count"), "allow_group_count", default=False)

    if allowed_targets_override is not None:
        raw_targets = list(allowed_targets_override)
    elif "allowed_targets" in data:
        raw_targets = list(data["allowed_targets"] or [])
    else:
        raw_targets = list(DEFAULT_ALLOWED_TARGETS)

    allowed_targets = tuple(str(t).strip() for t in raw_targets if str(t).strip())

    return WeChatSendConfig(
        iphone_agent_cli=cli_path,
        iphone_agent_url=url,
        skill_name=skill_name,
        allowed_targets=allowed_targets,
        ocr_backend=ocr_backend,
        evidence_root=evidence_root,
        cli_timeout_seconds=cli_timeout,
        paste_flow_timeout_seconds=paste_flow_timeout,
        point_rehome_retries=point_rehome_retries,
        pre_send_point_retries=pre_send_point_retries,
        require_mounted_skill=require_mounted_skill,
        allow_group_count=allow_group_count,
    )
