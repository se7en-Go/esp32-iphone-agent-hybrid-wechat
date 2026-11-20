"""Configuration loader for M2 WeChat group monitoring and two-stage hybrid auto-reply."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from wechat_agent.config import (
    DEFAULT_IPHONE_AGENT_CLI,
    DEFAULT_IPHONE_AGENT_URL,
    repo_root,
)
from wechat_agent.guards.rate_limit import DEFAULT_FORBIDDEN_PATTERNS, RateLimitConfig
from wechat_agent.monitor.llm_client import LlmConfig
from wechat_agent.monitor.models import MonitorMode


@dataclass(frozen=True)
class GroupPolicyConfig:
    name: str
    gate: str | None = None
    min_confidence: float | None = None
    max_risk_score: float | None = None
    mention_min_confidence: float | None = None
    mention_must_reply: bool | None = None
    per_group_cooldown_seconds: float | None = None
    per_group_max_per_hour: int | None = None
    self_aliases: tuple[str, ...] = ()
    role_prompt: str | None = None
    allow_dry_run_paste: bool = False


@dataclass(frozen=True)
class MonitorConfig:
    iphone_agent_cli: str = DEFAULT_IPHONE_AGENT_CLI
    iphone_agent_url: str = DEFAULT_IPHONE_AGENT_URL
    skill_name: str = "wechat-group-reply"
    ocr_backend: str = "rapidocr"
    mode: MonitorMode = MonitorMode.OBSERVE_ONLY
    allow_live_send: bool = False
    target_groups: tuple[GroupPolicyConfig, ...] = ()
    self_aliases: tuple[str, ...] = ()
    role_prompt: str = ""
    simulation_targets: tuple[str, ...] = ("文件传输助手",)
    rules_path: Path | None = None
    default_gate: str = "hybrid"
    min_confidence: float = 0.70
    max_risk_score: float = 0.30
    mention_min_confidence: float = 0.50
    mention_must_reply: bool = False
    context_window: int = 8
    poll_interval_min_seconds: float = 2.0
    poll_interval_max_seconds: float = 3.0
    pending_ttl_seconds: float = 180.0
    burst_merge_window_seconds: float = 5.0
    short_discontinuity_window_seconds: float = 90.0
    cli_timeout_seconds: float = 90.0
    paste_flow_timeout_seconds: float = 180.0
    require_mounted_skill: bool = True
    state_db_path: Path = Path(".local-data/state/wechat_monitor.db")
    audit_log_path: Path = Path(".local-data/state/monitor_audit.jsonl")
    evidence_root: Path = Path(".local-data/runs")
    rate_limits: RateLimitConfig = RateLimitConfig()
    llm: LlmConfig = LlmConfig()

    @property
    def active_group(self) -> GroupPolicyConfig | None:
        return self.target_groups[0] if self.target_groups else None

    def resolved_path(self, path: Path) -> Path:
        return path if path.is_absolute() else (repo_root() / path).resolve()


def _parse_group_entry(entry: Any) -> GroupPolicyConfig:
    if isinstance(entry, str):
        name = entry.strip()
        if not name:
            raise ValueError("Group name in target_groups cannot be empty")
        return GroupPolicyConfig(name=name)
    if not isinstance(entry, dict):
        raise ValueError(f"target_groups entry must be string or mapping, got {type(entry).__name__}")
    name = str(entry.get("name") or "").strip()
    if not name:
        raise ValueError("Group policy mapping missing 'name'")
    aliases = tuple(str(a).strip() for a in (entry.get("self_aliases") or []) if str(a).strip())
    return GroupPolicyConfig(
        name=name,
        gate=str(entry["gate"]).strip() if entry.get("gate") else None,
        min_confidence=float(entry["min_confidence"]) if entry.get("min_confidence") is not None else None,
        max_risk_score=float(entry["max_risk_score"]) if entry.get("max_risk_score") is not None else None,
        mention_min_confidence=(
            float(entry["mention_min_confidence"]) if entry.get("mention_min_confidence") is not None else None
        ),
        mention_must_reply=(
            bool(entry["mention_must_reply"]) if entry.get("mention_must_reply") is not None else None
        ),
        per_group_cooldown_seconds=(
            float(entry["per_group_cooldown_seconds"])
            if entry.get("per_group_cooldown_seconds") is not None
            else None
        ),
        per_group_max_per_hour=(
            int(entry["per_group_max_per_hour"]) if entry.get("per_group_max_per_hour") is not None else None
        ),
        self_aliases=aliases,
        role_prompt=(
            str(entry["role_prompt"]).strip() if entry.get("role_prompt") is not None else None
        ),
        allow_dry_run_paste=bool(entry.get("allow_dry_run_paste", False)),
    )


def load_monitor_config(
    config_path: str | Path | None = None,
    *,
    group_override: str | None = None,
    mode_override: str | None = None,
    allow_live_send_override: bool = False,
    gate_override: str | None = None,
    rules_override: str | Path | None = None,
    cli_override: str | None = None,
    url_override: str | None = None,
) -> MonitorConfig:
    data: dict[str, Any] = {}
    config_dir: Path | None = None
    if config_path is not None:
        p = Path(config_path)
        if not p.is_file():
            raise FileNotFoundError(f"Monitor config not found: {p}")
        loaded = yaml.safe_load(p.read_text(encoding="utf-8"))
        if loaded is not None and not isinstance(loaded, dict):
            raise ValueError(f"Monitor config {p} must be a YAML mapping")
        data = loaded or {}
        config_dir = p.parent.resolve()

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

    mode_str = (mode_override or str(data.get("mode") or MonitorMode.OBSERVE_ONLY.value)).strip().lower()
    mode = MonitorMode(mode_str)

    if group_override:
        groups = (GroupPolicyConfig(name=group_override.strip()),)
    else:
        groups = tuple(_parse_group_entry(item) for item in (data.get("target_groups") or []))

    self_aliases = tuple(str(a).strip() for a in (data.get("self_aliases") or []) if str(a).strip())
    sim_targets = tuple(
        str(t).strip()
        for t in (data.get("simulation_targets") or ["文件传输助手"])
        if str(t).strip()
    )

    raw_rules = rules_override or data.get("rules_path")
    rules_path: Path | None = None
    if raw_rules:
        rp = Path(str(raw_rules))
        if not rp.is_absolute():
            candidate = (config_dir / rp).resolve() if config_dir else (repo_root() / rp).resolve()
            rules_path = candidate if candidate.is_file() else (repo_root() / rp).resolve()
        else:
            rules_path = rp

    rl_data = data.get("rate_limits") or {}
    forbidden = tuple(
        str(pat)
        for pat in (data.get("forbidden_reply_patterns") or DEFAULT_FORBIDDEN_PATTERNS)
    )
    rate_limits = RateLimitConfig(
        global_min_interval_seconds=float(rl_data.get("global_min_interval_seconds", 15.0)),
        global_max_per_hour=int(rl_data.get("global_max_per_hour", 15)),
        per_group_cooldown_seconds=float(rl_data.get("per_group_cooldown_seconds", 60.0)),
        per_group_max_per_hour=int(rl_data.get("per_group_max_per_hour", 6)),
        human_takeover_cooldown_seconds=float(rl_data.get("human_takeover_cooldown_seconds", 600.0)),
        quiet_hours=str(data.get("quiet_hours") or rl_data.get("quiet_hours") or "23:00-07:00"),
        max_reply_chars=int(rl_data.get("max_reply_chars", 240)),
        forbidden_reply_patterns=forbidden,
    )

    llm_cfg = LlmConfig.from_dict_and_env(data.get("llm") or {})
    poll_min = max(0.5, float(data.get("poll_interval_min_seconds", 2.0)))
    poll_max = max(poll_min, float(data.get("poll_interval_max_seconds", 3.0)))
    pending_ttl = max(10.0, float(data.get("pending_ttl_seconds", 180.0)))
    burst_merge = max(0.0, float(data.get("burst_merge_window_seconds", 5.0)))
    short_disc = max(0.0, float(data.get("short_discontinuity_window_seconds", 90.0)))

    return MonitorConfig(
        iphone_agent_cli=cli_path,
        iphone_agent_url=url,
        skill_name=str(data.get("skill_name") or "wechat-group-reply"),
        ocr_backend=str(data.get("ocr_backend") or "rapidocr"),
        mode=mode,
        allow_live_send=bool(allow_live_send_override or data.get("allow_live_send", False)),
        target_groups=groups,
        self_aliases=self_aliases,
        role_prompt=str(data.get("role_prompt") or "").strip(),
        simulation_targets=sim_targets,
        rules_path=rules_path,
        default_gate=str(gate_override or data.get("default_gate") or "hybrid"),
        min_confidence=0.70 if data.get("min_confidence") is None else float(data["min_confidence"]),
        max_risk_score=0.30 if data.get("max_risk_score") is None else float(data["max_risk_score"]),
        mention_min_confidence=(
            0.50 if data.get("mention_min_confidence") is None else float(data["mention_min_confidence"])
        ),
        mention_must_reply=bool(data.get("mention_must_reply", False)),
        context_window=max(1, int(data.get("context_window", 8))),
        poll_interval_min_seconds=poll_min,
        poll_interval_max_seconds=poll_max,
        pending_ttl_seconds=pending_ttl,
        burst_merge_window_seconds=burst_merge,
        short_discontinuity_window_seconds=short_disc,
        cli_timeout_seconds=float(data.get("cli_timeout_seconds", 90.0)),
        paste_flow_timeout_seconds=float(data.get("paste_flow_timeout_seconds", 180.0)),
        require_mounted_skill=bool(data.get("require_mounted_skill", True)),
        state_db_path=Path(str(data.get("state_db_path") or ".local-data/state/wechat_monitor.db")),
        audit_log_path=Path(str(data.get("audit_log_path") or ".local-data/state/monitor_audit.jsonl")),
        evidence_root=Path(str(data.get("evidence_root") or ".local-data/runs")),
        rate_limits=rate_limits,
        llm=llm_cfg,
    )
