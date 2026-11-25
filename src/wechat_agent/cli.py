"""CLI entry point for `wechat-agent` (doctor / verify-skills / send)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from wechat_agent.adapters.driver import DriverError
from wechat_agent.adapters.iphone_cli import IphoneAgentCliDriver
from wechat_agent.config import WeChatSendConfig, load_config
from wechat_agent.guards.skill_guard import validate_skills_dir
from wechat_agent.monitor.config import load_monitor_config
from wechat_agent.monitor.decision import build_decision_gate, load_rules_yaml
from wechat_agent.monitor.generators import HybridReplyGenerator, LlmReplyGenerator, RuleTemplateGenerator
from wechat_agent.monitor.llm_client import OpenAICompatibleLlmClient
from wechat_agent.monitor.runner import MonitorRunner
from wechat_agent.monitor.store import MonitorInstanceLockError, MonitorStore
from wechat_agent.vision.ocr import create_ocr_engine
from wechat_agent.workflows.send_message import SendMessageWorkflow


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="wechat-agent",
        description="Independent WeChat automation CLI built on top of iphone-agent.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # verify-skills
    verify = sub.add_parser("verify-skills", help="Validate local skills/*/SKILL.md contract (offline)")
    verify.add_argument("--skills-dir", type=Path, help="Optional custom skills directory")

    # doctor
    doctor = sub.add_parser("doctor", help="Read-only health check for base server, OCR, and HIDAGENT_SKILLS")
    doctor.add_argument("--config", type=Path, help="Optional YAML config file")
    doctor.add_argument("--cli", help="Override iphone-agent CLI path")
    doctor.add_argument("--url", help="Override base server URL")

    # send
    send_cmd = sub.add_parser("send", help="Send a WeChat message (defaults to --dry-run paste-only)")
    send_cmd.add_argument("--target", required=True, help="Exact target conversation title (M1 default: 文件传输助手)")
    group = send_cmd.add_mutually_exclusive_group(required=True)
    group.add_argument("--text", help="Exact message text to paste/send")
    group.add_argument("--text-file", type=Path, help="UTF-8 text file containing message text")
    mode = send_cmd.add_mutually_exclusive_group()
    mode.add_argument(
        "--dry-run",
        dest="send",
        action="store_false",
        help="Paste & verify only, stop before Send (default)",
    )
    mode.add_argument("--send", dest="send", action="store_true", help="Explicitly execute guarded Send after paste")
    send_cmd.set_defaults(send=False)
    send_cmd.add_argument("--config", type=Path, help="Optional YAML config file")
    send_cmd.add_argument(
        "--allow-target",
        action="append",
        dest="allow_targets",
        help="Explicitly allow additional target",
    )
    send_cmd.add_argument("--skill", help="Override skill name (default: wechat-send)")
    send_cmd.add_argument("--ocr", help="Override OCR backend (default: rapidocr)")
    send_cmd.add_argument(
        "--send-method",
        choices=("enter", "click"),
        help="Override send method after paste verification (default: enter)",
    )
    send_cmd.add_argument("--cli", help="Override iphone-agent CLI path")
    send_cmd.add_argument("--url", help="Override base server URL")

    # monitor (M2)
    mon_cmd = sub.add_parser(
        "monitor",
        help="Run designated WeChat group monitoring (defaults to --mode observe-only)",
    )
    mon_cmd.add_argument("--config", type=Path, help="Path to group monitor YAML config")
    mon_cmd.add_argument("--group", help="Override target group title")
    mon_cmd.add_argument(
        "--mode",
        choices=("observe-only", "dry-run", "live"),
        help="Monitor mode (default: observe-only)",
    )
    mon_cmd.add_argument(
        "--allow-live-send",
        action="store_true",
        help="Required explicit safety lock alongside --mode live",
    )
    mon_cmd.add_argument(
        "--gate",
        choices=("observe_only", "rule", "llm", "hybrid", "jev"),
        help="Override Layer-1 DecisionGate type",
    )
    mon_cmd.add_argument("--rules", type=Path, help="Override YAML rules file path")
    mon_cmd.add_argument("--once", action="store_true", help="Run exactly 1 tick and exit")
    mon_cmd.add_argument("--max-ticks", type=int, help="Maximum polling ticks before stopping")
    mon_cmd.add_argument("--cli", help="Override iphone-agent CLI path")
    mon_cmd.add_argument("--url", help="Override base server URL")

    return parser


def _cmd_verify_skills(skills_dir: Path | None) -> int:
    target_dir = skills_dir or WeChatSendConfig().skills_dir()
    try:
        names = validate_skills_dir(target_dir)
    except (OSError, ValueError) as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False, indent=2), file=sys.stderr)
        return 2
    print(json.dumps({"ok": True, "skills_dir": str(target_dir), "validated": names}, ensure_ascii=False, indent=2))
    return 0


def _cmd_doctor(cfg: WeChatSendConfig) -> int:
    skills_dir = cfg.skills_dir()
    local_skills = validate_skills_dir(skills_dir)
    driver = IphoneAgentCliDriver(
        cfg.iphone_agent_cli,
        cfg.iphone_agent_url,
        timeout=cfg.cli_timeout_seconds,
        paste_flow_timeout=cfg.paste_flow_timeout_seconds,
    )

    result: dict = {
        "cli": cfg.iphone_agent_cli,
        "url": cfg.iphone_agent_url,
        "local_skills": local_skills,
        "ocr_backend": cfg.ocr_backend,
        "allowed_targets": list(cfg.allowed_targets),
        "mounted_skills": [],
        "wechat_send_mounted": False,
        "staged_v1_ready": False,
        "base_ready": False,
    }
    try:
        st = driver.status()
        caps = driver.capabilities()
        mounted = driver.skill_list()
        clip = (caps.get("features") or {}).get("clipboard") or {}
        result["status"] = {
            "device": st.get("device"),
            "screen": st.get("screen"),
            "calibrated": st.get("calibrated"),
            "ready": st.get("ready"),
            "guard": st.get("guard"),
        }
        result["base_ready"] = bool(
            st.get("device")
            and st.get("screen")
            and st.get("calibrated")
            and st.get("ready")
            and not (st.get("guard") or {}).get("tripped")
        )
        result["staged_v1_ready"] = bool(clip.get("staged_available") and clip.get("protocol") == "staged-v1")
        result["mounted_skills"] = mounted
        result["wechat_send_mounted"] = cfg.skill_name in mounted
        if not result["wechat_send_mounted"]:
            result["mount_hint"] = (
                f"export HIDAGENT_SKILLS={skills_dir} and restart base worker"
            )
    except DriverError as exc:
        result["error"] = str(exc)
        print(json.dumps(result, ensure_ascii=False, indent=2), file=sys.stderr)
        return 2

    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if (result["base_ready"] and result["staged_v1_ready"] and result["wechat_send_mounted"]) else 2


def _cmd_send(args: argparse.Namespace) -> int:
    try:
        cfg = load_config(
            args.config,
            allowed_targets_override=args.allow_targets,
            skill_name_override=args.skill,
            ocr_backend_override=args.ocr,
            send_method_override=args.send_method,
            cli_override=args.cli,
            url_override=args.url,
        )
        text = args.text if args.text is not None else Path(args.text_file).read_text(encoding="utf-8")
    except (OSError, ValueError) as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False, indent=2), file=sys.stderr)
        return 2
    driver = IphoneAgentCliDriver(
        cfg.iphone_agent_cli,
        cfg.iphone_agent_url,
        timeout=cfg.cli_timeout_seconds,
        paste_flow_timeout=cfg.paste_flow_timeout_seconds,
    )
    ocr = create_ocr_engine(cfg.ocr_backend)
    workflow = SendMessageWorkflow(driver, ocr, cfg)
    report = workflow.run(args.target, text, send=bool(args.send))
    print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
    return 0 if report.succeeded else 2


def _cmd_monitor(args: argparse.Namespace) -> int:
    if args.max_ticks is not None and args.max_ticks <= 0:
        print(
            json.dumps(
                {"ok": False, "error": f"--max-ticks must be a positive integer, got {args.max_ticks}"},
                ensure_ascii=False,
                indent=2,
            ),
            file=sys.stderr,
        )
        return 2
    try:
        cfg = load_monitor_config(
            args.config,
            group_override=args.group,
            mode_override=args.mode,
            allow_live_send_override=bool(args.allow_live_send),
            gate_override=args.gate,
            rules_override=args.rules,
            cli_override=args.cli,
            url_override=args.url,
        )
    except (OSError, ValueError) as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False, indent=2), file=sys.stderr)
        return 2
    rules = load_rules_yaml(cfg.rules_path) if cfg.rules_path else []
    llm_client = OpenAICompatibleLlmClient(cfg.llm) if cfg.llm.is_ready else None

    active_group = cfg.active_group
    gate_type = (active_group.gate if active_group and active_group.gate else cfg.default_gate)
    mention_must_reply = (
        active_group.mention_must_reply
        if active_group and active_group.mention_must_reply is not None
        else cfg.mention_must_reply
    )
    gate = build_decision_gate(
        gate_type,
        rules=rules,
        llm_client=llm_client,
        mention_must_reply=mention_must_reply,
    )
    skill_md_path = WeChatSendConfig().skills_dir() / cfg.skill_name / "SKILL.md"
    skill_instructions = skill_md_path.read_text(encoding="utf-8") if skill_md_path.is_file() else ""
    generator = HybridReplyGenerator(
        RuleTemplateGenerator(rules),
        LlmReplyGenerator(llm_client, skill_instructions=skill_instructions) if llm_client else None,
    )

    driver = IphoneAgentCliDriver(
        cfg.iphone_agent_cli,
        cfg.iphone_agent_url,
        timeout=cfg.cli_timeout_seconds,
        paste_flow_timeout=cfg.paste_flow_timeout_seconds,
    )
    ocr = create_ocr_engine(cfg.ocr_backend)
    max_ticks = 1 if args.once else args.max_ticks

    try:
        with MonitorStore(
            cfg.resolved_path(cfg.state_db_path),
            cfg.resolved_path(cfg.audit_log_path),
        ) as store:
            runner = MonitorRunner(driver, ocr, cfg, store, gate, generator)
            reports = runner.run_loop(max_ticks=max_ticks)
    except MonitorInstanceLockError as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False, indent=2), file=sys.stderr)
        return 2

    payload = [r.to_dict() for r in reports]
    print(json.dumps(payload[0] if (args.once and len(payload) == 1) else payload, ensure_ascii=False, indent=2))
    if reports and reports[-1].outcome.value.startswith("halted_"):
        return 2
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "verify-skills":
        return _cmd_verify_skills(args.skills_dir)
    if args.command == "doctor":
        cfg = load_config(args.config, cli_override=args.cli, url_override=args.url)
        return _cmd_doctor(cfg)
    if args.command == "send":
        return _cmd_send(args)
    if args.command == "monitor":
        return _cmd_monitor(args)
    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
