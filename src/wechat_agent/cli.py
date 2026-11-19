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
    mode.add_argument("--dry-run", dest="send", action="store_false", help="Paste & verify only, stop before Send (default)")
    mode.add_argument("--send", dest="send", action="store_true", help="Explicitly execute guarded Send after paste")
    send_cmd.set_defaults(send=False)
    send_cmd.add_argument("--config", type=Path, help="Optional YAML config file")
    send_cmd.add_argument("--allow-target", action="append", dest="allow_targets", help="Explicitly allow additional target")
    send_cmd.add_argument("--skill", help="Override skill name (default: wechat-send)")
    send_cmd.add_argument("--ocr", help="Override OCR backend (default: rapidocr)")
    send_cmd.add_argument("--cli", help="Override iphone-agent CLI path")
    send_cmd.add_argument("--url", help="Override base server URL")

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
    return 0 if (result["base_ready"] and result["staged_v1_ready"] and result["wechat_send_mounted"]) else 1


def _cmd_send(args: argparse.Namespace) -> int:
    cfg = load_config(
        args.config,
        allowed_targets_override=args.allow_targets,
        skill_name_override=args.skill,
        ocr_backend_override=args.ocr,
        cli_override=args.cli,
        url_override=args.url,
    )
    text = args.text if args.text is not None else Path(args.text_file).read_text(encoding="utf-8")
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
    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
