"""Thin subprocess adapter over the base `iphone-agent` CLI (zero base code imports)."""

from __future__ import annotations

import base64
import binascii
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any

from wechat_agent.models import (
    Box,
    ClickResult,
    KeyActionResult,
    MeasuredPointer,
    Observation,
    PasteFlowResult,
    PasteRecoveryState,
    PointResult,
    SkillCheckResult,
    SkillConfirmResult,
    SkillPrepareResult,
)

from .driver import ConflictError, DriverError

_HTTP_STATUS_RE = re.compile(r"\bHTTP\s+(\d{3})\b")


def _extract_image_bytes(payload: dict[str, Any], image_output: Path | None) -> tuple[bytes, str | None]:
    image_path = payload.get("image_path")
    try:
        if image_path:
            path = Path(str(image_path))
            if path.is_file():
                os.chmod(path, 0o600)
                return path.read_bytes(), str(path)
        if image_output is not None and image_output.is_file():
            os.chmod(image_output, 0o600)
            return image_output.read_bytes(), str(image_output)
        b64 = payload.get("image_base64")
        if isinstance(b64, str) and b64:
            return base64.b64decode(b64), None
    except (OSError, ValueError, binascii.Error) as exc:
        raise DriverError(f"Failed to read/decode screenshot payload from iphone-agent: {exc}") from exc
    return b"", str(image_path) if image_path else None


def _parse_json_blob(text: str) -> dict[str, Any] | None:
    stripped = (text or "").strip()
    if not stripped:
        return None
    try:
        parsed = json.loads(stripped)
        return parsed if isinstance(parsed, dict) else {"value": parsed}
    except json.JSONDecodeError:
        for line in reversed(stripped.splitlines()):
            line = line.strip()
            if line.startswith("{") and line.endswith("}"):
                try:
                    parsed = json.loads(line)
                    if isinstance(parsed, dict):
                        return parsed
                except json.JSONDecodeError:
                    continue
    return None


class IphoneAgentCliDriver:
    """Calls `/path/to/iphone-agent` CLI via subprocess and parses strict JSON contracts."""

    def __init__(
        self,
        cli_path: str,
        base_url: str = "http://127.0.0.1:8890",
        *,
        timeout: float = 90.0,
        paste_flow_timeout: float = 180.0,
    ) -> None:
        self.cli_path = cli_path
        self.base_url = base_url
        self.timeout = timeout
        self.paste_flow_timeout = max(float(timeout), float(paste_flow_timeout))

    def _run_raw(self, args: list[str], *, timeout: float | None = None) -> subprocess.CompletedProcess[str]:
        cmd = [self.cli_path, "--url", self.base_url, *args]
        effective_timeout = self.timeout if timeout is None else timeout
        try:
            return subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding="utf-8",
                timeout=effective_timeout,
                check=False,
            )
        except FileNotFoundError as exc:
            raise DriverError(f"iphone-agent CLI not found: {self.cli_path}", command=cmd) from exc
        except subprocess.TimeoutExpired as exc:
            raise DriverError(
                f"iphone-agent CLI timed out after {effective_timeout}s: {' '.join(args)}",
                command=cmd,
            ) from exc
        except (OSError, UnicodeDecodeError, ValueError) as exc:
            raise DriverError(
                f"iphone-agent CLI execution/decoding error: {exc}",
                command=cmd,
            ) from exc

    def _run_json(self, args: list[str]) -> dict[str, Any]:
        proc = self._run_raw(args)
        if proc.returncode == 0:
            parsed = _parse_json_blob(proc.stdout)
            if parsed is None:
                raise DriverError(
                    f"iphone-agent returned non-JSON stdout: {proc.stdout[:200]}",
                    command=[self.cli_path, *args],
                )
            return parsed

        err_payload = _parse_json_blob(proc.stderr) or _parse_json_blob(proc.stdout) or {}
        detail = str(
            err_payload.get("error")
            or err_payload.get("detail")
            or proc.stderr.strip()
            or proc.stdout.strip()
            or f"CLI exited with code {proc.returncode}"
        )
        # Prefer structured status_code / http_status field from JSON payload over text regex matching.
        status_code: int | None = None
        for key in ("status_code", "http_status", "status"):
            raw_code = err_payload.get(key)
            if isinstance(raw_code, int) and not isinstance(raw_code, bool) and 100 <= raw_code <= 599:
                status_code = raw_code
                break
        if status_code is None:
            match = _HTTP_STATUS_RE.search(detail)
            status_code = int(match.group(1)) if match else None
        if status_code == 409:
            raise ConflictError(
                detail,
                status_code=409,
                detail=detail,
                raw=err_payload,
                command=[self.cli_path, *args],
            )
        raise DriverError(
            detail,
            status_code=status_code,
            detail=detail,
            raw=err_payload,
            command=[self.cli_path, *args],
        )

    def status(self) -> dict[str, Any]:
        return self._run_json(["status"])

    def capabilities(self) -> dict[str, Any]:
        return self._run_json(["capabilities"])

    def skill_list(self) -> list[str]:
        payload = self._run_json(["skill-list"])
        return [str(s) for s in (payload.get("skills") or [])]

    def skill_show(self, name: str) -> dict[str, Any]:
        return self._run_json(["skill-show", name])

    def skill_check(self, observed_text: str, skill: str = "") -> SkillCheckResult:
        args = ["skill-check", "--observed-text", observed_text]
        if skill:
            args.extend(["--skill", skill])
        payload = self._run_json(args)
        return SkillCheckResult(
            stop=bool(payload.get("stop", False)),
            counted=bool(payload.get("counted", False)),
            allowed=bool(payload.get("allowed", False)),
            reason=str(payload.get("reason") or ""),
            raw=payload,
        )

    def skill_prepare_send(self, skill: str = "") -> SkillPrepareResult:
        args = ["skill-prepare-send"]
        if skill:
            args.extend(["--skill", skill])
        payload = self._run_json(args)
        return SkillPrepareResult(
            stop=bool(payload.get("stop", False)),
            counted=bool(payload.get("counted", False)),
            allowed=bool(payload.get("allowed", False)),
            reason=str(payload.get("reason") or ""),
            reservation=str(payload.get("reservation") or ""),
            raw=payload,
        )

    def skill_confirm_send(
        self,
        bubble_visible: bool,
        observed_text: str = "",
        skill: str = "",
    ) -> SkillConfirmResult:
        args = [
            "skill-confirm-send",
            "--bubble-visible" if bubble_visible else "--no-bubble-visible",
            "--observed-text",
            observed_text,
        ]
        if skill:
            args.extend(["--skill", skill])
        payload = self._run_json(args)
        return SkillConfirmResult(
            stop=bool(payload.get("stop", False)),
            counted=bool(payload.get("counted", False)),
            allowed=bool(payload.get("allowed", False)),
            reason=str(payload.get("reason") or ""),
            raw=payload,
        )

    def observe(
        self,
        *,
        for_click: bool = False,
        image_output: Path | None = None,
    ) -> Observation:
        args = ["observe"]
        if for_click:
            args.append("--for-click")
        if image_output is not None:
            image_output.parent.mkdir(parents=True, exist_ok=True)
            args.append(str(image_output))
        payload = self._run_json(args)
        img_bytes, img_path = _extract_image_bytes(payload, image_output)
        size = payload.get("size") or {}
        width = int(payload.get("width") or size.get("width") or 0)
        height = int(payload.get("height") or size.get("height") or 0)
        return Observation(
            screen_id=str(payload.get("screen_id") or ""),
            image_bytes=img_bytes,
            image_path=img_path,
            width=width,
            height=height,
            fresh=bool(payload.get("fresh", True)),
            raw=payload,
        )

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
    ) -> PointResult:
        args = ["point", f"{x:.6f}", f"{y:.6f}", "--screen-id", screen_id]
        if target_box is not None:
            args.extend(
                [
                    "--target-box",
                    f"{target_box.left:.6f}",
                    f"{target_box.top:.6f}",
                    f"{target_box.right:.6f}",
                    f"{target_box.bottom:.6f}",
                ]
            )
        if rehome:
            args.append("--rehome")
        if caret_scope is not None:
            args.extend(
                [
                    "--caret-scope",
                    f"{caret_scope.left:.6f}",
                    f"{caret_scope.top:.6f}",
                    f"{caret_scope.right:.6f}",
                    f"{caret_scope.bottom:.6f}",
                ]
            )
        if image_output is not None:
            image_output.parent.mkdir(parents=True, exist_ok=True)
            args.extend(["--image-output", str(image_output)])
        payload = self._run_json(args)
        img_bytes, img_path = _extract_image_bytes(payload, image_output)
        mp = MeasuredPointer.from_point_response(payload)
        return PointResult(
            screen_id=payload.get("screen_id"),
            measured_pointer=mp,
            pointer_seen=bool(payload.get("pointer_seen", mp is not None)),
            image_bytes=img_bytes,
            image_path=img_path,
            raw=payload,
        )

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
    ) -> ClickResult:
        args = [
            "click-observed",
            f"{pointer_x:.6f}",
            f"{pointer_y:.6f}",
            "--screen-id",
            screen_id,
            "--target-box",
            f"{target_box.left:.6f}",
            f"{target_box.top:.6f}",
            f"{target_box.right:.6f}",
            f"{target_box.bottom:.6f}",
        ]
        if target_label:
            args.extend(["--target-label", target_label])
        if duration is not None:
            args.extend(["--duration", f"{duration:.2f}"])
        if caret_scope is not None:
            args.extend(
                [
                    "--caret-scope",
                    f"{caret_scope.left:.6f}",
                    f"{caret_scope.top:.6f}",
                    f"{caret_scope.right:.6f}",
                    f"{caret_scope.bottom:.6f}",
                ]
            )
        if image_output is not None:
            image_output.parent.mkdir(parents=True, exist_ok=True)
            args.extend(["--image-output", str(image_output)])
        payload = self._run_json(args)
        img_bytes, img_path = _extract_image_bytes(payload, image_output)
        return ClickResult(
            screen_id=payload.get("screen_id"),
            clicked=bool(payload.get("clicked", False)),
            command_acknowledged=bool(payload.get("command_acknowledged", False)),
            effect_verified=bool(payload.get("effect_verified", False)),
            changed=payload.get("changed"),
            image_bytes=img_bytes,
            image_path=img_path,
            raw=payload,
        )

    def send_enter_key(
        self,
        *,
        screen_id: str,
        caret_scope: Box,
        image_output: Path | None = None,
    ) -> KeyActionResult:
        """Dispatch a protected keyboard Enter action without `enter_intent` (consumes send reservation)."""
        if not isinstance(screen_id, str) or not screen_id.strip():
            raise ValueError("send_enter_key requires a non-empty screen_id")
        # Intentionally omit `enter_intent` so base guard.py enforces prepare-send reservation & circuit breaker.
        action_payload: dict[str, Any] = {
            "action": "key",
            "text": "enter",
            "screen_id": screen_id,
            "caret_scope": [
                round(caret_scope.left, 6),
                round(caret_scope.top, 6),
                round(caret_scope.right, 6),
                round(caret_scope.bottom, 6),
            ],
        }
        args = ["action", json.dumps(action_payload, ensure_ascii=False, separators=(",", ":"))]
        if image_output is not None:
            image_output.parent.mkdir(parents=True, exist_ok=True)
            args.extend(["--image-output", str(image_output)])
        payload = self._run_json(args)
        img_bytes, img_path = _extract_image_bytes(payload, image_output)
        raw_may = payload.get("action_may_have_executed")
        may_exec = raw_may if isinstance(raw_may, bool) else None
        return KeyActionResult(
            screen_id=payload.get("screen_id"),
            command_acknowledged=bool(payload.get("command_acknowledged", False)),
            action_may_have_executed=may_exec,
            effect_verified=bool(payload.get("effect_verified", False)),
            changed=payload.get("changed"),
            error=str(payload.get("error") or ""),
            image_bytes=img_bytes,
            image_path=img_path,
            raw=payload,
        )

    def find_text(
        self,
        text: str,
        *,
        screen_id: str,
        nth: int = 1,
        exact: bool = False,
        region: Box | None = None,
    ) -> dict[str, Any]:
        args = ["find-text", text, "--screen-id", screen_id, "--nth", str(nth)]
        if exact:
            args.append("--exact")
        if region is not None:
            args.extend(
                [
                    "--region",
                    f"{region.left:.6f}",
                    f"{region.top:.6f}",
                    f"{region.right:.6f}",
                    f"{region.bottom:.6f}",
                ]
            )
        return self._run_json(args)

    def search_app(
        self,
        text: str,
        *,
        screen_id: str,
        image_output: Path | None = None,
    ) -> Observation:
        if not text.isascii():
            raise ValueError(f"search-app query must be ASCII, got {text!r}")
        args = ["search-app", text, "--screen-id", screen_id]
        if image_output is not None:
            image_output.parent.mkdir(parents=True, exist_ok=True)
            args.extend(["--image-output", str(image_output)])
        payload = self._run_json(args)
        img_bytes, img_path = _extract_image_bytes(payload, image_output)
        return Observation(
            screen_id=str(payload.get("screen_id") or ""),
            image_bytes=img_bytes,
            image_path=img_path,
            fresh=bool(payload.get("fresh", True)),
            raw=payload,
        )

    def scroll(
        self,
        x: float,
        y: float,
        lines: int,
        *,
        screen_id: str,
        image_output: Path | None = None,
    ) -> Observation:
        args = ["scroll", f"{x:.6f}", f"{y:.6f}", str(int(lines)), "--screen-id", screen_id]
        if image_output is not None:
            image_output.parent.mkdir(parents=True, exist_ok=True)
            args.extend(["--image-output", str(image_output)])
        payload = self._run_json(args)
        img_bytes, img_path = _extract_image_bytes(payload, image_output)
        return Observation(
            screen_id=str(payload.get("screen_id") or ""),
            image_bytes=img_bytes,
            image_path=img_path,
            fresh=bool(payload.get("fresh", True)),
            raw=payload,
        )

    def paste_flow(
        self,
        payload_path: Path,
        *,
        target_box: Box,
        expect_title: str,
        caret_box: Box | None = None,
        evidence_dir: Path | None = None,
        image_output: Path | None = None,
    ) -> PasteFlowResult:
        args = [
            "paste-flow",
            f"@{payload_path}",
            "--target-box",
            f"{target_box.left:.6f}",
            f"{target_box.top:.6f}",
            f"{target_box.right:.6f}",
            f"{target_box.bottom:.6f}",
            "--expect-title",
            expect_title,
        ]
        if caret_box is not None:
            args.extend(
                [
                    "--caret-box",
                    f"{caret_box.left:.6f}",
                    f"{caret_box.top:.6f}",
                    f"{caret_box.right:.6f}",
                    f"{caret_box.bottom:.6f}",
                ]
            )
        if evidence_dir is not None:
            evidence_dir.mkdir(parents=True, exist_ok=True)
            os.chmod(evidence_dir, 0o700)
            args.extend(["--evidence-dir", str(evidence_dir)])
        if image_output is not None:
            image_output.parent.mkdir(parents=True, exist_ok=True)
            args.extend(["--image-output", str(image_output)])

        try:
            proc = self._run_raw(args, timeout=self.paste_flow_timeout)
        except DriverError as exc:
            return PasteFlowResult(
                ok=False,
                cmd_v_sent=0,
                phase="error",
                effect_verified=False,
                recovery_state=PasteRecoveryState.UNKNOWN,
                next_action=(
                    "paste-flow subprocess timed out or errored; inspect the input field and base clipboard "
                    "state before resolving or restarting worker. Never blindly rerun paste-flow."
                ),
                error=str(exc),
                image_path=str(image_output) if image_output and image_output.is_file() else None,
                raw={"error": str(exc), "recovery_state": PasteRecoveryState.UNKNOWN.value},
            )

        ok = proc.returncode == 0
        raw = _parse_json_blob(proc.stdout if ok else proc.stderr)
        if raw is None:
            raw = _parse_json_blob(proc.stderr if ok else proc.stdout) or {}
        if image_output is not None and image_output.is_file():
            os.chmod(image_output, 0o600)

        return PasteFlowResult(
            ok=ok,
            cmd_v_sent=int(raw.get("cmd_v_sent") or 0),
            phase=str(raw.get("phase") or ""),
            effect_verified=bool(raw.get("effect_verified", False)),
            recovery_state=PasteRecoveryState.from_raw(raw.get("recovery_state")),
            next_action=str(raw.get("next_action") or ""),
            copy_trigger=str(raw.get("copy_trigger") or ""),
            permission_sheet_seen=bool(raw.get("permission_sheet_seen", False)),
            error=str(raw.get("error") or raw.get("stopped") or ("" if ok else proc.stderr.strip())),
            image_path=str(image_output) if image_output and image_output.is_file() else None,
            confirmation_evidence=str(raw.get("confirmation_evidence") or ""),
            client_text_match=str(raw.get("client_text_match") or ""),
            raw=raw,
        )
