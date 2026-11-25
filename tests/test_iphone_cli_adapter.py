"""Unit tests for IphoneAgentCliDriver subprocess parsing and RunEvidence privacy/redaction."""

from __future__ import annotations

import json
import stat
import subprocess
from pathlib import Path

import pytest

from wechat_agent.adapters.driver import ConflictError
from wechat_agent.adapters.iphone_cli import IphoneAgentCliDriver
from wechat_agent.evidence import RunEvidence, redact_dict
from wechat_agent.models import Box, PasteRecoveryState


def test_cli_driver_raises_conflict_on_409(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(cmd, **kwargs):
        return subprocess.CompletedProcess(
            cmd,
            2,
            stdout="",
            stderr=json.dumps({"error": "HTTP 409: 等待点击期间原画面已变化"}, ensure_ascii=False),
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    driver = IphoneAgentCliDriver("/fake/iphone-agent", "http://127.0.0.1:8890")
    with pytest.raises(ConflictError) as exc_info:
        driver.point(0.5, 0.5, screen_id="sid-1", target_box=Box(0.4, 0.4, 0.6, 0.6))
    assert exc_info.value.status_code == 409


def test_cli_driver_paste_flow_parses_stderr_recovery_receipt(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    payload = tmp_path / "payload.txt"
    payload.write_text("测试内容", encoding="utf-8")

    err_receipt = {
        "error": "Paste stage stopped",
        "cmd_v_sent": 0,
        "phase": "copied",
        "effect_verified": False,
        "recovery_state": "copied_not_pasted",
        "next_action": "The Shortcut copied the text but Cmd+V was NOT sent; resolve first.",
        "copy_trigger": "assistive_touch_hold",
    }

    def fake_run(cmd, **kwargs):
        return subprocess.CompletedProcess(cmd, 2, stdout="", stderr=json.dumps(err_receipt, ensure_ascii=False))

    monkeypatch.setattr(subprocess, "run", fake_run)
    driver = IphoneAgentCliDriver("/fake/iphone-agent")
    res = driver.paste_flow(payload, target_box=Box(0.14, 0.51, 0.78, 0.57), expect_title="文件传输助手")
    assert res.ok is False
    assert res.is_strictly_completed is False
    assert res.recovery_state == PasteRecoveryState.COPIED_NOT_PASTED
    assert res.cmd_v_sent == 0
    assert "Cmd+V was NOT sent" in res.next_action


def test_evidence_permissions_and_redaction(tmp_path: Path) -> None:
    ev = RunEvidence(tmp_path / "runs", run_id="test-run")
    assert (ev.dir.stat().st_mode & 0o777) == 0o700
    assert (ev.paste_evidence_dir.stat().st_mode & 0o777) == 0o700

    payload_file = ev.write_payload("你好")
    assert (payload_file.stat().st_mode & 0o777) == 0o600
    assert stat.S_IMODE(payload_file.stat().st_mode) == 0o600

    redacted = redact_dict(
        {
            "transaction_id": "tx-999",
            "reservation": "res-abc",
            "image_base64": "AAAA",
            "note": "手机号 13812345678",
        }
    )
    assert redacted["transaction_id"] == "[REDACTED]"
    assert redacted["reservation"] == "[REDACTED]"
    assert redacted["image_base64"].startswith("<redacted_base64")
    assert "13812345678" not in redacted["note"]


def test_cli_driver_forwards_caret_scope_on_point_and_click_observed(monkeypatch: pytest.MonkeyPatch) -> None:
    recorded_cmds: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        recorded_cmds.append(list(cmd))
        if "point" in cmd:
            out = {"screen_id": "pt-1", "measured_pointer": {"x": 0.86, "y": 0.89}, "pointer_seen": True}
        else:
            out = {"screen_id": "clk-1", "clicked": True, "command_acknowledged": True, "effect_verified": True}
        return subprocess.CompletedProcess(cmd, 0, stdout=json.dumps(out), stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    driver = IphoneAgentCliDriver("/fake/iphone-agent", "http://127.0.0.1:8890")
    scope = Box(0.132, 0.528, 0.764, 0.576)
    target = Box(0.75, 0.85, 0.985, 0.935)

    driver.point(0.86, 0.89, screen_id="sid-1", target_box=target, caret_scope=scope)
    driver.click_observed(0.86, 0.89, screen_id="sid-2", target_box=target, target_label="发送", caret_scope=scope)
    driver.point(0.5, 0.5, screen_id="sid-3", target_box=target)

    pt_cmd, clk_cmd, no_scope_cmd = recorded_cmds
    for cmd in (pt_cmd, clk_cmd):
        idx = cmd.index("--caret-scope")
        assert cmd[idx + 1 : idx + 5] == ["0.132000", "0.528000", "0.764000", "0.576000"]
    assert "--caret-scope" not in no_scope_cmd


def test_cli_driver_paste_flow_uses_dedicated_timeout_and_handles_timeout_gracefully(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    payload = tmp_path / "payload.txt"
    payload.write_text("超时恢复测试", encoding="utf-8")

    observed_timeouts: list[float] = []

    def fake_run_ok(cmd, **kwargs):
        observed_timeouts.append(float(kwargs["timeout"]))
        if "status" in cmd:
            return subprocess.CompletedProcess(cmd, 0, stdout=json.dumps({"ready": True}), stderr="")
        return subprocess.CompletedProcess(
            cmd,
            0,
            stdout=json.dumps(
                {
                    "cmd_v_sent": 1,
                    "phase": "completed",
                    "effect_verified": True,
                    "recovery_state": "completed",
                }
            ),
            stderr="",
        )

    monkeypatch.setattr(subprocess, "run", fake_run_ok)
    driver = IphoneAgentCliDriver("/fake/iphone-agent", timeout=90.0, paste_flow_timeout=180.0)
    driver.status()
    res_ok = driver.paste_flow(payload, target_box=Box(0.13, 0.52, 0.76, 0.58), expect_title="文件传输助手")
    assert res_ok.is_strictly_completed is True
    assert observed_timeouts == [90.0, 180.0]

    def fake_run_timeout(cmd, **kwargs):
        raise subprocess.TimeoutExpired(cmd, kwargs["timeout"])

    monkeypatch.setattr(subprocess, "run", fake_run_timeout)
    res_timeout = driver.paste_flow(payload, target_box=Box(0.13, 0.52, 0.76, 0.58), expect_title="文件传输助手")
    assert res_timeout.ok is False
    assert res_timeout.is_strictly_completed is False
    assert res_timeout.recovery_state == PasteRecoveryState.UNKNOWN
    assert "180.0s" in res_timeout.error
    assert "Never blindly rerun paste-flow" in res_timeout.next_action


def test_cli_driver_send_enter_key_payload_image_output_and_fields(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    recorded_cmds: list[list[str]] = []
    out_img = tmp_path / "evidence" / "send_enter.jpg"

    def fake_run_ok(cmd, **kwargs):
        recorded_cmds.append(list(cmd))
        out_idx = cmd.index("--image-output") + 1
        Path(cmd[out_idx]).write_bytes(b"fake-jpeg-bytes")
        return subprocess.CompletedProcess(
            cmd,
            0,
            stdout=json.dumps(
                {
                    "screen_id": "scr-after-enter",
                    "command_acknowledged": True,
                    "action_may_have_executed": True,
                    "effect_verified": False,
                    "changed": True,
                }
            ),
            stderr="",
        )

    monkeypatch.setattr(subprocess, "run", fake_run_ok)
    driver = IphoneAgentCliDriver("/fake/iphone-agent", "http://127.0.0.1:8890")
    scope = Box(0.132, 0.528, 0.764, 0.576)
    res = driver.send_enter_key(screen_id="scr-fresh-1", caret_scope=scope, image_output=out_img)

    cmd = recorded_cmds[0]
    action_idx = cmd.index("action")
    payload = json.loads(cmd[action_idx + 1])
    assert payload == {
        "action": "key",
        "text": "enter",
        "screen_id": "scr-fresh-1",
        "caret_scope": [0.132, 0.528, 0.764, 0.576],
    }
    assert "enter_intent" not in payload
    assert "--image-output" in cmd
    assert res.screen_id == "scr-after-enter"
    assert res.command_acknowledged is True
    assert res.action_may_have_executed is True
    assert res.effect_verified is False
    assert res.changed is True
    assert res.image_bytes == b"fake-jpeg-bytes"

    # Parse action_may_have_executed=False and None strictly
    def fake_run_false_and_missing(cmd, **kwargs):
        return subprocess.CompletedProcess(
            cmd,
            0,
            stdout=json.dumps(
                {
                    "screen_id": "scr-2",
                    "command_acknowledged": False,
                    "action_may_have_executed": False,
                    "error": "busy",
                }
            ),
            stderr="",
        )

    monkeypatch.setattr(subprocess, "run", fake_run_false_and_missing)
    res_false = driver.send_enter_key(screen_id="scr-fresh-2", caret_scope=scope)
    assert res_false.command_acknowledged is False
    assert res_false.action_may_have_executed is False
    assert res_false.error == "busy"

    # Raise ConflictError on HTTP 409 preserving raw payload
    def fake_run_409(cmd, **kwargs):
        return subprocess.CompletedProcess(
            cmd,
            2,
            stdout="",
            stderr=json.dumps(
                {
                    "status_code": 409,
                    "error": "HTTP 409: screen changed since the original observation; observe again before acting",
                    "action_may_have_executed": False,
                },
                ensure_ascii=False,
            ),
        )

    monkeypatch.setattr(subprocess, "run", fake_run_409)
    with pytest.raises(ConflictError) as exc_info:
        driver.send_enter_key(screen_id="scr-fresh-3", caret_scope=scope)
    assert exc_info.value.status_code == 409
    assert exc_info.value.raw.get("action_may_have_executed") is False

    with pytest.raises(ValueError, match="screen_id"):
        driver.send_enter_key(screen_id="", caret_scope=scope)


