"""Offline unit tests for Batch W4: error isolation, resident self-healing, LLM resilience, storage governance, CLI."""

from __future__ import annotations

import io
import json
import math
import os
import sqlite3
import subprocess
import urllib.error
import urllib.request
from email.message import Message
from pathlib import Path

import pytest

from tests.fakes import FakePhoneDriver, MockOcrEngine, make_collapsed_chat_image
from wechat_agent.adapters.driver import ConflictError, DriverError
from wechat_agent.adapters.iphone_cli import IphoneAgentCliDriver
from wechat_agent.cli import main as cli_main
from wechat_agent.evidence import RunEvidence, prune_evidence_runs
from wechat_agent.models import Box, Observation, OcrBox
from wechat_agent.monitor.config import GroupPolicyConfig, MonitorConfig
from wechat_agent.monitor.decision import LlmIntentGate, RuleDecisionGate
from wechat_agent.monitor.extractor import _decode_rgb as extractor_decode_rgb
from wechat_agent.monitor.llm_client import (
    MAX_LLM_TIMEOUT_SECONDS,
    LlmClientError,
    LlmConfig,
    OpenAICompatibleLlmClient,
)
from wechat_agent.monitor.models import ExtractedMessage, MessageRole, MonitorMode, TickOutcome
from wechat_agent.monitor.runner import MonitorRunner
from wechat_agent.monitor.store import MonitorStore
from wechat_agent.vision.ocr import OcrError, RapidOcrEngine
from wechat_agent.vision.wechat_ui import _decode_rgb as ui_decode_rgb
from wechat_agent.vision.wechat_ui import _image_aspect


def _monitor_cfg(tmp_path: Path, **overrides) -> MonitorConfig:
    defaults = dict(
        mode=MonitorMode.OBSERVE_ONLY,
        target_groups=(GroupPolicyConfig(name="示例测试群", self_aliases=("示例助手",)),),
        state_db_path=tmp_path / "state" / "monitor.db",
        audit_log_path=tmp_path / "state" / "audit.jsonl",
        evidence_root=tmp_path / "runs",
        poll_interval_min_seconds=0.1,
        poll_interval_max_seconds=0.1,
    )
    defaults.update(overrides)
    return MonitorConfig(**defaults)


def _chat_frame(
    ocr: MockOcrEngine,
    *,
    incoming: list[tuple[str, str, Box]] | None = None,
    suffix: bytes = b"",
) -> bytes:
    img = make_collapsed_chat_image(incoming_boxes=[b for _, _, b in (incoming or [])]) + suffix
    boxes = [OcrBox("示例测试群(12)", Box(0.28, 0.06, 0.72, 0.09))]
    for sender, text, b in incoming or []:
        if sender:
            boxes.append(OcrBox(sender, Box(b.left, b.top - 0.030, min(b.left + 0.14, b.right), b.top - 0.008)))
        boxes.append(OcrBox(text, Box(b.left + 0.02, b.top + 0.012, b.right - 0.02, b.bottom - 0.012)))
    ocr.register(img, boxes)
    return img


def test_error_isolation_ocr_corrupt_image_and_cli_structured_409(monkeypatch: pytest.MonkeyPatch) -> None:
    # 1. Corrupt bytes raise structured OcrError (never raw PIL UnidentifiedImageError)
    with pytest.raises(OcrError, match="Failed to decode image bytes"):
        RapidOcrEngine().read_boxes(b"corrupt-not-png-bytes")

    # 2. Helper decoders degrade safely to None / default aspect ratio
    assert ui_decode_rgb(b"corrupt") is None
    assert extractor_decode_rgb(b"corrupt") is None
    assert _image_aspect(b"corrupt", default=0.5) == 0.5

    # 3. IphoneAgentCliDriver prefers structured JSON status_code over text regex & isolates OSError
    driver = IphoneAgentCliDriver("/fake/iphone-agent")

    def fake_run_409(*args, **kwargs):
        return subprocess.CompletedProcess(
            args=args[0],
            returncode=1,
            stdout="",
            stderr=json.dumps({"status_code": 409, "error": "screen_id stale without http text"}),
        )

    monkeypatch.setattr(subprocess, "run", fake_run_409)
    with pytest.raises(ConflictError) as exc_info:
        driver.status()
    assert exc_info.value.status_code == 409

    def fake_run_oserror(*args, **kwargs):
        raise PermissionError("permission denied")

    monkeypatch.setattr(subprocess, "run", fake_run_oserror)
    with pytest.raises(DriverError, match="execution/decoding error"):
        driver.status()


def test_monitor_transient_error_recovery_consecutive_halt_and_bounded_resident_loop(tmp_path: Path) -> None:
    driver = FakePhoneDriver()
    class _FlakyOcr(MockOcrEngine):
        def __init__(self) -> None:
            super().__init__()
            self.fail_next = False

        def read_boxes(self, image_bytes: bytes) -> list[OcrBox]:
            if self.fail_next:
                self.fail_next = False
                raise OcrError("transient corrupt screenshot")
            return super().read_boxes(image_bytes)

    flaky_ocr = _FlakyOcr()
    f0 = _chat_frame(flaky_ocr, incoming=[("测试群友A", "你好", Box(0.14, 0.25, 0.45, 0.31))])

    # Tick 0: DriverError (transient 1/3), Tick 1: OcrError (transient 2/3), Tick 2: healthy (resets to 0!)
    # Tick 3..5: 3 consecutive DriverErrors -> halts on Tick 5!
    driver.observe_queue = [
        DriverError("bridge jitter 1"),
        Observation("s-corrupt", f0),
        Observation("s-ok", f0),
        DriverError("bridge down 1"),
        DriverError("bridge down 2"),
        DriverError("bridge down 3"),
    ]
    flaky_ocr.fail_next = True

    sleeps: list[float] = []
    cfg = _monitor_cfg(tmp_path)
    with MonitorStore(cfg.state_db_path, cfg.audit_log_path) as store:
        runner = MonitorRunner(
            driver,
            flaky_ocr,
            cfg,
            store,
            RuleDecisionGate([]),
            sleeper=sleeps.append,
            max_consecutive_errors=3,
        )
        reports = runner.run_loop(max_ticks=10, install_signal_handlers=False)

    assert [r.outcome for r in reports] == [
        TickOutcome.TRANSIENT_ERROR,
        TickOutcome.TRANSIENT_ERROR,
        TickOutcome.BASELINE_SEEDED,
        TickOutcome.TRANSIENT_ERROR,
        TickOutcome.TRANSIENT_ERROR,
        TickOutcome.HALTED_PREFLIGHT,
    ]
    assert reports[-1].should_halt is True
    assert len(sleeps) == 5

    # Resident loop (max_ticks=None): bounded report deque + graceful request_stop() exit
    driver2 = FakePhoneDriver()
    driver2.observe_queue = [Observation(f"r-{i}", f0) for i in range(10)]
    cfg2 = _monitor_cfg(tmp_path / "resident")
    with MonitorStore(cfg2.state_db_path, cfg2.audit_log_path) as store2:
        ticks_seen = {"n": 0}
        runner2 = MonitorRunner(
            driver2,
            flaky_ocr,
            cfg2,
            store2,
            RuleDecisionGate([]),
            max_retained_reports=3,
            sleeper=lambda _: None,
        )

        def _stop_after_seven(_delay: float) -> None:
            ticks_seen["n"] += 1
            if ticks_seen["n"] >= 7:
                runner2.request_stop()

        runner2._sleeper = _stop_after_seven
        res_reports = runner2.run_loop(max_ticks=None, install_signal_handlers=False)
    assert len(res_reports) == 3  # capped at max_retained_reports=3 despite running 7 ticks!
    assert [r.tick_index for r in res_reports] == [4, 5, 6]


def test_llm_client_retries_retry_after_size_cap_timeout_cap_and_per_tick_circuit_breaker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # 1. Timeout upper cap & validation
    cfg_capped = LlmConfig(
        enabled=True,
        base_url="https://api.example.com/v1",
        api_key="k",
        model="m",
        timeout_seconds=999.0,
    )
    assert cfg_capped.timeout_seconds == MAX_LLM_TIMEOUT_SECONDS
    for bad_t in (True, math.nan, math.inf):
        with pytest.raises(ValueError):
            LlmConfig(timeout_seconds=bad_t)  # type: ignore[arg-type]

    # 2. HTTP 429 (with Retry-After) + HTTP 503 retry and succeed on attempt 3
    class _OkResp:
        def __init__(self, body: bytes) -> None:
            self._body = body

        def read(self, amt: int = -1) -> bytes:
            return self._body if amt < 0 else self._body[:amt]

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return None

    attempts = {"n": 0}
    sleeps: list[float] = []

    def fake_urlopen_retry(req, timeout=30.0):
        attempts["n"] += 1
        if attempts["n"] == 1:
            hdrs = Message()
            hdrs["Retry-After"] = "1.5"
            raise urllib.error.HTTPError(req.full_url, 429, "Too Many Requests", hdrs, io.BytesIO(b""))
        if attempts["n"] == 2:
            raise urllib.error.HTTPError(req.full_url, 503, "Unavailable", Message(), io.BytesIO(b""))
        return _OkResp(json.dumps({"choices": [{"message": {"content": "ok-after-retry"}}]}).encode("utf-8"))

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen_retry)
    client = OpenAICompatibleLlmClient(
        LlmConfig(enabled=True, base_url="https://api.example.com/v1", api_key="k", model="m", max_retries=2),
        sleeper=sleeps.append,
    )
    assert client.chat_completion([{"role": "user", "content": "hi"}]) == "ok-after-retry"
    assert attempts["n"] == 3
    assert sleeps == [1.5, 2.0]

    # 3. Non-retryable 401 fails immediately without sleeping and trips per-tick circuit
    attempts_401 = {"n": 0}

    def fake_urlopen_401(req, timeout=30.0):
        attempts_401["n"] += 1
        raise urllib.error.HTTPError(req.full_url, 401, "Unauthorized", Message(), io.BytesIO(b""))

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen_401)
    client_401 = OpenAICompatibleLlmClient(
        LlmConfig(enabled=True, base_url="https://api.example.com/v1", api_key="k", model="m", max_retries=2),
        sleeper=lambda _: pytest.fail("401 must not sleep/retry"),
    )
    with pytest.raises(LlmClientError, match="HTTP 401"):
        client_401.chat_completion([{"role": "user", "content": "hi"}])
    assert attempts_401["n"] == 1
    assert client_401.tick_circuit_tripped is True

    # 4. Oversized response body rejected cleanly by max_response_bytes cap
    oversized = b"x" * 5000
    monkeypatch.setattr(urllib.request, "urlopen", lambda req, timeout=30.0: _OkResp(oversized))
    client_cap = OpenAICompatibleLlmClient(
        LlmConfig(
            enabled=True,
            base_url="https://api.example.com/v1",
            api_key="k",
            model="m",
            max_response_bytes=4096,
        ),
        sleeper=lambda _: None,
    )
    with pytest.raises(LlmClientError, match="exceeded max_response_bytes"):
        client_cap.chat_completion([{"role": "user", "content": "hi"}])

    # 5. Per-tick fail-fast circuit breaker inside MonitorRunner prevents N*timeout across 3 pending messages
    llm_calls = {"n": 0}

    def fake_urlopen_timeout(req, timeout=30.0):
        llm_calls["n"] += 1
        raise TimeoutError("timed out")

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen_timeout)
    tick_client = OpenAICompatibleLlmClient(
        LlmConfig(enabled=True, base_url="https://api.example.com/v1", api_key="k", model="m", max_retries=0),
        sleeper=lambda _: None,
    )
    driver = FakePhoneDriver()
    ocr = MockOcrEngine()
    f_base = _chat_frame(ocr, incoming=[("测试群友A", "开场", Box(0.14, 0.22, 0.45, 0.28))], suffix=b"b")
    f_3new = _chat_frame(
        ocr,
        incoming=[
            ("测试群友A", "开场", Box(0.14, 0.22, 0.45, 0.28)),
            ("测试群友B", "问题一", Box(0.14, 0.34, 0.45, 0.40)),
            ("测试群友C", "问题二", Box(0.14, 0.46, 0.45, 0.52)),
            ("测试群友D", "问题三", Box(0.14, 0.58, 0.45, 0.64)),
        ],
        suffix=b"3n",
    )
    driver.observe_queue = [Observation("t0", f_base), Observation("t1", f_3new)]
    cfg = _monitor_cfg(tmp_path / "breaker")
    with MonitorStore(cfg.state_db_path, cfg.audit_log_path) as store:
        runner = MonitorRunner(driver, ocr, cfg, store, LlmIntentGate(tick_client), sleeper=lambda _: None)
        assert runner.run_tick(0).outcome == TickOutcome.BASELINE_SEEDED
        r1 = runner.run_tick(1)
        assert r1.outcome == TickOutcome.TRANSIENT_ERROR
        assert "circuit breaker" in r1.message
        # Only 1 LLM HTTP attempt for all 3 queued messages!
        assert llm_calls["n"] == 1
        # All 3 messages stay pending in queue for retry on a future tick
        assert len(store.list_pending("示例测试群")) == 3


def test_storage_sql_index_prune_audit_rotation_evidence_prune_and_discontinuity_audit(tmp_path: Path) -> None:
    db_path = tmp_path / "state" / "gov.db"
    audit_path = tmp_path / "state" / "audit.jsonl"

    # 1. Seed a legacy seen_messages table without content_hash to verify migration + backfill + SQL index
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        """
        CREATE TABLE seen_messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            fingerprint TEXT UNIQUE NOT NULL,
            group_title TEXT NOT NULL,
            role TEXT NOT NULL,
            sender TEXT NOT NULL,
            text TEXT NOT NULL,
            is_baseline INTEGER NOT NULL DEFAULT 0,
            first_seen_at REAL NOT NULL,
            last_seen_at REAL NOT NULL
        )
        """
    )
    conn.execute(
        "INSERT INTO seen_messages"
        "(fingerprint, group_title, role, sender, text, is_baseline, first_seen_at, last_seen_at) "
        "VALUES ('legacy-1', '示例测试群', 'incoming', '测试群友A', '老消息', 1, 100.0, 100.0)"
    )
    conn.commit()
    conn.close()

    with MonitorStore(db_path, audit_path, audit_max_bytes=1024, audit_backup_count=2) as store:
        # Backfilled row is found via SQL content_hash index
        assert store.has_recent_message_content(
            "示例测试群", MessageRole.INCOMING, "测试群友A", "老消息", since_ts=50.0
        )
        plan = store._conn.execute(
            "EXPLAIN QUERY PLAN SELECT 1 FROM seen_messages "
            "WHERE group_title = ? AND role = ? AND content_hash = ? AND last_seen_at >= ? LIMIT 1",
            ("示例测试群", "incoming", "abc", 0.0),
        ).fetchall()
        plan_text = " ".join(str(row["detail"]) for row in plan)
        assert "idx_seen_group_role_hash_time" in plan_text

        # Populate old & fresh rows across all 4 tables + an active pending row
        store.record_seen(
            ExtractedMessage("示例测试群", MessageRole.INCOMING, "新消息", sender="测试群友B", fingerprint="fresh-1"),
            now=2000.0,
        )
        store.record_reply_decision(
            fingerprint="old-rep", group_title="示例测试群", mode="live", status="sent_confirmed", now=100.0
        )
        store.record_reply_decision(
            fingerprint="new-rep", group_title="示例测试群", mode="live", status="sent_confirmed", now=2000.0
        )
        store.record_send_event("示例测试群", "sent_confirmed", now=100.0)
        store.record_send_event("示例测试群", "sent_confirmed", now=2000.0)
        store.enqueue_pending(
            ExtractedMessage("示例测试群", MessageRole.INCOMING, "已处理旧项", sender="A", fingerprint="pq-old"),
            now=100.0,
        )
        store.mark_pending_status("pq-old", "processed", now=100.0)
        store.enqueue_pending(
            ExtractedMessage("示例测试群", MessageRole.INCOMING, "活跃待处理", sender="B", fingerprint="pq-live"),
            ttl_seconds=5000.0,
            now=100.0,
        )

        deleted = store.prune_storage(
            now=2000.0,
            seen_ttl_seconds=500.0,
            reply_ttl_seconds=500.0,
            send_events_ttl_seconds=500.0,
            pending_ttl_seconds=500.0,
        )
        assert deleted == {"seen_messages": 1, "reply_history": 1, "send_events": 1, "pending_queue": 1}
        # Active pending item was NOT deleted!
        assert [m.fingerprint for m in store.list_pending("示例测试群", now=2000.0)] == ["pq-live"]

        # Audit JSONL size-based rotation with 0o600 permissions
        for idx in range(25):
            store.append_audit({"idx": idx, "pad": "x" * 100})
        rotated_1 = audit_path.with_name(f"{audit_path.name}.1")
        assert audit_path.is_file() and rotated_1.is_file()
        assert (rotated_1.stat().st_mode & 0o777) == 0o600

    # 2. Evidence run retention pruning
    ev_root = tmp_path / "runs"
    for idx in range(5):
        ev = RunEvidence(ev_root, run_id=f"run-{idx}")
        mtime = 1000.0 + idx * 100.0
        os.utime(ev.dir, (mtime, mtime))
    removed = prune_evidence_runs(ev_root, max_age_seconds=250.0, max_runs=2, now=1400.0)
    assert removed == 3
    assert sorted(p.name for p in ev_root.iterdir() if p.is_dir()) == ["run-3", "run-4"]

    # 3. Discontinuity after long gap logs suspected_missed_count in audit.jsonl
    driver = FakePhoneDriver()
    ocr = MockOcrEngine()
    f0 = _chat_frame(ocr, incoming=[("测试群友A", "旧消息", Box(0.14, 0.25, 0.45, 0.31))], suffix=b"d0")
    f_jump = _chat_frame(
        ocr,
        incoming=[
            ("测试群友B", "漏掉消息1", Box(0.14, 0.25, 0.48, 0.31)),
            ("测试群友C", "漏掉消息2", Box(0.14, 0.38, 0.48, 0.44)),
        ],
        suffix=b"d1",
    )
    driver.observe_queue = [Observation("d0", f0), Observation("d1", f_jump)]
    now = {"t": 1000.0}
    cfg_disc = _monitor_cfg(tmp_path / "disc")
    with MonitorStore(cfg_disc.state_db_path, cfg_disc.audit_log_path) as store_d:
        runner = MonitorRunner(driver, ocr, cfg_disc, store_d, RuleDecisionGate([]), clock=lambda: now["t"])
        assert runner.run_tick(0).outcome == TickOutcome.BASELINE_SEEDED
        now["t"] += 200.0  # > short_discontinuity_window_seconds (90s)
        rep = runner.run_tick(1)
        assert rep.outcome == TickOutcome.BASELINE_SEEDED
        assert "suspected_missed=2" in rep.message
    audit_lines = [
        json.loads(line) for line in cfg_disc.audit_log_path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    assert audit_lines[-1]["discontinuity"] is True
    assert audit_lines[-1]["suspected_missed_count"] == 2


def test_cli_unified_exit_codes_and_max_ticks_validation(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # 1. monitor --max-ticks <= 0 rejected with exit code 2
    assert cli_main(["monitor", "--max-ticks", "0"]) == 2
    assert "--max-ticks" in capsys.readouterr().err
    assert cli_main(["monitor", "--max-ticks", "-3"]) == 2
    assert "--max-ticks" in capsys.readouterr().err

    # 2. doctor returns 0 when ready and 2 (unified) when unready
    monkeypatch.setattr(
        IphoneAgentCliDriver,
        "status",
        lambda self: {"device": True, "screen": True, "calibrated": True, "ready": False, "guard": {}},
    )
    monkeypatch.setattr(
        IphoneAgentCliDriver,
        "capabilities",
        lambda self: {"features": {"clipboard": {"staged_available": True, "protocol": "staged-v1"}}},
    )
    monkeypatch.setattr(IphoneAgentCliDriver, "skill_list", lambda self: ["wechat-send"])
    assert cli_main(["doctor"]) == 2

    monkeypatch.setattr(
        IphoneAgentCliDriver,
        "status",
        lambda self: {"device": True, "screen": True, "calibrated": True, "ready": True, "guard": {}},
    )
    assert cli_main(["doctor"]) == 0
