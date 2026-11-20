"""Offline unit tests for SQLite WAL MonitorStore cross-restart dedup and MonitorRateLimiter."""

from datetime import datetime
from pathlib import Path

from wechat_agent.guards.rate_limit import MonitorRateLimiter, RateLimitConfig
from wechat_agent.monitor.models import ExtractedMessage, MessageRole
from wechat_agent.monitor.store import MonitorStore


def test_store_permissions_and_cross_restart_dedup_and_context(tmp_path: Path) -> None:
    db_path = tmp_path / "state" / "monitor.db"
    audit_path = tmp_path / "state" / "audit.jsonl"

    m1 = ExtractedMessage(
        group_title="测试群",
        role=MessageRole.INCOMING,
        sender="张三",
        text="请问明天开门吗",
        mentioned_self=True,
        fingerprint="fp-1",
    )
    m2 = ExtractedMessage(
        group_title="测试群",
        role=MessageRole.SELF,
        sender="",
        text="明天正常营业",
        fingerprint="fp-2",
    )

    with MonitorStore(db_path, audit_path) as store1:
        assert (db_path.parent.stat().st_mode & 0o777) == 0o700
        assert (db_path.stat().st_mode & 0o777) == 0o600
        assert store1.record_seen(m1, is_baseline=False, now=100.0) is True
        assert store1.record_seen(m1, is_baseline=False, now=101.0) is False
        assert store1.record_seen(m2, is_baseline=False, now=102.0) is True
        store1.mark_baseline_seeded("测试群", now=102.0)
        assert store1.record_reply_decision(
            fingerprint=m1.fingerprint,
            group_title="测试群",
            mode="live",
            status="sent_confirmed",
            reply_text="明天正常营业",
            now=102.0,
        ) is True
        store1.record_send_event("测试群", "sent_confirmed", now=102.0)
        store1.append_audit({"event": "test", "transaction_id": "secret-tx"})

    # Reopen brand new MonitorStore instance simulating process restart
    with MonitorStore(db_path, audit_path) as store2:
        assert store2.is_baseline_seeded("测试群") is True
        assert store2.has_seen("fp-1") is True
        assert store2.record_seen(m1, now=200.0) is False
        assert store2.record_reply_decision(
            fingerprint=m1.fingerprint,
            group_title="测试群",
            mode="live",
            status="pending",
            now=200.0,
        ) is False
        turns = store2.get_recent_context("测试群", limit=8)
        assert [t.role for t in turns] == [MessageRole.INCOMING, MessageRole.SELF]
        assert turns[0].format_line() == "[对方/张三]: 请问明天开门吗"
        assert turns[1].format_line() == "[我]: 明天正常营业"
        assert store2.claim_bot_reply_match("测试群", "明天正常营业", since_ts=50.0) is True
        assert store2.claim_bot_reply_match("测试群", "明天正常营业", since_ts=50.0) is False

    assert (audit_path.stat().st_mode & 0o777) == 0o600
    assert "[REDACTED]" in audit_path.read_text(encoding="utf-8")


def test_rate_limiter_quiet_hours_takeover_and_dual_caps(tmp_path: Path) -> None:
    db_path = tmp_path / "state" / "rl.db"
    cfg = RateLimitConfig(
        global_min_interval_seconds=15.0,
        global_max_per_hour=15,
        per_group_cooldown_seconds=60.0,
        per_group_max_per_hour=6,
        human_takeover_cooldown_seconds=600.0,
        quiet_hours="23:00-07:00",
    )

    daytime = datetime(2026, 11, 19, 14, 0, 0).timestamp()
    nighttime = datetime(2026, 11, 19, 23, 30, 0).timestamp()

    with MonitorStore(db_path) as store:
        rl = MonitorRateLimiter(store, cfg)
        assert rl.check_can_reply("群A", now=nighttime).block_kind == "quiet_hours"
        assert rl.check_can_reply("群A", now=daytime).allowed is True

        # Human takeover silences group for 600s
        rl.activate_human_takeover("群A", now=daytime)
        assert rl.check_can_reply("群A", now=daytime + 100).block_kind == "human_takeover"
        assert rl.check_can_reply("群B", now=daytime + 100).allowed is True

        # Global interval (15s) and per-group cooldown (60s) across reopen
        t0 = daytime + 700.0
        store.record_send_event("群A", "sent_confirmed", now=t0)
        assert rl.check_can_reply("群B", now=t0 + 10).block_kind == "global_interval"
        assert rl.check_can_reply("群B", now=t0 + 20).allowed is True
        assert rl.check_can_reply("群A", now=t0 + 30).block_kind == "group_cooldown"
        assert rl.check_can_reply("群A", now=t0 + 65).allowed is True

        # Per-group hourly cap (6/h)
        for i in range(1, 6):
            store.record_send_event("群A", "sent_confirmed", now=t0 + i * 70.0)
        assert rl.check_can_reply("群A", now=t0 + 500.0).block_kind == "group_hourly_cap"

        # Content guard rejects sensitive / forbidden replies
        assert rl.check_reply_content("请把验证码发我")[0] is False
        assert rl.check_reply_content("联系13812345678")[0] is False
        assert rl.check_reply_content("明天上午十点开门")[0] is True


def test_store_viewport_event_seq_recent_content_and_legacy_migration(tmp_path: Path) -> None:
    import sqlite3

    db_path = tmp_path / "state" / "legacy.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    # Create legacy group_state without last_viewport_json or event_seq
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        """
        CREATE TABLE group_state (
            group_title TEXT PRIMARY KEY,
            baseline_seeded INTEGER NOT NULL DEFAULT 0,
            last_seen_at REAL NOT NULL DEFAULT 0.0,
            last_reply_at REAL NOT NULL DEFAULT 0.0,
            takeover_until REAL NOT NULL DEFAULT 0.0
        )
        """
    )
    conn.execute(
        "INSERT INTO group_state(group_title, baseline_seeded, last_seen_at) VALUES ('示例分享·测试3群(198)', 1, 50.0)"
    )
    conn.commit()
    conn.close()

    with MonitorStore(db_path) as store:
        assert store.is_baseline_seeded("示例分享•测试3群（198）") is True
        assert store.is_baseline_seeded("示例分享•测试3群（199）") is False
        assert store.get_last_viewport("示例分享•测试3群（198）") == []
        seq1 = store.next_event_seq("示例分享•测试3群（198）")
        seq2 = store.next_event_seq("示例分享·测试3群(198)")
        seq_other = store.next_event_seq("示例分享·测试3群(199)")
        assert (seq1, seq2, seq_other) == (1, 2, 1)

        viewport = [
            ("incoming", "", "在吗"),
            ("self", "", "在的，请问"),
        ]
        store.save_last_viewport("示例分享•测试3群（198）", viewport, now=100.0)
        store.record_seen(
            ExtractedMessage(
                group_title="示例分享•测试3群（198）",
                role=MessageRole.INCOMING,
                sender="",
                text="在吗",
                fingerprint="evt-1",
            ),
            now=100.0,
        )

    # Reopen store: migrated columns & viewport state survive process restart and isolate (198) from (199)
    with MonitorStore(db_path) as store2:
        assert store2.get_last_viewport("示例分享·测试3群(198)") == [
            ("incoming", "", "在吗"),
            ("self", "", "在的，请问"),
        ]
        assert store2.get_last_viewport("示例分享·测试3群(199)") == []
        assert store2.next_event_seq("示例分享·测试3群(198)") == 3
        assert store2.has_recent_message_content(
            "示例分享•测试3群（198）", MessageRole.INCOMING, "", "在 吗", since_ts=90.0
        ) is False
        assert store2.has_recent_message_content(
            "示例分享•测试3群（198）", MessageRole.INCOMING, "", "在吗", since_ts=90.0
        ) is True
        assert store2.has_recent_message_content(
            "示例分享·测试3群(199)", MessageRole.INCOMING, "", "在吗", since_ts=90.0
        ) is False
        assert store2.has_recent_message_content(
            "示例分享•测试3群（198）", MessageRole.INCOMING, "", "在吗", since_ts=105.0
        ) is False


def test_store_pending_queue_burst_merge_priority_and_ttl_expiry(tmp_path: Path) -> None:
    db_path = tmp_path / "state" / "pq.db"
    g = "示例分享•测试3群（198）"
    m_normal = ExtractedMessage(g, MessageRole.INCOMING, "价格多少", sender="测试群友A", mentioned_self=False, fingerprint="pq-1")
    m_burst = ExtractedMessage(g, MessageRole.INCOMING, "@示例助手", sender="测试群友A", mentioned_self=True, fingerprint="pq-2")
    m_other = ExtractedMessage(g, MessageRole.INCOMING, "请问活动规则", sender="测试群友B", mentioned_self=False, fingerprint="pq-3")
    m_other_mention = ExtractedMessage(g, MessageRole.INCOMING, "@示例助手 帮查一下", sender="测试群友C", mentioned_self=True, fingerprint="pq-4")

    with MonitorStore(db_path) as store:
        fp_a = store.enqueue_pending(m_normal, ttl_seconds=180.0, burst_merge_window_seconds=5.0, now=1000.0)
        fp_a2 = store.enqueue_pending(m_burst, ttl_seconds=180.0, burst_merge_window_seconds=5.0, now=1003.0)
        assert fp_a == fp_a2 == "pq-1"
        store.enqueue_pending(m_other, ttl_seconds=180.0, burst_merge_window_seconds=5.0, now=1010.0)
        store.enqueue_pending(m_other_mention, ttl_seconds=180.0, burst_merge_window_seconds=5.0, now=1015.0)

    with MonitorStore(db_path) as store2:
        pending = store2.list_pending(g, now=1020.0)
        # Priority: @self first (pq-1, pq-4 FIFO), then non-mention (pq-3)
        assert [(p.fingerprint, p.text, p.mentioned_self) for p in pending] == [
            ("pq-1", "价格多少\n@示例助手", True),
            ("pq-4", "@示例助手 帮查一下", True),
            ("pq-3", "请问活动规则", False),
        ]
        store2.mark_pending_status("pq-1", "processed", now=1025.0)
        assert [p.fingerprint for p in store2.list_pending(g, now=1025.0)] == ["pq-4", "pq-3"]

        # Advance past 180s TTL of pq-4 (expires at 1015+180=1195) and pq-3 (1010+180=1190) -> all expired
        assert store2.list_pending(g, now=1200.0) == []
        assert store2.has_pending(g, now=1200.0) is False
        statuses = dict(
            store2._conn.execute("SELECT fingerprint, status FROM pending_queue ORDER BY id ASC").fetchall()
        )
        assert statuses == {"pq-1": "processed", "pq-3": "expired", "pq-4": "expired"}
