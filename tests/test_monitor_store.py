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
            ("incoming", "", "在吗", 1),
            ("self", "", "在的，请问", 1),
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
    m_normal = ExtractedMessage(
        g, MessageRole.INCOMING, "价格多少", sender="测试群友A", mentioned_self=False, fingerprint="pq-1"
    )
    m_burst = ExtractedMessage(
        g, MessageRole.INCOMING, "@示例助手", sender="测试群友A", mentioned_self=True, fingerprint="pq-2"
    )
    m_other = ExtractedMessage(
        g, MessageRole.INCOMING, "请问活动规则", sender="测试群友B", mentioned_self=False, fingerprint="pq-3"
    )
    m_other_mention = ExtractedMessage(
        g, MessageRole.INCOMING, "@示例助手 帮查一下", sender="测试群友C", mentioned_self=True, fingerprint="pq-4"
    )

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


def test_store_single_instance_flock_and_atomic_batch_rollback(tmp_path: Path) -> None:
    import pytest

    from wechat_agent.monitor.store import MonitorInstanceLockError

    db_path = tmp_path / "state" / "locked.db"
    with MonitorStore(db_path) as store1:
        assert store1.lock_path.is_file()
        assert (store1.lock_path.stat().st_mode & 0o777) == 0o600
        with pytest.raises(MonitorInstanceLockError, match="already holds lock"):
            MonitorStore(db_path)

        # Verify commit_arrived_batch atomically persists seen_messages + viewport + pending_queue together
        g = "示例测试群"
        m1 = ExtractedMessage(g, MessageRole.INCOMING, "请问价格", sender="测试群友A", mentioned_self=False)
        m2 = ExtractedMessage(g, MessageRole.INCOMING, "@示例助手", sender="测试群友A", mentioned_self=True)
        events, merged = store1.commit_arrived_batch(
            g,
            [m1, m2],
            [("incoming", "测试群友A", "请问价格"), ("incoming", "测试群友A", "@示例助手")],
            now=100.0,
        )
        assert len(events) == 2
        assert len(merged) == 1
        assert merged[0].text == "请问价格\n@示例助手"
        assert store1.get_last_viewport(g) == [
            ("incoming", "测试群友A", "请问价格", 1),
            ("incoming", "测试群友A", "@示例助手", 1),
        ]
        assert len(store1.list_pending(g, now=105.0)) == 1

    # Once closed, a new instance acquires the lock cleanly
    with MonitorStore(db_path) as store2:
        assert store2.next_event_seq("示例测试群") == 3


def test_rate_limit_config_validation_and_quiet_range_bounds() -> None:
    import math

    import pytest

    from wechat_agent.guards.rate_limit import _parse_quiet_range

    assert _parse_quiet_range("") is None
    assert _parse_quiet_range("off") is None
    assert _parse_quiet_range("23:00-07:30") == ((23, 0), (7, 30))

    for bad_quiet in ("25:00-07:00", "23:60-07:00", "23:00-24:00", "23:00-07:99", "abc-def", "23:00"):
        with pytest.raises(ValueError):
            _parse_quiet_range(bad_quiet)
        with pytest.raises(ValueError):
            RateLimitConfig(quiet_hours=bad_quiet)

    for bad_kwargs in (
        {"global_min_interval_seconds": 2.0},
        {"global_min_interval_seconds": math.nan},
        {"global_min_interval_seconds": True},
        {"global_max_per_hour": 0},
        {"global_max_per_hour": 31},
        {"global_max_per_hour": True},
        {"per_group_cooldown_seconds": -1.0},
        {"per_group_cooldown_seconds": math.inf},
        {"per_group_max_per_hour": 0},
        {"per_group_max_per_hour": 31},
        {"human_takeover_cooldown_seconds": -5.0},
        {"human_takeover_cooldown_seconds": math.nan},
        {"max_reply_chars": 0},
        {"max_reply_chars": 5000},
        {"forbidden_reply_patterns": ("[invalid",)},
    ):
        with pytest.raises(ValueError):
            RateLimitConfig(**bad_kwargs)


def test_monitor_config_strict_bools_timeouts_forbidden_list_and_multigroup_warning(tmp_path: Path) -> None:
    import warnings

    import pytest

    from wechat_agent.config import load_config
    from wechat_agent.guards.rate_limit import DEFAULT_FORBIDDEN_PATTERNS
    from wechat_agent.monitor.config import load_monitor_config

    # bool("false") == True bug MUST be rejected on every boolean config switch
    for snippet in (
        'allow_live_send: "false"\n',
        'mention_must_reply: "false"\n',
        'require_mounted_skill: "false"\n',
        'target_groups:\n  - name: "示例测试群"\n    allow_dry_run_paste: "false"\n',
        'target_groups:\n  - name: "示例测试群"\n    mention_must_reply: "false"\n',
        'llm:\n  enabled: "false"\n',
    ):
        cfg_file = tmp_path / "bad_bool.yaml"
        cfg_file.write_text(snippet, encoding="utf-8")
        with pytest.raises(ValueError, match="boolean"):
            load_monitor_config(cfg_file)

    send_bad = tmp_path / "bad_send.yaml"
    send_bad.write_text('require_mounted_skill: "false"\n', encoding="utf-8")
    with pytest.raises(ValueError, match="boolean"):
        load_config(send_bad)

    # Explicit empty forbidden_reply_patterns: [] stays empty (distinct from omitted defaults),
    # timeouts clamp to safe lower bounds, and multi-group config warns clearly about active_group.
    good = tmp_path / "multi.yaml"
    good.write_text(
        """
mode: observe-only
allow_live_send: false
cli_timeout_seconds: 1
paste_flow_timeout_seconds: 2
forbidden_reply_patterns: []
target_groups:
  - name: "示例群A"
  - name: "示例群B"
""",
        encoding="utf-8",
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cfg = load_monitor_config(good)
    assert cfg.allow_live_send is False
    assert cfg.cli_timeout_seconds == 5.0
    assert cfg.paste_flow_timeout_seconds == 15.0
    assert cfg.rate_limits.forbidden_reply_patterns == ()
    assert cfg.active_group is not None and cfg.active_group.name == "示例群A"
    assert any("target_groups[0]" in str(w.message) for w in caught)

    # Omitted forbidden_reply_patterns keeps DEFAULT_FORBIDDEN_PATTERNS
    omitted = tmp_path / "default_forb.yaml"
    omitted.write_text('target_groups:\n  - name: "示例群A"\n', encoding="utf-8")
    cfg_def = load_monitor_config(omitted)
    assert cfg_def.rate_limits.forbidden_reply_patterns == DEFAULT_FORBIDDEN_PATTERNS


def test_claim_bot_reply_match_tightened_prevents_short_substring_takeover_bypass(tmp_path: Path) -> None:
    db_path = tmp_path / "state" / "claim.db"
    g = "示例测试群"
    with MonitorStore(db_path) as store:
        # 1. Short bot reply "好的" must NOT be swallowed by a longer human message containing "好的"
        store.record_reply_decision(
            fingerprint="bot-short-1",
            group_title=g,
            mode="live",
            status="sent_confirmed",
            reply_text="好的",
            now=100.0,
        )
        assert store.claim_bot_reply_match(g, "好的，我现在马上处理您的订单", since_ts=50.0) is False
        # Exact bot echo still matches cleanly and consumes the record
        assert store.claim_bot_reply_match(g, "好的", since_ts=50.0) is True
        # Consumed record cannot be claimed a second time
        assert store.claim_bot_reply_match(g, "好的", since_ts=50.0) is False

        # 2. Long bot reply (>= 10 canonical chars) tolerates minor OCR edge truncation (>= 85% length ratio)
        store.record_reply_decision(
            fingerprint="bot-long-1",
            group_title=g,
            mode="live",
            status="sent_confirmed",
            reply_text="请按群内官方团购链接下单即可哦",
            now=200.0,
        )
        # Much longer human message containing the bot reply (ratio < 0.85) must NOT claim it
        assert (
            store.claim_bot_reply_match(
                g,
                "补充一下，请按群内官方团购链接下单即可哦，有任何问题随时私信群主确认",
                since_ts=150.0,
            )
            is False
        )
        # Single-char OCR truncation (14/15 = 93.3% >= 85%) DOES match
        assert store.claim_bot_reply_match(g, "请按群内官方团购链接下单即可", since_ts=150.0) is True


def test_claim_bot_reply_match_ocr_punctuation_drop_and_strict_rejections(tmp_path: Path) -> None:
    db_path = tmp_path / "state" / "claim_punct.db"
    g = "好物分享·品牌特卖3群(198)"
    reply = (
        "满减和包邮门槛每期活动不太一样，具体看群里官方团购链接或小程序里的活动说明；"
        "单件运费也以下单页显示的为准，拿不准就找群主确认这期规则。"
    )
    ocr_dropped_comma = (
        "满减和包邮门槛每期活动不太\n"
        "一样，具体看群里官方团购链\n"
        "接或小程序里的活动说明；单\n"
        "件运费也以下单页显示的为准\n"
        "拿不准就找群主确认这期规则。"
    )
    with MonitorStore(db_path) as store:
        # 1. Dropped internal comma in multiline OCR bubble matches once and consumes record
        store.record_reply_decision(
            fingerprint="bot-punct-1",
            group_title=g,
            mode="live",
            status="sent_confirmed",
            reply_text=reply,
            now=300.0,
        )
        assert store.claim_bot_reply_match(g, ocr_dropped_comma, since_ts=250.0) is True
        assert store.claim_bot_reply_match(g, ocr_dropped_comma, since_ts=250.0) is False

        # 2. Short reply typo MUST be rejected
        store.record_reply_decision(
            fingerprint="bot-short-typo",
            group_title=g,
            mode="live",
            status="sent_confirmed",
            reply_text="请明天发货",
            now=310.0,
        )
        assert store.claim_bot_reply_match(g, "请后天发货", since_ts=250.0) is False

        # 3. Decimal-dot drop MUST be rejected
        store.record_reply_decision(
            fingerprint="bot-decimal",
            group_title=g,
            mode="live",
            status="sent_confirmed",
            reply_text="价格19.9元",
            now=320.0,
        )
        assert store.claim_bot_reply_match(g, "价格199元", since_ts=250.0) is False

        # 4. Long unrelated human message MUST be rejected
        store.record_reply_decision(
            fingerprint="bot-long-unrelated",
            group_title=g,
            mode="live",
            status="sent_confirmed",
            reply_text="这期商品和价格看群里的官方团购链接或小程序确认吧",
            now=330.0,
        )
        assert (
            store.claim_bot_reply_match(
                g,
                "请联系售后客服提交订单信息并等待人工处理回复吧",
                since_ts=250.0,
            )
            is False
        )


