"""Cross-restart SQLite WAL persistence and 0600 JSONL audit logging for M2 monitoring."""

from __future__ import annotations

import json
import os
import re
import sqlite3
import time
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

from wechat_agent.evidence import redact_dict
from wechat_agent.vision.geometry import canonicalize_chat_title, normalize_text

from .extractor import ViewportItem, _canonical_for_hash
from .models import ContextTurn, ExtractedMessage, MessageRole


def _ensure_private_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    os.chmod(path.parent, 0o700)


class MonitorStore:
    """Persistent state store for deduplication, idempotent reply locking, and rate-limit history."""

    def __init__(self, db_path: Path, audit_log_path: Path | None = None) -> None:
        self.db_path = Path(db_path)
        self.audit_log_path = Path(audit_log_path) if audit_log_path is not None else None
        _ensure_private_parent(self.db_path)
        if self.audit_log_path is not None:
            _ensure_private_parent(self.audit_log_path)

        self._conn = sqlite3.connect(str(self.db_path), timeout=10.0)
        self._conn.row_factory = sqlite3.Row
        os.chmod(self.db_path, 0o600)
        self._init_schema()

    def _init_schema(self) -> None:
        with self._conn:
            self._conn.execute("PRAGMA journal_mode=WAL;")
            self._conn.execute("PRAGMA synchronous=NORMAL;")
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS seen_messages (
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
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS reply_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    fingerprint TEXT UNIQUE NOT NULL,
                    group_title TEXT NOT NULL,
                    mode TEXT NOT NULL,
                    status TEXT NOT NULL,
                    intent TEXT NOT NULL DEFAULT '',
                    confidence REAL NOT NULL DEFAULT 0.0,
                    risk_score REAL NOT NULL DEFAULT 0.0,
                    matched_rule TEXT,
                    reply_text TEXT,
                    reason TEXT NOT NULL DEFAULT '',
                    run_id TEXT NOT NULL DEFAULT '',
                    created_at REAL NOT NULL
                )
                """
            )
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS group_state (
                    group_title TEXT PRIMARY KEY,
                    baseline_seeded INTEGER NOT NULL DEFAULT 0,
                    last_seen_at REAL NOT NULL DEFAULT 0.0,
                    last_reply_at REAL NOT NULL DEFAULT 0.0,
                    takeover_until REAL NOT NULL DEFAULT 0.0,
                    last_viewport_json TEXT NOT NULL DEFAULT '[]',
                    event_seq INTEGER NOT NULL DEFAULT 0
                )
                """
            )
            cols = {
                str(r["name"])
                for r in self._conn.execute("PRAGMA table_info(group_state)").fetchall()
            }
            if "last_viewport_json" not in cols:
                self._conn.execute(
                    "ALTER TABLE group_state ADD COLUMN last_viewport_json TEXT NOT NULL DEFAULT '[]'"
                )
            if "event_seq" not in cols:
                self._conn.execute(
                    "ALTER TABLE group_state ADD COLUMN event_seq INTEGER NOT NULL DEFAULT 0"
                )
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS send_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    group_title TEXT NOT NULL,
                    sent_at REAL NOT NULL,
                    status TEXT NOT NULL
                )
                """
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_seen_group_time ON seen_messages(group_title, id)"
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_send_events_time ON send_events(sent_at, group_title)"
            )

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> MonitorStore:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _ensure_group_row(self, group_title: str) -> str:
        norm = canonicalize_chat_title(group_title)
        self._conn.execute(
            "INSERT OR IGNORE INTO group_state(group_title) VALUES (?)",
            (norm,),
        )
        return norm

    def is_baseline_seeded(self, group_title: str) -> bool:
        norm = canonicalize_chat_title(group_title)
        row = self._conn.execute(
            "SELECT baseline_seeded FROM group_state WHERE group_title = ?",
            (norm,),
        ).fetchone()
        return bool(row and row["baseline_seeded"])

    def mark_baseline_seeded(self, group_title: str, now: float | None = None) -> None:
        ts = time.time() if now is None else float(now)
        with self._conn:
            norm = self._ensure_group_row(group_title)
            self._conn.execute(
                "UPDATE group_state SET baseline_seeded = 1, last_seen_at = ? WHERE group_title = ?",
                (ts, norm),
            )

    def get_last_viewport(self, group_title: str) -> list[ViewportItem]:
        """Return the persisted ordered viewport tuple sequence from the previous tick."""
        norm = canonicalize_chat_title(group_title)
        row = self._conn.execute(
            "SELECT last_viewport_json FROM group_state WHERE group_title = ?",
            (norm,),
        ).fetchone()
        if not row or not row["last_viewport_json"]:
            return []
        try:
            parsed = json.loads(str(row["last_viewport_json"]))
        except json.JSONDecodeError:
            return []
        if not isinstance(parsed, list):
            return []
        return [
            (str(item[0]), str(item[1]), str(item[2]))
            for item in parsed
            if isinstance(item, (list, tuple)) and len(item) == 3
        ]

    def save_last_viewport(
        self,
        group_title: str,
        items: Sequence[ViewportItem],
        now: float | None = None,
    ) -> None:
        """Persist the ordered visible message sequence for suffix-overlap alignment."""
        ts = time.time() if now is None else float(now)
        payload = json.dumps([[r, s, t] for r, s, t in items], ensure_ascii=False)
        with self._conn:
            norm = self._ensure_group_row(group_title)
            self._conn.execute(
                "UPDATE group_state SET last_viewport_json = ?, last_seen_at = ? WHERE group_title = ?",
                (payload, ts, norm),
            )

    def next_event_seq(self, group_title: str) -> int:
        """Atomically allocate the next monotonic event sequence number for `group_title`."""
        with self._conn:
            norm = self._ensure_group_row(group_title)
            self._conn.execute(
                "UPDATE group_state SET event_seq = event_seq + 1 WHERE group_title = ?",
                (norm,),
            )
            row = self._conn.execute(
                "SELECT event_seq FROM group_state WHERE group_title = ?",
                (norm,),
            ).fetchone()
            return int(row["event_seq"]) if row else 1

    def has_recent_message_content(
        self,
        group_title: str,
        role: MessageRole,
        sender: str,
        text: str,
        *,
        since_ts: float,
    ) -> bool:
        """Check whether `(role, sender, text)` was recently recorded in `seen_messages`."""
        norm_group = canonicalize_chat_title(group_title)
        norm_sender = normalize_text(sender)
        canon_text = _canonical_for_hash(text)
        rows = self._conn.execute(
            """
            SELECT sender, text FROM seen_messages
            WHERE group_title = ? AND role = ? AND last_seen_at >= ?
            """,
            (norm_group, role.value, float(since_ts)),
        ).fetchall()
        return any(
            normalize_text(str(r["sender"] or "")) == norm_sender
            and _canonical_for_hash(str(r["text"] or "")) == canon_text
            for r in rows
        )

    def has_seen(self, fingerprint: str) -> bool:
        row = self._conn.execute(
            "SELECT 1 FROM seen_messages WHERE fingerprint = ?",
            (fingerprint,),
        ).fetchone()
        return row is not None

    def record_seen(
        self,
        msg: ExtractedMessage,
        *,
        is_baseline: bool = False,
        now: float | None = None,
    ) -> bool:
        """Record a message fingerprint. Returns True if newly inserted, False if already seen."""
        ts = time.time() if now is None else float(now)
        norm_group = canonicalize_chat_title(msg.group_title)
        with self._conn:
            self._ensure_group_row(norm_group)
            cur = self._conn.execute(
                """
                INSERT OR IGNORE INTO seen_messages
                    (fingerprint, group_title, role, sender, text, is_baseline, first_seen_at, last_seen_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    msg.fingerprint,
                    norm_group,
                    msg.role.value,
                    msg.sender,
                    msg.text,
                    1 if is_baseline else 0,
                    ts,
                    ts,
                ),
            )
            inserted = cur.rowcount > 0
            if not inserted:
                self._conn.execute(
                    "UPDATE seen_messages SET last_seen_at = ? WHERE fingerprint = ?",
                    (ts, msg.fingerprint),
                )
            self._conn.execute(
                "UPDATE group_state SET last_seen_at = ? WHERE group_title = ?",
                (ts, norm_group),
            )
            return inserted

    def get_recent_context(self, group_title: str, *, limit: int = 8) -> list[ContextTurn]:
        """Return up to `limit` recent INCOMING/SELF turns in chronological order."""
        if limit <= 0:
            return []
        norm_group = canonicalize_chat_title(group_title)
        rows = self._conn.execute(
            """
            SELECT role, sender, text, first_seen_at
            FROM (
                SELECT id, role, sender, text, first_seen_at
                FROM seen_messages
                WHERE group_title = ? AND role IN ('incoming', 'self')
                ORDER BY id DESC
                LIMIT ?
            )
            ORDER BY id ASC
            """,
            (norm_group, int(limit)),
        ).fetchall()
        return [
            ContextTurn(
                role=MessageRole(row["role"]),
                sender=row["sender"],
                text=row["text"],
                captured_at=float(row["first_seen_at"]),
            )
            for row in rows
        ]

    def has_reply_record(self, fingerprint: str) -> bool:
        row = self._conn.execute(
            "SELECT 1 FROM reply_history WHERE fingerprint = ?",
            (fingerprint,),
        ).fetchone()
        return row is not None

    def record_reply_decision(
        self,
        *,
        fingerprint: str,
        group_title: str,
        mode: str,
        status: str,
        intent: str = "",
        confidence: float = 0.0,
        risk_score: float = 0.0,
        matched_rule: str | None = None,
        reply_text: str | None = None,
        reason: str = "",
        run_id: str = "",
        now: float | None = None,
    ) -> bool:
        """Acquire a persistent per-fingerprint decision/send record (`INSERT OR IGNORE`)."""
        ts = time.time() if now is None else float(now)
        norm_group = canonicalize_chat_title(group_title)
        with self._conn:
            cur = self._conn.execute(
                """
                INSERT OR IGNORE INTO reply_history
                    (fingerprint, group_title, mode, status, intent, confidence, risk_score,
                     matched_rule, reply_text, reason, run_id, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    fingerprint,
                    norm_group,
                    mode,
                    status,
                    intent,
                    float(confidence),
                    float(risk_score),
                    matched_rule,
                    reply_text,
                    reason,
                    run_id,
                    ts,
                ),
            )
            return cur.rowcount > 0

    def update_reply_status(
        self,
        fingerprint: str,
        *,
        status: str,
        run_id: str = "",
        reason: str | None = None,
    ) -> None:
        with self._conn:
            if reason is None:
                self._conn.execute(
                    "UPDATE reply_history SET status = ?, run_id = ? WHERE fingerprint = ?",
                    (status, run_id, fingerprint),
                )
            else:
                self._conn.execute(
                    "UPDATE reply_history SET status = ?, run_id = ?, reason = ? WHERE fingerprint = ?",
                    (status, run_id, reason, fingerprint),
                )

    def claim_bot_reply_match(self, group_title: str, text: str, *, since_ts: float) -> bool:
        """Consume an unacknowledged `sent_confirmed` bot reply matching `text` so it isn't flagged as human takeover."""
        canon = re.sub(r"\s+", "", normalize_text(text))
        if not canon:
            return False
        norm_group = canonicalize_chat_title(group_title)
        rows = self._conn.execute(
            """
            SELECT id, reply_text FROM reply_history
            WHERE group_title = ? AND status = 'sent_confirmed' AND created_at >= ?
            ORDER BY id ASC
            """,
            (norm_group, float(since_ts)),
        ).fetchall()
        for row in rows:
            row_canon = re.sub(r"\s+", "", normalize_text(str(row["reply_text"] or "")))
            if row_canon and (row_canon == canon or (len(row_canon) >= 2 and row_canon in canon)):
                with self._conn:
                    self._conn.execute(
                        "UPDATE reply_history SET status = 'sent_confirmed_seen' WHERE id = ?",
                        (int(row["id"]),),
                    )
                return True
        return False

    def record_send_event(self, group_title: str, status: str, now: float | None = None) -> None:
        ts = time.time() if now is None else float(now)
        norm_group = canonicalize_chat_title(group_title)
        with self._conn:
            self._ensure_group_row(norm_group)
            self._conn.execute(
                "INSERT INTO send_events (group_title, sent_at, status) VALUES (?, ?, ?)",
                (norm_group, ts, status),
            )
            self._conn.execute(
                "UPDATE group_state SET last_reply_at = ? WHERE group_title = ?",
                (ts, norm_group),
            )

    def recent_send_timestamps(self, since: float, *, group_title: str | None = None) -> list[float]:
        if group_title is None:
            rows = self._conn.execute(
                "SELECT sent_at FROM send_events WHERE sent_at >= ? ORDER BY sent_at ASC",
                (float(since),),
            ).fetchall()
        else:
            norm_group = canonicalize_chat_title(group_title)
            rows = self._conn.execute(
                "SELECT sent_at FROM send_events WHERE sent_at >= ? AND group_title = ? ORDER BY sent_at ASC",
                (float(since), norm_group),
            ).fetchall()
        return [float(r["sent_at"]) for r in rows]

    def set_human_takeover(self, group_title: str, until_ts: float) -> None:
        with self._conn:
            norm = self._ensure_group_row(group_title)
            self._conn.execute(
                "UPDATE group_state SET takeover_until = MAX(takeover_until, ?) WHERE group_title = ?",
                (float(until_ts), norm),
            )

    def get_takeover_until(self, group_title: str) -> float:
        norm = canonicalize_chat_title(group_title)
        row = self._conn.execute(
            "SELECT takeover_until FROM group_state WHERE group_title = ?",
            (norm,),
        ).fetchone()
        return float(row["takeover_until"]) if row else 0.0

    def append_audit(self, event: dict[str, Any]) -> None:
        if self.audit_log_path is None:
            return
        _ensure_private_parent(self.audit_log_path)
        record = {
            "ts": datetime.now().isoformat(timespec="milliseconds"),
            **redact_dict(event),
        }
        line = json.dumps(record, ensure_ascii=False) + "\n"
        fd = os.open(str(self.audit_log_path), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        with os.fdopen(fd, "a", encoding="utf-8") as handle:
            handle.write(line)
        os.chmod(self.audit_log_path, 0o600)
