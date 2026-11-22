"""Cross-restart SQLite WAL persistence and 0600 JSONL audit logging for M2 monitoring."""

from __future__ import annotations

import errno
import hashlib
import json
import os
import re
import sqlite3
import time
from collections.abc import Sequence
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    import fcntl
except ImportError:  # pragma: no cover - non-POSIX fallback
    fcntl = None  # type: ignore[assignment]

from wechat_agent.evidence import redact_dict
from wechat_agent.vision.geometry import canonicalize_chat_title, normalize_text

from .extractor import (
    LegacyViewportItem,
    ViewportItem,
    _canonical_for_hash,
    _coerce_viewport_items,
    compute_event_fingerprint,
    merge_same_sender_bursts,
)
from .models import ContextTurn, ExtractedMessage, MessageRole


class MonitorInstanceLockError(RuntimeError):
    """Raised when another monitor process already holds the single-instance lock for this state DB."""


def _ensure_private_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    os.chmod(path.parent, 0o700)


def _content_hash(sender: str, text: str) -> str:
    payload = f"{normalize_text(sender)}\x1f{_canonical_for_hash(text)}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


class MonitorStore:
    """Persistent state store for deduplication, idempotent reply locking, and rate-limit history."""

    def __init__(
        self,
        db_path: Path,
        audit_log_path: Path | None = None,
        *,
        acquire_lock: bool = True,
        audit_max_bytes: int = 10 * 1024 * 1024,
        audit_backup_count: int = 5,
    ) -> None:
        self.db_path = Path(db_path)
        self.lock_path = self.db_path.with_name(f"{self.db_path.name}.lock")
        self.audit_log_path = Path(audit_log_path) if audit_log_path is not None else None
        self.audit_max_bytes = max(1024, int(audit_max_bytes))
        self.audit_backup_count = max(1, int(audit_backup_count))
        self._lock_fd: int | None = None
        self._closed = False
        _ensure_private_parent(self.db_path)
        if self.audit_log_path is not None:
            _ensure_private_parent(self.audit_log_path)

        if acquire_lock:
            self._acquire_instance_lock()

        try:
            self._conn = sqlite3.connect(str(self.db_path), timeout=10.0)
            self._conn.row_factory = sqlite3.Row
            os.chmod(self.db_path, 0o600)
            self._init_schema()
        except Exception:
            self._release_instance_lock()
            raise

    def _acquire_instance_lock(self) -> None:
        fd = os.open(str(self.lock_path), os.O_RDWR | os.O_CREAT, 0o600)
        os.chmod(self.lock_path, 0o600)
        if fcntl is not None:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                os.close(fd)
                if exc.errno in (errno.EACCES, errno.EAGAIN, errno.EWOULDBLOCK) or isinstance(
                    exc, BlockingIOError
                ):
                    raise MonitorInstanceLockError(
                        f"Another monitor instance already holds lock {self.lock_path} (db={self.db_path})"
                    ) from exc
                raise
        try:
            os.ftruncate(fd, 0)
            os.lseek(fd, 0, os.SEEK_SET)
            os.write(fd, f"{os.getpid()}\n".encode("ascii"))
        except OSError:
            pass
        self._lock_fd = fd

    def _release_instance_lock(self) -> None:
        fd = self._lock_fd
        self._lock_fd = None
        if fd is None:
            return
        try:
            if fcntl is not None:
                fcntl.flock(fd, fcntl.LOCK_UN)
        except OSError:
            pass
        try:
            os.close(fd)
        except OSError:
            pass

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
                    content_hash TEXT NOT NULL DEFAULT '',
                    is_baseline INTEGER NOT NULL DEFAULT 0,
                    first_seen_at REAL NOT NULL,
                    last_seen_at REAL NOT NULL
                )
                """
            )
            seen_cols = {
                str(r["name"])
                for r in self._conn.execute("PRAGMA table_info(seen_messages)").fetchall()
            }
            if "content_hash" not in seen_cols:
                self._conn.execute(
                    "ALTER TABLE seen_messages ADD COLUMN content_hash TEXT NOT NULL DEFAULT ''"
                )
            missing_hashes = self._conn.execute(
                "SELECT id, sender, text FROM seen_messages WHERE content_hash = ''"
            ).fetchall()
            if missing_hashes:
                self._conn.executemany(
                    "UPDATE seen_messages SET content_hash = ? WHERE id = ?",
                    [
                        (_content_hash(str(r["sender"] or ""), str(r["text"] or "")), int(r["id"]))
                        for r in missing_hashes
                    ],
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
                """
                CREATE TABLE IF NOT EXISTS pending_queue (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    fingerprint TEXT UNIQUE NOT NULL,
                    group_title TEXT NOT NULL,
                    sender TEXT NOT NULL,
                    text TEXT NOT NULL,
                    mentioned_self INTEGER NOT NULL DEFAULT 0,
                    status TEXT NOT NULL DEFAULT 'pending',
                    created_at REAL NOT NULL,
                    expires_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                )
                """
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_seen_group_time ON seen_messages(group_title, id)"
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_seen_group_role_hash_time "
                "ON seen_messages(group_title, role, content_hash, last_seen_at)"
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_send_events_time ON send_events(sent_at, group_title)"
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_pending_group_status "
                "ON pending_queue(group_title, status, mentioned_self DESC, id ASC)"
            )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._conn.close()
        finally:
            self._release_instance_lock()

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
        raw_items: list[ViewportItem | LegacyViewportItem] = []
        for item in parsed:
            if not isinstance(item, (list, tuple)):
                continue
            if len(item) == 4:
                try:
                    occ = max(1, int(item[3]))
                except (TypeError, ValueError):
                    occ = 1
                raw_items.append((str(item[0]), str(item[1]), str(item[2]), occ))
            elif len(item) == 3:
                raw_items.append((str(item[0]), str(item[1]), str(item[2])))
        return _coerce_viewport_items(raw_items)

    def _save_last_viewport_in_tx(
        self,
        norm_group: str,
        items: Sequence[ViewportItem | LegacyViewportItem],
        *,
        ts: float,
        mark_baseline: bool = False,
    ) -> None:
        normalized = _coerce_viewport_items(items)
        payload = json.dumps([[r, s, t, occ] for r, s, t, occ in normalized], ensure_ascii=False)
        self._ensure_group_row(norm_group)
        if mark_baseline:
            self._conn.execute(
                "UPDATE group_state SET baseline_seeded = 1, last_viewport_json = ?, last_seen_at = ? "
                "WHERE group_title = ?",
                (payload, ts, norm_group),
            )
        else:
            self._conn.execute(
                "UPDATE group_state SET last_viewport_json = ?, last_seen_at = ? WHERE group_title = ?",
                (payload, ts, norm_group),
            )

    def save_last_viewport(
        self,
        group_title: str,
        items: Sequence[ViewportItem],
        now: float | None = None,
    ) -> None:
        """Persist the ordered visible message sequence for suffix-overlap alignment."""
        ts = time.time() if now is None else float(now)
        norm = canonicalize_chat_title(group_title)
        with self._conn:
            self._save_last_viewport_in_tx(norm, items, ts=ts, mark_baseline=False)

    def _next_event_seq_in_tx(self, norm_group: str) -> int:
        self._ensure_group_row(norm_group)
        self._conn.execute(
            "UPDATE group_state SET event_seq = event_seq + 1 WHERE group_title = ?",
            (norm_group,),
        )
        row = self._conn.execute(
            "SELECT event_seq FROM group_state WHERE group_title = ?",
            (norm_group,),
        ).fetchone()
        return int(row["event_seq"]) if row else 1

    def next_event_seq(self, group_title: str) -> int:
        """Atomically allocate the next monotonic event sequence number for `group_title`."""
        norm = canonicalize_chat_title(group_title)
        with self._conn:
            return self._next_event_seq_in_tx(norm)

    def has_recent_message_content(
        self,
        group_title: str,
        role: MessageRole,
        sender: str,
        text: str,
        *,
        since_ts: float,
    ) -> bool:
        """Check whether `(role, sender, text)` was recently recorded using the SQL content_hash index."""
        norm_group = canonicalize_chat_title(group_title)
        chash = _content_hash(sender, text)
        row = self._conn.execute(
            """
            SELECT 1 FROM seen_messages
            WHERE group_title = ? AND role = ? AND content_hash = ? AND last_seen_at >= ?
            LIMIT 1
            """,
            (norm_group, role.value, chash, float(since_ts)),
        ).fetchone()
        return row is not None

    def has_seen(self, fingerprint: str) -> bool:
        row = self._conn.execute(
            "SELECT 1 FROM seen_messages WHERE fingerprint = ?",
            (fingerprint,),
        ).fetchone()
        return row is not None

    def _record_seen_in_tx(
        self,
        msg: ExtractedMessage,
        *,
        is_baseline: bool,
        ts: float,
    ) -> bool:
        norm_group = canonicalize_chat_title(msg.group_title)
        self._ensure_group_row(norm_group)
        chash = _content_hash(msg.sender, msg.text)
        cur = self._conn.execute(
            """
            INSERT OR IGNORE INTO seen_messages
                (fingerprint, group_title, role, sender, text, content_hash, is_baseline, first_seen_at, last_seen_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                msg.fingerprint,
                norm_group,
                msg.role.value,
                msg.sender,
                msg.text,
                chash,
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

    def record_seen(
        self,
        msg: ExtractedMessage,
        *,
        is_baseline: bool = False,
        now: float | None = None,
    ) -> bool:
        """Record a message fingerprint. Returns True if newly inserted, False if already seen."""
        ts = time.time() if now is None else float(now)
        with self._conn:
            return self._record_seen_in_tx(msg, is_baseline=is_baseline, ts=ts)

    def _assign_event_in_tx(self, norm_group: str, msg: ExtractedMessage) -> ExtractedMessage:
        seq = self._next_event_seq_in_tx(norm_group)
        fp = compute_event_fingerprint(norm_group, seq, msg.role, msg.sender, msg.text)
        return replace(msg, group_title=norm_group, fingerprint=fp)

    def commit_baseline_batch(
        self,
        group_title: str,
        messages: Sequence[ExtractedMessage],
        viewport_items: Sequence[ViewportItem],
        *,
        mark_seeded: bool = True,
        dedup_since_ts: float | None = None,
        now: float | None = None,
    ) -> list[ExtractedMessage]:
        """Atomically persist baseline messages, viewport anchor, and seeded flag in a single transaction."""
        ts = time.time() if now is None else float(now)
        norm_group = canonicalize_chat_title(group_title)
        persisted: list[ExtractedMessage] = []
        with self._conn:
            for msg in messages:
                if dedup_since_ts is not None and self.has_recent_message_content(
                    norm_group,
                    msg.role,
                    msg.sender,
                    msg.text,
                    since_ts=dedup_since_ts,
                ):
                    continue
                evt = self._assign_event_in_tx(norm_group, msg)
                self._record_seen_in_tx(evt, is_baseline=True, ts=ts)
                persisted.append(evt)
            self._save_last_viewport_in_tx(
                norm_group,
                viewport_items,
                ts=ts,
                mark_baseline=mark_seeded,
            )
        return persisted

    def commit_arrived_batch(
        self,
        group_title: str,
        arrived_messages: Sequence[ExtractedMessage],
        viewport_items: Sequence[ViewportItem],
        *,
        skipped_baseline_messages: Sequence[ExtractedMessage] = (),
        burst_merge_window_seconds: float = 5.0,
        pending_ttl_seconds: float = 180.0,
        now: float | None = None,
    ) -> tuple[list[ExtractedMessage], list[ExtractedMessage]]:
        """Atomically allocate event_seq, persist seen_messages, advance viewport, and enqueue merged incoming."""
        ts = time.time() if now is None else float(now)
        norm_group = canonicalize_chat_title(group_title)
        arrived_events: list[ExtractedMessage] = []
        merged_incoming: list[ExtractedMessage] = []
        with self._conn:
            for skipped in skipped_baseline_messages:
                evt_skip = self._assign_event_in_tx(norm_group, skipped)
                self._record_seen_in_tx(evt_skip, is_baseline=True, ts=ts)
            for msg in arrived_messages:
                evt = self._assign_event_in_tx(norm_group, msg)
                self._record_seen_in_tx(evt, is_baseline=False, ts=ts)
                arrived_events.append(evt)
            self._save_last_viewport_in_tx(norm_group, viewport_items, ts=ts, mark_baseline=False)

            raw_incoming = [evt for evt in arrived_events if evt.role == MessageRole.INCOMING]
            merged_incoming = merge_same_sender_bursts(
                raw_incoming,
                max_gap_seconds=burst_merge_window_seconds,
            )
            for inc in merged_incoming:
                self._enqueue_pending_in_tx(
                    inc,
                    ttl_seconds=pending_ttl_seconds,
                    burst_merge_window_seconds=burst_merge_window_seconds,
                    ts=ts,
                )
        return arrived_events, merged_incoming

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
        allow_reclaim_statuses: Sequence[str] = (),
    ) -> bool:
        """Acquire a persistent per-fingerprint decision/send record.

        Normally uses `INSERT OR IGNORE` so each fingerprint can only be processed once.
        Only when `allow_reclaim_statuses` explicitly names definitely-unsent statuses
        (e.g. `aborted_existing_draft`) may an existing row in one of those statuses be
        atomically reclaimed; potentially-sent statuses are never reclaimed.
        """
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
            if cur.rowcount > 0:
                return True
            reclaim = tuple(str(s) for s in allow_reclaim_statuses if str(s))
            if not reclaim:
                return False
            placeholders = ",".join("?" for _ in reclaim)
            upd = self._conn.execute(
                f"""
                UPDATE reply_history
                SET group_title = ?, mode = ?, status = ?, intent = ?, confidence = ?,
                    risk_score = ?, matched_rule = ?, reply_text = ?, reason = ?,
                    run_id = ?, created_at = ?
                WHERE fingerprint = ? AND status IN ({placeholders})
                """,
                (
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
                    fingerprint,
                    *reclaim,
                ),
            )
            return upd.rowcount > 0

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
        """Consume an unacknowledged `sent_confirmed` bot reply matching `text` to avoid false takeover.

        Requires exact normalized equality, or — for long replies (>= 10 canonical chars) — a
        high-ratio (>= 85%) containment match to tolerate minor single-edge OCR truncation without
        ever allowing a human message containing a short bot reply (e.g. "好的") to swallow the match.
        """
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
            if not row_canon:
                continue
            matched = row_canon == canon
            if not matched and len(row_canon) >= 10 and len(canon) >= 10:
                ratio = min(len(row_canon), len(canon)) / max(len(row_canon), len(canon))
                if ratio >= 0.85 and (row_canon in canon or canon in row_canon):
                    matched = True
            if matched:
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

    def get_last_seen_at(self, group_title: str) -> float:
        norm = canonicalize_chat_title(group_title)
        row = self._conn.execute(
            "SELECT last_seen_at FROM group_state WHERE group_title = ?",
            (norm,),
        ).fetchone()
        return float(row["last_seen_at"]) if row else 0.0

    def touch_last_seen(self, group_title: str, now: float | None = None) -> None:
        ts = time.time() if now is None else float(now)
        with self._conn:
            norm = self._ensure_group_row(group_title)
            self._conn.execute(
                "UPDATE group_state SET last_seen_at = ? WHERE group_title = ?",
                (ts, norm),
            )

    def _expire_pending_locked(self, norm_group: str, ts: float) -> int:
        cur = self._conn.execute(
            """
            UPDATE pending_queue
            SET status = 'expired', updated_at = ?
            WHERE group_title = ? AND status = 'pending' AND expires_at <= ?
            """,
            (ts, norm_group, ts),
        )
        return int(cur.rowcount or 0)

    def expire_pending(self, group_title: str, *, now: float | None = None) -> int:
        ts = time.time() if now is None else float(now)
        norm_group = canonicalize_chat_title(group_title)
        with self._conn:
            return self._expire_pending_locked(norm_group, ts)

    def _enqueue_pending_in_tx(
        self,
        msg: ExtractedMessage,
        *,
        ttl_seconds: float,
        burst_merge_window_seconds: float,
        ts: float,
    ) -> str:
        ttl = max(1.0, float(ttl_seconds))
        norm_group = canonicalize_chat_title(msg.group_title)
        norm_sender = normalize_text(msg.sender)
        self._expire_pending_locked(norm_group, ts)
        if norm_sender:
            latest = self._conn.execute(
                """
                SELECT id, fingerprint, sender, text, mentioned_self, updated_at
                FROM pending_queue
                WHERE group_title = ? AND status = 'pending'
                ORDER BY id DESC
                LIMIT 1
                """,
                (norm_group,),
            ).fetchone()
            if (
                latest is not None
                and normalize_text(str(latest["sender"] or "")) == norm_sender
                and 0.0 <= (ts - float(latest["updated_at"])) <= burst_merge_window_seconds
            ):
                prev_text = str(latest["text"] or "")
                merged_text = (
                    f"{prev_text}\n{msg.text}" if prev_text and msg.text else (prev_text or msg.text)
                )
                merged_mention = 1 if (latest["mentioned_self"] or msg.mentioned_self) else 0
                self._conn.execute(
                    """
                    UPDATE pending_queue
                    SET text = ?, mentioned_self = ?, expires_at = ?, updated_at = ?
                    WHERE id = ?
                    """,
                    (merged_text, merged_mention, ts + ttl, ts, int(latest["id"])),
                )
                return str(latest["fingerprint"])

        self._conn.execute(
            """
            INSERT OR IGNORE INTO pending_queue
                (fingerprint, group_title, sender, text, mentioned_self, status, created_at, expires_at, updated_at)
            VALUES (?, ?, ?, ?, ?, 'pending', ?, ?, ?)
            """,
            (
                msg.fingerprint,
                norm_group,
                msg.sender,
                msg.text,
                1 if msg.mentioned_self else 0,
                ts,
                ts + ttl,
                ts,
            ),
        )
        return msg.fingerprint

    def enqueue_pending(
        self,
        msg: ExtractedMessage,
        *,
        ttl_seconds: float = 180.0,
        burst_merge_window_seconds: float = 5.0,
        now: float | None = None,
    ) -> str:
        """Enqueue an incoming message, merging rapid adjacent messages from the same non-empty sender."""
        ts = time.time() if now is None else float(now)
        with self._conn:
            return self._enqueue_pending_in_tx(
                msg,
                ttl_seconds=ttl_seconds,
                burst_merge_window_seconds=burst_merge_window_seconds,
                ts=ts,
            )

    def list_pending(self, group_title: str, *, now: float | None = None) -> list[ExtractedMessage]:
        """Expire stale items and return unexpired pending messages (@self prioritized, then FIFO)."""
        ts = time.time() if now is None else float(now)
        norm_group = canonicalize_chat_title(group_title)
        with self._conn:
            self._expire_pending_locked(norm_group, ts)
        rows = self._conn.execute(
            """
            SELECT fingerprint, group_title, sender, text, mentioned_self, created_at
            FROM pending_queue
            WHERE group_title = ? AND status = 'pending'
            ORDER BY mentioned_self DESC, id ASC
            """,
            (norm_group,),
        ).fetchall()
        return [
            ExtractedMessage(
                group_title=str(r["group_title"]),
                role=MessageRole.INCOMING,
                text=str(r["text"]),
                sender=str(r["sender"]),
                mentioned_self=bool(r["mentioned_self"]),
                fingerprint=str(r["fingerprint"]),
                captured_at=float(r["created_at"]),
            )
            for r in rows
        ]

    def has_pending(self, group_title: str, *, now: float | None = None) -> bool:
        """Expire stale items and return True if any unexpired pending message remains."""
        ts = time.time() if now is None else float(now)
        norm_group = canonicalize_chat_title(group_title)
        with self._conn:
            self._expire_pending_locked(norm_group, ts)
        row = self._conn.execute(
            "SELECT 1 FROM pending_queue WHERE group_title = ? AND status = 'pending' LIMIT 1",
            (norm_group,),
        ).fetchone()
        return row is not None

    def mark_pending_status(
        self,
        fingerprint: str,
        status: str,
        *,
        now: float | None = None,
    ) -> None:
        ts = time.time() if now is None else float(now)
        with self._conn:
            self._conn.execute(
                "UPDATE pending_queue SET status = ?, updated_at = ? WHERE fingerprint = ?",
                (status, ts, fingerprint),
            )

    def prune_storage(
        self,
        *,
        now: float | None = None,
        seen_ttl_seconds: float = 14 * 86400.0,
        reply_ttl_seconds: float = 30 * 86400.0,
        send_events_ttl_seconds: float = 30 * 86400.0,
        pending_ttl_seconds: float = 7 * 86400.0,
        max_rows_per_table: int = 50_000,
    ) -> dict[str, int]:
        """Enforce TTL and bounded row-count retention across all 4 growing SQLite tables."""
        ts = time.time() if now is None else float(now)
        cap = max(1, int(max_rows_per_table))
        deleted = {"seen_messages": 0, "reply_history": 0, "send_events": 0, "pending_queue": 0}
        with self._conn:
            c1 = self._conn.execute(
                "DELETE FROM seen_messages WHERE last_seen_at < ?",
                (ts - max(0.0, float(seen_ttl_seconds)),),
            )
            deleted["seen_messages"] += int(c1.rowcount or 0)
            c1_cap = self._conn.execute(
                """
                DELETE FROM seen_messages
                WHERE id NOT IN (SELECT id FROM seen_messages ORDER BY id DESC LIMIT ?)
                """,
                (cap,),
            )
            deleted["seen_messages"] += int(c1_cap.rowcount or 0)

            c2 = self._conn.execute(
                "DELETE FROM reply_history WHERE created_at < ?",
                (ts - max(0.0, float(reply_ttl_seconds)),),
            )
            deleted["reply_history"] += int(c2.rowcount or 0)
            c2_cap = self._conn.execute(
                """
                DELETE FROM reply_history
                WHERE id NOT IN (SELECT id FROM reply_history ORDER BY id DESC LIMIT ?)
                """,
                (cap,),
            )
            deleted["reply_history"] += int(c2_cap.rowcount or 0)

            c3 = self._conn.execute(
                "DELETE FROM send_events WHERE sent_at < ?",
                (ts - max(0.0, float(send_events_ttl_seconds)),),
            )
            deleted["send_events"] += int(c3.rowcount or 0)
            c3_cap = self._conn.execute(
                """
                DELETE FROM send_events
                WHERE id NOT IN (SELECT id FROM send_events ORDER BY id DESC LIMIT ?)
                """,
                (cap,),
            )
            deleted["send_events"] += int(c3_cap.rowcount or 0)

            # Never delete active 'pending' queue items; prune only settled/expired rows.
            c4 = self._conn.execute(
                "DELETE FROM pending_queue WHERE status != 'pending' AND updated_at < ?",
                (ts - max(0.0, float(pending_ttl_seconds)),),
            )
            deleted["pending_queue"] += int(c4.rowcount or 0)
            c4_cap = self._conn.execute(
                """
                DELETE FROM pending_queue
                WHERE status != 'pending'
                  AND id NOT IN (
                      SELECT id FROM pending_queue WHERE status != 'pending' ORDER BY id DESC LIMIT ?
                  )
                """,
                (cap,),
            )
            deleted["pending_queue"] += int(c4_cap.rowcount or 0)
        return deleted

    def _rotate_audit_if_needed(self) -> None:
        if self.audit_log_path is None or not self.audit_log_path.is_file():
            return
        try:
            size = self.audit_log_path.stat().st_size
        except OSError:
            return
        if size < self.audit_max_bytes:
            return
        base = self.audit_log_path
        for idx in range(self.audit_backup_count, 0, -1):
            src = base if idx == 1 else base.with_name(f"{base.name}.{idx - 1}")
            dst = base.with_name(f"{base.name}.{idx}")
            if idx == self.audit_backup_count and dst.exists():
                try:
                    dst.unlink()
                except OSError:
                    pass
            if src.exists():
                os.replace(src, dst)
                try:
                    os.chmod(dst, 0o600)
                except OSError:
                    pass

    def append_audit(self, event: dict[str, Any]) -> None:
        if self.audit_log_path is None:
            return
        _ensure_private_parent(self.audit_log_path)
        self._rotate_audit_if_needed()
        record = {
            "ts": datetime.now().isoformat(timespec="milliseconds"),
            **redact_dict(event),
        }
        line = json.dumps(record, ensure_ascii=False) + "\n"
        fd = os.open(str(self.audit_log_path), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        with os.fdopen(fd, "a", encoding="utf-8") as handle:
            handle.write(line)
        os.chmod(self.audit_log_path, 0o600)
