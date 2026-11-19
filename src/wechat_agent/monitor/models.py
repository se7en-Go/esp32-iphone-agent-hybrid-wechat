"""M2 domain models for group monitoring, two-stage decision/generation, and tick reporting."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from wechat_agent.models import Box


class MessageRole(str, Enum):
    INCOMING = "incoming"
    SELF = "self"
    SYSTEM = "system"


class MonitorMode(str, Enum):
    OBSERVE_ONLY = "observe-only"
    DRY_RUN = "dry-run"
    LIVE = "live"


@dataclass(frozen=True)
class ExtractedMessage:
    group_title: str
    role: MessageRole
    text: str
    sender: str = ""
    box: Box | None = None
    mentioned_self: bool = False
    fingerprint: str = ""
    captured_at: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "group_title": self.group_title,
            "role": self.role.value,
            "sender": self.sender,
            "text": self.text,
            "mentioned_self": self.mentioned_self,
            "fingerprint": self.fingerprint,
            "captured_at": self.captured_at,
        }


@dataclass(frozen=True)
class ContextTurn:
    role: MessageRole
    sender: str
    text: str
    captured_at: float = 0.0

    def format_line(self) -> str:
        if self.role == MessageRole.SELF:
            return f"[我]: {self.text}"
        who = self.sender.strip() or "群友"
        return f"[对方/{who}]: {self.text}"


@dataclass(frozen=True)
class DecisionContext:
    group_title: str
    message: ExtractedMessage
    recent_context: tuple[ContextTurn, ...] = ()
    self_aliases: tuple[str, ...] = ()
    role_prompt: str = ""


@dataclass(frozen=True)
class DecisionResult:
    should_reply: bool
    intent: str
    confidence: float
    risk_score: float
    matched_rule: str | None = None
    reason: str = ""
    gate_name: str = ""
    raw: dict[str, Any] = field(default_factory=dict)

    def passes_thresholds(self, min_confidence: float, max_risk: float) -> bool:
        return (
            self.should_reply
            and self.confidence >= min_confidence
            and self.risk_score <= max_risk
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "should_reply": self.should_reply,
            "intent": self.intent,
            "confidence": round(self.confidence, 4),
            "risk_score": round(self.risk_score, 4),
            "matched_rule": self.matched_rule,
            "reason": self.reason,
            "gate_name": self.gate_name,
        }


@dataclass(frozen=True)
class GeneratedReply:
    text: str
    generator_name: str
    rule_id: str | None = None
    reason: str = ""
    raw: dict[str, Any] = field(default_factory=dict)


class TickOutcome(str, Enum):
    BASELINE_SEEDED = "baseline_seeded"
    NO_NEW_MESSAGES = "no_new_messages"
    OBSERVED = "observed"
    DRY_RUN_LOGGED = "dry_run_logged"
    DRY_RUN_PASTED = "dry_run_pasted"
    REPLIED = "replied"
    SKIPPED_COOLDOWN = "skipped_cooldown"
    SKIPPED_QUIET_HOURS = "skipped_quiet_hours"
    SKIPPED_HUMAN_TAKEOVER = "skipped_human_takeover"
    SKIPPED_FORBIDDEN_REPLY = "skipped_forbidden_reply"
    SKIPPED_EXISTING_DRAFT = "skipped_existing_draft"
    PAUSED_LEFT_GROUP = "paused_left_group"
    PAUSED_LOCK_SCREEN = "paused_lock_screen"
    PAUSED_NON_WECHAT = "paused_non_wechat"
    HALTED_RISK_PHRASE = "halted_risk_phrase"
    HALTED_SEND_ERROR = "halted_send_error"
    HALTED_PREFLIGHT = "halted_preflight"


@dataclass
class TickReport:
    tick_index: int
    group_title: str
    mode: MonitorMode
    outcome: TickOutcome
    should_halt: bool = False
    new_incoming_count: int = 0
    human_takeover_detected: bool = False
    decision: dict[str, Any] | None = None
    reply_text: str | None = None
    message: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "tick_index": self.tick_index,
            "group_title": self.group_title,
            "mode": self.mode.value,
            "outcome": self.outcome.value,
            "should_halt": self.should_halt,
            "new_incoming_count": self.new_incoming_count,
            "human_takeover_detected": self.human_takeover_detected,
            "decision": self.decision,
            "reply_text": self.reply_text,
            "message": self.message,
        }
