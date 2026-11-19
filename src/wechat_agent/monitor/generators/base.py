"""Base protocol for Layer 2 ReplyGenerator."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from wechat_agent.monitor.models import DecisionContext, DecisionResult, GeneratedReply


@runtime_checkable
class ReplyGenerator(Protocol):
    def generate(self, ctx: DecisionContext, decision: DecisionResult) -> GeneratedReply | None: ...
