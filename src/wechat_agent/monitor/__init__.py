"""M2 WeChat group monitoring and two-stage hybrid auto-reply framework."""

from .config import GroupPolicyConfig, MonitorConfig, load_monitor_config
from .extractor import (
    ChatMessageExtractor,
    ViewportAlignment,
    ViewportItem,
    align_viewport_suffix,
    build_viewport_keys,
    classify_bubble_role,
    compute_event_fingerprint,
    compute_message_fingerprint,
    mentions_any_alias,
    merge_same_sender_bursts,
    message_viewport_key,
    normalize_group_title,
)
from .llm_client import LlmClient, LlmClientError, LlmConfig, OpenAICompatibleLlmClient
from .models import (
    ContextTurn,
    DecisionContext,
    DecisionResult,
    ExtractedMessage,
    GeneratedReply,
    MessageRole,
    MonitorMode,
    TickOutcome,
    TickReport,
)
from .runner import MonitorRunner
from .store import MonitorInstanceLockError, MonitorStore

__all__ = [
    "ChatMessageExtractor",
    "ContextTurn",
    "DecisionContext",
    "DecisionResult",
    "ExtractedMessage",
    "GeneratedReply",
    "GroupPolicyConfig",
    "LlmClient",
    "LlmClientError",
    "LlmConfig",
    "MessageRole",
    "MonitorConfig",
    "MonitorInstanceLockError",
    "MonitorMode",
    "MonitorRunner",
    "MonitorStore",
    "OpenAICompatibleLlmClient",
    "TickOutcome",
    "TickReport",
    "ViewportAlignment",
    "ViewportItem",
    "align_viewport_suffix",
    "build_viewport_keys",
    "classify_bubble_role",
    "compute_event_fingerprint",
    "compute_message_fingerprint",
    "load_monitor_config",
    "mentions_any_alias",
    "merge_same_sender_bursts",
    "message_viewport_key",
    "normalize_group_title",
]
