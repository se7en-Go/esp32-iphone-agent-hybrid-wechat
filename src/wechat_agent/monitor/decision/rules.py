"""Deterministic YAML keyword/regex and @mention DecisionGate."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from wechat_agent.monitor.models import DecisionContext, DecisionResult
from wechat_agent.vision.geometry import canonicalize_chat_title, normalize_text


@dataclass(frozen=True)
class RuleSpec:
    rule_id: str
    enabled: bool = True
    priority: int = 100
    groups: tuple[str, ...] = ()
    require_mention: bool = False
    keywords: tuple[str, ...] = ()
    regex: str | None = None
    exclude_keywords: tuple[str, ...] = ()
    intent: str = "rule_match"
    confidence: float = 0.95
    risk_score: float = 0.05
    reply_templates: tuple[str, ...] = ()

    def applies_to_group(self, group_title: str) -> bool:
        if not self.groups:
            return True
        norm_target = canonicalize_chat_title(group_title)
        return any(canonicalize_chat_title(g) == norm_target for g in self.groups)


def load_rules_yaml(path: str | Path) -> list[RuleSpec]:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Rules YAML file not found: {p}")
    raw = yaml.safe_load(p.read_text(encoding="utf-8"))
    if raw is None:
        return []
    if not isinstance(raw, dict):
        raise ValueError(f"Rules file {p} must be a YAML mapping containing 'rules'")

    items = raw.get("rules") or []
    if not isinstance(items, list):
        raise ValueError("'rules' must be a list in rules YAML")

    parsed: list[RuleSpec] = []
    for idx, entry in enumerate(items):
        if not isinstance(entry, dict):
            raise ValueError(f"Rule entry #{idx} must be a mapping")
        rule_id = str(entry.get("id") or f"rule_{idx}").strip()
        groups = tuple(str(g).strip() for g in (entry.get("groups") or []) if str(g).strip())
        keywords = tuple(str(k).strip() for k in (entry.get("keywords") or []) if str(k).strip())
        excludes = tuple(str(k).strip() for k in (entry.get("exclude_keywords") or []) if str(k).strip())
        templates = tuple(str(t).strip() for t in (entry.get("reply_templates") or []) if str(t).strip())
        regex_val = str(entry["regex"]).strip() if entry.get("regex") else None
        if regex_val:
            re.compile(regex_val)
        parsed.append(
            RuleSpec(
                rule_id=rule_id,
                enabled=bool(entry.get("enabled", True)),
                priority=int(entry.get("priority", 100)),
                groups=groups,
                require_mention=bool(entry.get("require_mention", False)),
                keywords=keywords,
                regex=regex_val,
                exclude_keywords=excludes,
                intent=str(entry.get("intent") or rule_id),
                confidence=float(entry.get("confidence", 0.95)),
                risk_score=float(entry.get("risk_score", 0.05)),
                reply_templates=templates,
            )
        )
    parsed.sort(key=lambda r: -r.priority)
    return parsed


class RuleDecisionGate:
    """Evaluates deterministic YAML keyword/regex rules and optional @mention trigger."""

    def __init__(
        self,
        rules: list[RuleSpec],
        *,
        enable_mention_trigger: bool = True,
        mention_confidence: float = 0.90,
    ) -> None:
        self.rules = sorted((r for r in rules if r.enabled), key=lambda r: -r.priority)
        self.enable_mention_trigger = enable_mention_trigger
        self.mention_confidence = float(mention_confidence)
        self._compiled_regex: dict[str, re.Pattern[str]] = {
            r.rule_id: re.compile(r.regex, re.IGNORECASE)
            for r in self.rules
            if r.regex
        }

    def match_rule(self, ctx: DecisionContext) -> DecisionResult | None:
        text_norm = normalize_text(ctx.message.text)
        text_lower = text_norm.lower()
        for rule in self.rules:
            if not rule.applies_to_group(ctx.group_title):
                continue
            if rule.require_mention and not ctx.message.mentioned_self:
                continue
            if any(ex.lower() in text_lower for ex in rule.exclude_keywords):
                continue
            kw_hit = any(kw.lower() in text_lower for kw in rule.keywords) if rule.keywords else False
            re_pat = self._compiled_regex.get(rule.rule_id)
            re_hit = bool(re_pat.search(text_norm)) if re_pat is not None else False
            if kw_hit or re_hit:
                return DecisionResult(
                    should_reply=True,
                    intent=rule.intent,
                    confidence=rule.confidence,
                    risk_score=rule.risk_score,
                    matched_rule=rule.rule_id,
                    reason=f"Matched deterministic YAML rule {rule.rule_id!r}",
                    gate_name="rule",
                )
        return None

    def evaluate_mention(self, ctx: DecisionContext) -> DecisionResult | None:
        if self.enable_mention_trigger and ctx.message.mentioned_self:
            return DecisionResult(
                should_reply=True,
                intent="direct_mention",
                confidence=self.mention_confidence,
                risk_score=0.05,
                matched_rule=None,
                reason="Message directly mentioned @self",
                gate_name="rule_mention",
            )
        return None

    def evaluate(self, ctx: DecisionContext) -> DecisionResult:
        matched = self.match_rule(ctx)
        if matched is not None:
            return matched
        mention = self.evaluate_mention(ctx)
        if mention is not None:
            return mention
        return DecisionResult(
            should_reply=False,
            intent="no_rule_match",
            confidence=0.0,
            risk_score=0.0,
            matched_rule=None,
            reason="No YAML rule or @mention matched",
            gate_name="rule",
        )
