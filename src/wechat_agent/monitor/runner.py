"""M2 resident WeChat group monitoring runner with baseline seeding, takeover detection, and safe mode dispatch."""

from __future__ import annotations

import random
import time
from collections.abc import Callable
from dataclasses import replace

from wechat_agent.adapters.driver import DriverError, PhoneDriver
from wechat_agent.config import WeChatSendConfig
from wechat_agent.guards.rate_limit import MonitorRateLimiter
from wechat_agent.guards.skill_guard import SkillGuard
from wechat_agent.models import SendReport, SendStatus
from wechat_agent.vision.ocr import OcrEngine
from wechat_agent.vision.wechat_ui import (
    ScreenKind,
    detect_existing_draft,
    detect_keyboard_and_focused_input,
    detect_screen_context,
    has_exact_chat_title,
    is_lock_screen,
)
from wechat_agent.workflows.send_message import SendMessageWorkflow

from .config import GroupPolicyConfig, MonitorConfig
from .decision.base import DecisionGate, ObserveOnlyGate
from .decision.jev_gate import to_jev_semantics
from .extractor import (
    ChatMessageExtractor,
    align_viewport_suffix,
    compute_event_fingerprint,
    message_viewport_key,
    normalize_group_title,
)
from .generators.base import ReplyGenerator
from .models import (
    DecisionContext,
    ExtractedMessage,
    MessageRole,
    MonitorMode,
    TickOutcome,
    TickReport,
)
from .store import MonitorStore


class MonitorRunner:
    """Single-group resident monitor loop connecting extraction, SQLite dedup, 2-stage AI, and M1 Send."""

    def __init__(
        self,
        driver: PhoneDriver,
        ocr: OcrEngine,
        config: MonitorConfig,
        store: MonitorStore,
        gate: DecisionGate,
        generator: ReplyGenerator | None = None,
        *,
        rate_limiter: MonitorRateLimiter | None = None,
        extractor: ChatMessageExtractor | None = None,
        sender_factory: Callable[[GroupPolicyConfig], SendMessageWorkflow] | None = None,
        clock: Callable[[], float] = time.time,
        sleeper: Callable[[float], None] = time.sleep,
        rng: random.Random | None = None,
    ) -> None:
        self.driver = driver
        self.ocr = ocr
        self.config = config
        self.store = store
        self.gate = gate
        self.generator = generator
        self._clock = clock
        self._sleeper = sleeper
        self._rng = rng or random.Random()
        self.rate_limiter = rate_limiter or MonitorRateLimiter(store, config.rate_limits, clock=clock)
        self.extractor = extractor or ChatMessageExtractor()
        self.skill_guard = SkillGuard(driver, skill_name=config.skill_name)
        self._sender_factory = sender_factory or self._default_sender_factory

    def _default_sender_factory(self, group_cfg: GroupPolicyConfig) -> SendMessageWorkflow:
        send_cfg = WeChatSendConfig(
            iphone_agent_cli=self.config.iphone_agent_cli,
            iphone_agent_url=self.config.iphone_agent_url,
            skill_name=self.config.skill_name,
            allowed_targets=(group_cfg.name,),
            ocr_backend=self.config.ocr_backend,
            evidence_root=self.config.evidence_root,
            cli_timeout_seconds=self.config.cli_timeout_seconds,
            paste_flow_timeout_seconds=self.config.paste_flow_timeout_seconds,
            require_mounted_skill=self.config.require_mounted_skill,
            allow_group_count=True,
        )
        return SendMessageWorkflow(self.driver, self.ocr, send_cfg)

    def preflight(self) -> str | None:
        group_cfg = self.config.active_group
        if group_cfg is None or not normalize_group_title(group_cfg.name):
            return "No target group configured in monitor target_groups"

        if self.config.mode == MonitorMode.LIVE:
            if not self.config.allow_live_send:
                return "Live mode requires explicit allow_live_send (--allow-live-send)"
            if isinstance(self.gate, ObserveOnlyGate):
                return "Live mode cannot run with ObserveOnlyGate"

        try:
            st = self.driver.status()
            caps = self.driver.capabilities()
            skills = self.driver.skill_list()
        except DriverError as exc:
            return f"Base server unreachable during monitor preflight: {exc}"

        if not (st.get("device") and st.get("screen") and st.get("calibrated") and st.get("ready")):
            return "Base device is not ready for monitoring"
        guard_state = st.get("guard") or {}
        if guard_state.get("tripped"):
            return f"Base guard is tripped: {guard_state.get('reason', '')}"

        if self.config.mode == MonitorMode.LIVE:
            clip = (caps.get("features") or {}).get("clipboard") or {}
            if not clip.get("staged_available") or clip.get("protocol") != "staged-v1":
                return "Live mode requires clipboard staged-v1 protocol"

        if self.config.require_mounted_skill and self.config.skill_name not in skills:
            return f"Skill {self.config.skill_name!r} is not mounted on base server (mounted: {skills})"

        return None

    def _finish_tick(self, report: TickReport, extra_audit: dict | None = None) -> TickReport:
        payload = report.to_dict()
        if extra_audit:
            payload.update(extra_audit)
        self.store.append_audit(payload)
        return report

    def _assign_event_message(self, group_name: str, msg: ExtractedMessage) -> ExtractedMessage:
        seq = self.store.next_event_seq(group_name)
        fp = compute_event_fingerprint(group_name, seq, msg.role, msg.sender, msg.text)
        return replace(msg, fingerprint=fp)

    def run_tick(self, tick_index: int = 0) -> TickReport:
        group_cfg = self.config.active_group
        if group_cfg is None:
            return self._finish_tick(
                TickReport(
                    tick_index=tick_index,
                    group_title="",
                    mode=self.config.mode,
                    outcome=TickOutcome.HALTED_PREFLIGHT,
                    should_halt=True,
                    message="No target group configured",
                )
            )
        group_name = normalize_group_title(group_cfg.name)
        now = self._clock()

        def tick_report(outcome: TickOutcome, message: str, **kwargs) -> TickReport:
            return self._finish_tick(
                TickReport(
                    tick_index=tick_index,
                    group_title=group_name,
                    mode=self.config.mode,
                    outcome=outcome,
                    message=message,
                    **kwargs,
                )
            )

        # 1. Observe + OCR + circuit-breaker check
        try:
            obs = self.driver.observe()
            ocr_boxes = self.ocr.read_boxes(obs.image_bytes)
            check = self.skill_guard.check_screen_text(ocr_boxes)
        except DriverError as exc:
            return tick_report(
                TickOutcome.HALTED_PREFLIGHT,
                f"Monitor observe/skill-check failed: {exc}",
                should_halt=True,
            )

        if check.stop:
            return tick_report(
                TickOutcome.HALTED_RISK_PHRASE,
                f"Circuit breaker triggered on screen: {check.reason}",
                should_halt=True,
            )

        # 2. Long-running screen guard: pause safely on Lock Screen, non-WeChat, or leaving target group
        if is_lock_screen(ocr_boxes, obs.image_bytes):
            return tick_report(
                TickOutcome.PAUSED_LOCK_SCREEN,
                "iPhone screen is locked or asleep; pausing safely without touching device",
            )

        if not has_exact_chat_title(ocr_boxes, group_name, allow_group_count=True):
            kind = detect_screen_context(ocr_boxes, group_name, allow_group_count=True)
            if kind in {ScreenKind.WECHAT_CHAT_LIST, ScreenKind.OTHER_CHAT}:
                return tick_report(
                    TickOutcome.PAUSED_LEFT_GROUP,
                    f"Screen is no longer in target group {group_name!r} (context={kind.value}); pausing safely",
                )
            return tick_report(
                TickOutcome.PAUSED_NON_WECHAT,
                f"Screen is outside target WeChat conversation (context={kind.value}); pausing safely",
            )

        # 3. Extract visible bubbles (excluding clipped header bubbles & grey system lines)
        focused = detect_keyboard_and_focused_input(obs.image_bytes, ocr_boxes)
        bottom_limit = (focused.input_box.top - 0.01) if focused is not None else 0.875
        aliases = group_cfg.self_aliases or self.config.self_aliases
        visible = self.extractor.extract(
            obs.image_bytes,
            ocr_boxes,
            group_title=group_name,
            self_aliases=aliases,
            bottom_limit=bottom_limit,
            captured_at=now,
        )

        curr_keys = [message_viewport_key(m) for m in visible]

        # 4. Baseline seeding on first entry into this group: mark existing history seen, never reply to old messages
        if not self.store.is_baseline_seeded(group_name):
            for msg in visible:
                self.store.record_seen(
                    self._assign_event_message(group_name, msg),
                    is_baseline=True,
                    now=now,
                )
            self.store.save_last_viewport(group_name, curr_keys, now=now)
            self.store.mark_baseline_seeded(group_name, now=now)
            return tick_report(
                TickOutcome.BASELINE_SEEDED,
                f"Seeded baseline with {len(visible)} visible message(s)",
            )

        # 5. Suffix-overlap viewport alignment against previous persisted viewport sequence
        prev_keys = self.store.get_last_viewport(group_name)
        alignment = align_viewport_suffix(prev_keys, curr_keys)

        if alignment.kind == "empty_curr":
            return tick_report(
                TickOutcome.NO_NEW_MESSAGES,
                "No visible messages in viewport",
            )

        if alignment.kind == "scroll_up":
            return tick_report(
                TickOutcome.NO_NEW_MESSAGES,
                "Detected scroll-up or partial viewport; keeping previous viewport anchor",
            )

        if alignment.kind == "discontinuity":
            since_ts = now - 3600.0
            for msg in visible:
                if not self.store.has_recent_message_content(
                    group_name,
                    msg.role,
                    msg.sender,
                    msg.text,
                    since_ts=since_ts,
                ):
                    self.store.record_seen(
                        self._assign_event_message(group_name, msg),
                        is_baseline=True,
                        now=now,
                    )
            self.store.save_last_viewport(group_name, curr_keys, now=now)
            return tick_report(
                TickOutcome.BASELINE_SEEDED,
                f"Viewport discontinuity (no overlap with previous screen); safely re-anchored baseline with {len(visible)} visible message(s)",
            )

        newly_arrived = visible[alignment.new_start:]
        if not newly_arrived:
            return tick_report(
                TickOutcome.NO_NEW_MESSAGES,
                "No new incoming messages",
            )

        # Persist newly arrived events with monotonic independent event fingerprints and advance anchor
        arrived_events: list[ExtractedMessage] = []
        for msg in newly_arrived:
            evt_msg = self._assign_event_message(group_name, msg)
            self.store.record_seen(evt_msg, is_baseline=False, now=now)
            arrived_events.append(evt_msg)
        self.store.save_last_viewport(group_name, curr_keys, now=now)

        # 6. Process newly arrived SELF bubbles (takeover detection) and collect INCOMING messages
        takeover_triggered = False
        new_incoming: list[ExtractedMessage] = []
        for evt_msg in arrived_events:
            if evt_msg.role == MessageRole.SELF:
                if not self.store.claim_bot_reply_match(group_name, evt_msg.text, since_ts=now - 300.0):
                    takeover_triggered = True
                    self.rate_limiter.activate_human_takeover(group_name, now=now)
            elif evt_msg.role == MessageRole.INCOMING:
                new_incoming.append(evt_msg)

        if not new_incoming:
            if takeover_triggered:
                return tick_report(
                    TickOutcome.SKIPPED_HUMAN_TAKEOVER,
                    "Detected manual human reply in group; human takeover cooldown activated",
                    human_takeover_detected=True,
                )
            return tick_report(
                TickOutcome.NO_NEW_MESSAGES,
                "No new incoming messages",
            )

        # 7-9. Prioritize @self mentions first (in arrival order) so multi-message ticks never swallow @self
        prioritized = [
            *(m for m in new_incoming if m.mentioned_self),
            *(m for m in new_incoming if not m.mentioned_self),
        ]
        recent_turns = tuple(self.store.get_recent_context(group_name, limit=self.config.context_window))
        eff_role_prompt = (
            group_cfg.role_prompt
            if group_cfg.role_prompt is not None
            else self.config.role_prompt
        )

        selected: tuple[ExtractedMessage, object, dict, str] | None = None
        fallback_info: tuple[dict, str | None, str, bool] | None = None

        for candidate in prioritized:
            ctx = DecisionContext(
                group_title=group_name,
                message=candidate,
                recent_context=recent_turns,
                self_aliases=aliases,
                role_prompt=eff_role_prompt,
            )

            decision = self.gate.evaluate(ctx)
            decision_dict = {**decision.to_dict(), "jev": to_jev_semantics(decision)}
            if candidate.mentioned_self:
                eff_min_conf = (
                    group_cfg.mention_min_confidence
                    if group_cfg.mention_min_confidence is not None
                    else self.config.mention_min_confidence
                )
            else:
                eff_min_conf = (
                    group_cfg.min_confidence
                    if group_cfg.min_confidence is not None
                    else self.config.min_confidence
                )
            eff_max_risk = (
                group_cfg.max_risk_score
                if group_cfg.max_risk_score is not None
                else self.config.max_risk_score
            )
            thresholds_ok = decision.passes_thresholds(eff_min_conf, eff_max_risk)

            candidate_reply = None
            if thresholds_ok and self.generator is not None:
                candidate_reply = self.generator.generate(ctx, decision)

            cand_text = candidate_reply.text if candidate_reply is not None else None
            if cand_text is not None:
                safe_ok, safe_reason = self.rate_limiter.check_reply_content(cand_text)
                if not safe_ok:
                    self.store.record_reply_decision(
                        fingerprint=candidate.fingerprint,
                        group_title=group_name,
                        mode=self.config.mode.value,
                        status=TickOutcome.SKIPPED_FORBIDDEN_REPLY.value,
                        intent=decision.intent,
                        confidence=decision.confidence,
                        risk_score=decision.risk_score,
                        matched_rule=decision.matched_rule,
                        reply_text=None,
                        reason=safe_reason,
                        now=now,
                    )
                    return tick_report(
                        TickOutcome.SKIPPED_FORBIDDEN_REPLY,
                        f"Blocked unsafe candidate reply: {safe_reason}",
                        new_incoming_count=len(new_incoming),
                        human_takeover_detected=takeover_triggered,
                        decision=decision_dict,
                    )

            if self.config.mode == MonitorMode.OBSERVE_ONLY:
                self.store.record_reply_decision(
                    fingerprint=candidate.fingerprint,
                    group_title=group_name,
                    mode=self.config.mode.value,
                    status=TickOutcome.OBSERVED.value,
                    intent=decision.intent,
                    confidence=decision.confidence,
                    risk_score=decision.risk_score,
                    matched_rule=decision.matched_rule,
                    reply_text=cand_text,
                    reason=decision.reason,
                    now=now,
                )
                if thresholds_ok:
                    return tick_report(
                        TickOutcome.OBSERVED,
                        f"Observed {len(new_incoming)} new message(s) (thresholds_ok=True)",
                        new_incoming_count=len(new_incoming),
                        human_takeover_detected=takeover_triggered,
                        decision=decision_dict,
                        reply_text=cand_text,
                    )
                if fallback_info is None or (decision.should_reply and not fallback_info[3]):
                    fallback_info = (
                        decision_dict,
                        cand_text,
                        f"Observed {len(new_incoming)} new message(s) (thresholds_ok=False)",
                        decision.should_reply,
                    )
                continue

            if thresholds_ok and cand_text is not None:
                selected = (candidate, decision, decision_dict, cand_text)
                break

            skip_reason = (
                decision.reason
                if not decision.should_reply
                else (
                    f"Blocked by threshold (conf={decision.confidence:.2f} vs {eff_min_conf:.2f}, "
                    f"risk={decision.risk_score:.2f} vs {eff_max_risk:.2f})"
                    if not thresholds_ok
                    else "Generator produced no reply"
                )
            )
            self.store.record_reply_decision(
                fingerprint=candidate.fingerprint,
                group_title=group_name,
                mode=self.config.mode.value,
                status="skipped_decision",
                intent=decision.intent,
                confidence=decision.confidence,
                risk_score=decision.risk_score,
                matched_rule=decision.matched_rule,
                reply_text=None,
                reason=skip_reason,
                now=now,
            )
            if fallback_info is None or (decision.should_reply and not fallback_info[3]):
                fallback_info = (decision_dict, None, skip_reason, decision.should_reply)

        if selected is None:
            fb_decision, fb_reply, fb_msg, _ = fallback_info or (
                None,
                None,
                f"Observed {len(new_incoming)} new message(s) (thresholds_ok=False)",
                False,
            )
            return tick_report(
                TickOutcome.OBSERVED,
                fb_msg,
                new_incoming_count=len(new_incoming),
                human_takeover_detected=takeover_triggered,
                decision=fb_decision,
                reply_text=fb_reply,
            )

        target_msg, decision, decision_dict, candidate_text = selected

        # 10. Rate-limit & takeover cooldown check before dry-run paste or live send
        rl = self.rate_limiter.check_can_reply(
            group_name,
            now=now,
            per_group_cooldown_override=group_cfg.per_group_cooldown_seconds,
            per_group_max_per_hour_override=group_cfg.per_group_max_per_hour,
        )
        if not rl.allowed:
            outcome = (
                TickOutcome.SKIPPED_QUIET_HOURS
                if rl.block_kind == "quiet_hours"
                else (
                    TickOutcome.SKIPPED_HUMAN_TAKEOVER
                    if rl.block_kind == "human_takeover"
                    else TickOutcome.SKIPPED_COOLDOWN
                )
            )
            self.store.record_reply_decision(
                fingerprint=target_msg.fingerprint,
                group_title=group_name,
                mode=self.config.mode.value,
                status=outcome.value,
                intent=decision.intent,
                confidence=decision.confidence,
                risk_score=decision.risk_score,
                matched_rule=decision.matched_rule,
                reply_text=candidate_text,
                reason=rl.reason,
                now=now,
            )
            return tick_report(
                outcome,
                rl.reason,
                new_incoming_count=len(new_incoming),
                human_takeover_detected=takeover_triggered,
                decision=decision_dict,
                reply_text=candidate_text,
            )

        # 11. DRY_RUN red line: NEVER touch input box or paste drafts on real groups!
        is_sim_paste_allowed = (
            group_name in {normalize_group_title(t) for t in self.config.simulation_targets}
            and group_cfg.allow_dry_run_paste
        )
        if self.config.mode == MonitorMode.DRY_RUN and not is_sim_paste_allowed:
            self.store.record_reply_decision(
                fingerprint=target_msg.fingerprint,
                group_title=group_name,
                mode=self.config.mode.value,
                status=TickOutcome.DRY_RUN_LOGGED.value,
                intent=decision.intent,
                confidence=decision.confidence,
                risk_score=decision.risk_score,
                matched_rule=decision.matched_rule,
                reply_text=candidate_text,
                reason="Dry-run logged only (no input focus/paste on real groups)",
                now=now,
            )
            return tick_report(
                TickOutcome.DRY_RUN_LOGGED,
                "Dry-run logged decision and candidate reply (zero device input on real group)",
                new_incoming_count=len(new_incoming),
                human_takeover_detected=takeover_triggered,
                decision=decision_dict,
                reply_text=candidate_text,
            )

        # 12. Check for open existing human draft before invoking SendMessageWorkflow
        if focused is not None:
            draft = detect_existing_draft(ocr_boxes, focused.input_box)
            if draft is not None:
                return tick_report(
                    TickOutcome.SKIPPED_EXISTING_DRAFT,
                    f"Existing human draft in input box ({draft!r}); preserving draft and skipping reply",
                    new_incoming_count=len(new_incoming),
                    human_takeover_detected=takeover_triggered,
                    decision=decision_dict,
                    reply_text=candidate_text,
                )

        # 13. Acquire idempotent SQLite lock for this fingerprint (prevents duplicate reply across crashes/restarts)
        acquired = self.store.record_reply_decision(
            fingerprint=target_msg.fingerprint,
            group_title=group_name,
            mode=self.config.mode.value,
            status="pending_send" if self.config.mode == MonitorMode.LIVE else "pending_dry_paste",
            intent=decision.intent,
            confidence=decision.confidence,
            risk_score=decision.risk_score,
            matched_rule=decision.matched_rule,
            reply_text=candidate_text,
            reason=decision.reason,
            now=now,
        )
        if not acquired:
            return tick_report(
                TickOutcome.NO_NEW_MESSAGES,
                "Message fingerprint already holds a reply record (idempotent skip)",
                new_incoming_count=len(new_incoming),
                human_takeover_detected=takeover_triggered,
                decision=decision_dict,
            )

        # 14. Execute M1 SendMessageWorkflow (send=False for authorized sim dry-run paste, send=True for live)
        sender = self._sender_factory(group_cfg)
        want_send = self.config.mode == MonitorMode.LIVE
        send_report: SendReport = sender.run(group_name, candidate_text, send=want_send)
        self.store.update_reply_status(
            target_msg.fingerprint,
            status=send_report.status.value,
            run_id=send_report.run_id,
            reason=send_report.message,
        )

        if send_report.status == SendStatus.DRY_RUN_READY:
            return tick_report(
                TickOutcome.DRY_RUN_PASTED,
                send_report.message,
                new_incoming_count=len(new_incoming),
                human_takeover_detected=takeover_triggered,
                decision=decision_dict,
                reply_text=candidate_text,
            )

        if send_report.status == SendStatus.SENT_CONFIRMED:
            self.store.record_send_event(group_name, send_report.status.value, now=now)
            return tick_report(
                TickOutcome.REPLIED,
                send_report.message,
                new_incoming_count=len(new_incoming),
                human_takeover_detected=takeover_triggered,
                decision=decision_dict,
                reply_text=candidate_text,
            )

        if send_report.status == SendStatus.ABORTED_EXISTING_DRAFT:
            return tick_report(
                TickOutcome.SKIPPED_EXISTING_DRAFT,
                send_report.message,
                new_incoming_count=len(new_incoming),
                human_takeover_detected=takeover_triggered,
                decision=decision_dict,
                reply_text=candidate_text,
            )

        if send_report.status == SendStatus.ABORTED_RISK_PHRASE:
            return tick_report(
                TickOutcome.HALTED_RISK_PHRASE,
                send_report.message,
                should_halt=True,
                new_incoming_count=len(new_incoming),
                human_takeover_detected=takeover_triggered,
                decision=decision_dict,
                reply_text=candidate_text,
            )

        if send_report.status in {SendStatus.ABORTED_PASTE_RECOVERY, SendStatus.SEND_UNCONFIRMED}:
            return tick_report(
                TickOutcome.HALTED_SEND_ERROR,
                send_report.message,
                should_halt=True,
                new_incoming_count=len(new_incoming),
                human_takeover_detected=takeover_triggered,
                decision=decision_dict,
                reply_text=candidate_text,
            )

        return tick_report(
            TickOutcome.HALTED_SEND_ERROR if want_send else TickOutcome.DRY_RUN_LOGGED,
            send_report.message,
            should_halt=want_send and send_report.status != SendStatus.ABORTED_RATE_LIMIT,
            new_incoming_count=len(new_incoming),
            human_takeover_detected=takeover_triggered,
            decision=decision_dict,
            reply_text=candidate_text,
        )

    def run_loop(self, *, max_ticks: int | None = None) -> list[TickReport]:
        reports: list[TickReport] = []
        preflight_err = self.preflight()
        if preflight_err is not None:
            group_name = normalize_group_title(self.config.active_group.name) if self.config.active_group else ""
            halted = self._finish_tick(
                TickReport(
                    tick_index=0,
                    group_title=group_name,
                    mode=self.config.mode,
                    outcome=TickOutcome.HALTED_PREFLIGHT,
                    should_halt=True,
                    message=preflight_err,
                )
            )
            reports.append(halted)
            return reports

        tick = 0
        while max_ticks is None or tick < max_ticks:
            rep = self.run_tick(tick_index=tick)
            reports.append(rep)
            if rep.should_halt:
                break
            tick += 1
            if max_ticks is not None and tick >= max_ticks:
                break
            delay = self._rng.uniform(
                self.config.poll_interval_min_seconds,
                self.config.poll_interval_max_seconds,
            )
            self._sleeper(delay)
        return reports
