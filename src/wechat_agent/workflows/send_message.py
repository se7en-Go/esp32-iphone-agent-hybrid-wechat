"""M1 end-to-end WeChat message sending workflow with safety gates, draft protection, and guarded send."""

from __future__ import annotations

import time

from wechat_agent.adapters.driver import DriverError, PhoneDriver
from wechat_agent.config import WeChatSendConfig
from wechat_agent.evidence import RunEvidence, redact_dict
from wechat_agent.guards.skill_guard import SkillGuard
from wechat_agent.guards.target_policy import PolicyViolationError, TargetPolicy
from wechat_agent.models import (
    Box,
    Observation,
    SendReport,
    SendStatus,
)
from wechat_agent.vision.geometry import left_biased_input_aim
from wechat_agent.vision.ocr import OcrEngine
from wechat_agent.vision.wechat_ui import (
    detect_existing_draft,
    detect_keyboard_and_focused_input,
    extract_header_title,
    extract_outgoing_bubbles,
    has_exact_chat_title,
    has_new_outgoing_bubble,
    locate_blue_send_button,
    locate_collapsed_input_box,
    verify_pasted_text,
)

from .click_chain import ClickChain, PointerConflictError, PointerNotConfirmedError
from .navigation import NavigationError, WeChatNavigator


class SendMessageWorkflow:
    def __init__(
        self,
        driver: PhoneDriver,
        ocr: OcrEngine,
        config: WeChatSendConfig,
    ) -> None:
        self.driver = driver
        self.ocr = ocr
        self.config = config
        self.target_policy = TargetPolicy(config.allowed_targets)
        self.skill_guard = SkillGuard(driver, skill_name=config.skill_name)

    def _finish(self, evidence: RunEvidence, report: SendReport) -> SendReport:
        evidence.write_json("report.json", report.to_dict(), redact=True)
        evidence.log_event("workflow_finished", {"status": report.status.value, "message": report.message})
        return report

    def _preflight(self, steps: list[str]) -> str | None:
        try:
            st = self.driver.status()
            caps = self.driver.capabilities()
            skills = self.driver.skill_list()
        except DriverError as exc:
            return f"Base server unreachable or preflight failed: {exc}"

        if not (st.get("device") and st.get("screen") and st.get("calibrated") and st.get("ready")):
            return (
                f"Device not ready (device={st.get('device')}, screen={st.get('screen')}, "
                f"calibrated={st.get('calibrated')}, ready={st.get('ready')})"
            )
        guard_state = st.get("guard") or {}
        if guard_state.get("tripped"):
            return f"Base server guard is tripped: {guard_state.get('reason', '')}"

        clipboard = (caps.get("features") or {}).get("clipboard") or {}
        if not clipboard.get("staged_available") or clipboard.get("protocol") != "staged-v1":
            return (
                f"Clipboard staged-v1 not available (staged_available={clipboard.get('staged_available')}, "
                f"protocol={clipboard.get('protocol')})"
            )

        if self.config.require_mounted_skill and self.config.skill_name not in skills:
            return (
                f"Skill {self.config.skill_name!r} is not mounted on base server (mounted: {skills}). "
                f"Set HIDAGENT_SKILLS={self.config.skills_dir()} and restart the base worker."
            )

        steps.append("preflight_ok")
        return None

    def run(self, target: str, text: str, *, send: bool = False) -> SendReport:
        dry_run = not send
        evidence = RunEvidence(self.config.resolved_evidence_root())
        steps: list[str] = []

        def report(status: SendStatus, message: str, **kwargs) -> SendReport:
            return self._finish(
                evidence,
                SendReport(
                    status=status,
                    target=target,
                    text=text,
                    dry_run=dry_run,
                    message=message,
                    run_id=evidence.run_id,
                    evidence_dir=str(evidence.dir),
                    steps=steps,
                    **kwargs,
                ),
            )

        # 1. Target allowlist and text validation
        try:
            clean_target = self.target_policy.assert_allowed(target)
        except PolicyViolationError as exc:
            return report(SendStatus.ABORTED_POLICY, str(exc))

        try:
            clean_text = self.skill_guard.validate_message_text(text)
        except ValueError as exc:
            return report(SendStatus.ABORTED_POLICY, str(exc))
        steps.append("policy_ok")

        # 2. Base preflight readiness & mounted skill check
        preflight_err = self._preflight(steps)
        if preflight_err is not None:
            return report(SendStatus.ABORTED_PREFLIGHT, preflight_err)

        click_chain = ClickChain(
            self.driver,
            evidence,
            rehome_retries=self.config.point_rehome_retries,
        )
        navigator = WeChatNavigator(
            self.driver,
            self.ocr,
            click_chain,
            evidence,
            allow_group_count=self.config.allow_group_count,
        )

        # 3. Initial observe + risk phrase check
        try:
            obs = self.driver.observe(image_output=evidence.image_output_path("01_initial"))
            ocr_boxes = self.ocr.read_boxes(obs.image_bytes)
            check = self.skill_guard.check_screen_text(ocr_boxes)
        except DriverError as exc:
            return report(SendStatus.ABORTED_PREFLIGHT, f"Initial observe/skill-check failed: {exc}")

        if check.stop:
            return report(SendStatus.ABORTED_RISK_PHRASE, f"Circuit breaker triggered: {check.reason}")

        # 4. Navigate to target WeChat conversation
        try:
            obs, ocr_boxes = navigator.ensure_target_chat(clean_target, obs, ocr_boxes)
            steps.append("navigation_ok")
        except PointerNotConfirmedError as exc:
            return report(SendStatus.ABORTED_POINTER, str(exc))
        except (NavigationError, DriverError) as exc:
            return report(SendStatus.ABORTED_NAVIGATION, str(exc))

        # Re-check risk phrases on target conversation screen
        try:
            chat_check = self.skill_guard.check_screen_text(ocr_boxes)
        except DriverError as exc:
            return report(SendStatus.ABORTED_PREFLIGHT, f"Chat skill-check failed: {exc}")
        if chat_check.stop:
            return report(SendStatus.ABORTED_RISK_PHRASE, f"Circuit breaker triggered: {chat_check.reason}")

        # 5. Focus input box (if keyboard not open yet) and capture baseline outgoing bubbles
        focused = detect_keyboard_and_focused_input(obs.image_bytes, ocr_boxes)
        baseline_area = Box(0.0, 0.13, 1.0, focused.input_box.top - 0.01) if focused else Box(0.0, 0.13, 1.0, 0.86)
        baseline_bubbles = extract_outgoing_bubbles(obs.image_bytes, ocr_boxes, chat_area=baseline_area)

        if focused is None:
            click_res = None
            for attempt in range(3):
                collapsed_box = locate_collapsed_input_box(obs.image_bytes, ocr_boxes)
                aim_xy = left_biased_input_aim(collapsed_box)
                evidence.log_event(
                    "focus_collapsed_input",
                    {"attempt": attempt, "collapsed_box": collapsed_box.as_list(), "aim_xy": list(aim_xy)},
                )
                try:
                    click_res = click_chain.point_and_click(
                        obs.screen_id,
                        aim_xy,
                        collapsed_box,
                        step_name=f"focus_input_{attempt}" if attempt else "focus_input",
                    )
                    break
                except PointerConflictError as exc:
                    evidence.log_event(
                        "focus_pre_click_409_reobserve",
                        {"attempt": attempt, "detail": exc.detail},
                    )
                    if attempt == 2:
                        return report(SendStatus.ABORTED_NAVIGATION, f"Focusing input field hit 409 repeatedly: {exc}")
                    time.sleep(0.25)
                    obs = self.driver.observe(image_output=evidence.image_output_path(f"02_focus_reobs_{attempt}"))
                    ocr_boxes = self.ocr.read_boxes(obs.image_bytes)
                    if not has_exact_chat_title(
                        ocr_boxes,
                        clean_target,
                        allow_group_count=self.config.allow_group_count,
                    ):
                        return report(
                            SendStatus.ABORTED_NAVIGATION,
                            f"Conversation title {clean_target!r} disappeared while focusing input",
                        )
                    focused = detect_keyboard_and_focused_input(obs.image_bytes, ocr_boxes)
                    if focused is not None:
                        break
                except PointerNotConfirmedError as exc:
                    return report(SendStatus.ABORTED_POINTER, str(exc))
                except DriverError as exc:
                    return report(SendStatus.ABORTED_NAVIGATION, f"Focusing input field failed: {exc}")

            if focused is None and click_res is not None:
                try:
                    if click_res.screen_id and click_res.image_bytes:
                        obs = Observation(
                            screen_id=click_res.screen_id,
                            image_bytes=click_res.image_bytes,
                            image_path=click_res.image_path,
                            raw=click_res.raw,
                        )
                    else:
                        obs = self.driver.observe(image_output=evidence.image_output_path("02_after_focus"))
                    ocr_boxes = self.ocr.read_boxes(obs.image_bytes)
                    focused = detect_keyboard_and_focused_input(obs.image_bytes, ocr_boxes)
                    if focused is None:
                        # Passive re-observe once (NO second click!) in case iOS keyboard animation settled slightly later.
                        time.sleep(0.25)
                        obs = self.driver.observe(image_output=evidence.image_output_path("02_after_focus_reobs"))
                        ocr_boxes = self.ocr.read_boxes(obs.image_bytes)
                        focused = detect_keyboard_and_focused_input(obs.image_bytes, ocr_boxes)
                except DriverError as exc:
                    return report(SendStatus.ABORTED_NAVIGATION, f"Post-focus observation failed: {exc}")

            if focused is None:
                return report(
                    SendStatus.ABORTED_NAVIGATION,
                    "Input field click did not open native keyboard and focused input box",
                )
        steps.append("input_focused")

        # 6. Existing draft protection: preserve untouched and halt immediately
        existing_draft = detect_existing_draft(ocr_boxes, focused.input_box)
        if existing_draft is not None:
            evidence.log_event("existing_draft_detected", {"draft_length": len(existing_draft)})
            return report(
                SendStatus.ABORTED_EXISTING_DRAFT,
                f"Existing unsent draft detected in input box ({existing_draft!r}); preserving draft and stopping.",
                observed_draft=existing_draft,
            )
        steps.append("draft_clean")

        # 7. Single paste-flow execution (NEVER retry on failure)
        expect_header_title = (
            (extract_header_title(ocr_boxes) or clean_target)
            if self.config.allow_group_count
            else clean_target
        )
        payload_path = evidence.write_payload(clean_text)
        paste_res = self.driver.paste_flow(
            payload_path,
            target_box=focused.input_box,
            expect_title=expect_header_title,
            caret_box=focused.caret_box,
            evidence_dir=evidence.paste_evidence_dir,
            image_output=evidence.image_output_path("03_paste_confirm"),
        )
        evidence.write_json("paste_receipt.json", paste_res.raw, redact=False)
        redacted_receipt = redact_dict(paste_res.raw)

        if not paste_res.is_strictly_completed:
            return report(
                SendStatus.ABORTED_PASTE_RECOVERY,
                f"paste-flow did not complete strictly (recovery_state={paste_res.recovery_state.value}): "
                f"{paste_res.error or paste_res.next_action}",
                recovery_state=paste_res.recovery_state,
                next_action=paste_res.next_action,
                paste_receipt=redacted_receipt,
            )
        steps.append("paste_flow_completed")

        # 8. Fresh observe + OCR verification of exact title, pasted text, and absence of premature bubble
        try:
            post_obs = self.driver.observe(image_output=evidence.image_output_path("04_post_paste"))
            post_ocr = self.ocr.read_boxes(post_obs.image_bytes)
            post_check = self.skill_guard.check_screen_text(post_ocr)
        except DriverError as exc:
            return report(
                SendStatus.ABORTED_PASTE_VERIFY,
                f"Post-paste observation failed: {exc}",
                paste_receipt=redacted_receipt,
            )

        if post_check.stop:
            return report(
                SendStatus.ABORTED_RISK_PHRASE,
                f"Circuit breaker triggered after paste: {post_check.reason}",
                paste_receipt=redacted_receipt,
            )

        if not has_exact_chat_title(
            post_ocr,
            clean_target,
            allow_group_count=self.config.allow_group_count,
        ):
            return report(
                SendStatus.ABORTED_PASTE_VERIFY,
                f"Conversation title verification failed after paste (expected {clean_target!r})",
                paste_receipt=redacted_receipt,
            )

        focused_post = detect_keyboard_and_focused_input(post_obs.image_bytes, post_ocr)
        active_input_box = focused.input_box
        if focused_post is not None:
            active_input_box = Box.clamped(
                min(focused.input_box.left, focused_post.input_box.left),
                min(focused.input_box.top, focused_post.input_box.top),
                max(focused.input_box.right, focused_post.input_box.right),
                max(focused.input_box.bottom, focused_post.input_box.bottom),
            )
        text_ok, observed_pasted = verify_pasted_text(post_ocr, active_input_box, clean_text)
        if not text_ok:
            return report(
                SendStatus.ABORTED_PASTE_VERIFY,
                f"Post-paste text mismatch or duplication (observed={observed_pasted!r}, expected={clean_text!r})",
                paste_receipt=redacted_receipt,
            )

        post_chat_area = Box(0.0, 0.13, 1.0, active_input_box.top - 0.01)
        pre_send_bubbles = extract_outgoing_bubbles(post_obs.image_bytes, post_ocr, chat_area=post_chat_area)
        if has_new_outgoing_bubble(baseline_bubbles, pre_send_bubbles, clean_text):
            return report(
                SendStatus.ABORTED_PASTE_VERIFY,
                "Unexpected outgoing bubble appeared during paste before Send was clicked",
                paste_receipt=redacted_receipt,
            )
        steps.append("post_paste_verified")

        # 9. Default --dry-run stops cleanly before Send
        if dry_run:
            return report(
                SendStatus.DRY_RUN_READY,
                "Dry-run completed: pasted text verified intact in input box; stopped before Send.",
                paste_receipt=redacted_receipt,
            )

        # 10. Explicit --send path: locate dynamic blue Send button avoiding Space bar
        send_loc = locate_blue_send_button(post_obs.image_bytes, post_ocr)
        if send_loc is None:
            return report(
                SendStatus.ABORTED_SEND_BUTTON,
                "Could not uniquely locate lower-right Send button on keyboard",
                paste_receipt=redacted_receipt,
            )
        send_box, send_label = send_loc
        evidence.log_event(
            "blue_send_button_located",
            {"label": send_label, "send_box": send_box.as_list()},
        )

        # 11. Reserve send slot right before pre-send pointer movement (30s TTL)
        try:
            prep = self.skill_guard.prepare_send()
        except DriverError as exc:
            return report(
                SendStatus.ABORTED_RATE_LIMIT,
                f"skill-prepare-send failed: {exc}",
                paste_receipt=redacted_receipt,
            )
        if prep.stop:
            return report(
                SendStatus.ABORTED_RISK_PHRASE,
                f"Circuit breaker blocked prepare-send: {prep.reason}",
                paste_receipt=redacted_receipt,
            )
        if not prep.allowed:
            return report(
                SendStatus.ABORTED_RATE_LIMIT,
                f"Send blocked by rate limit or quiet hours: {prep.reason}",
                paste_receipt=redacted_receipt,
            )
        steps.append("send_slot_reserved")

        # 12. Pre-send pointer move ONLY (allows HTTP 409 caret-blink retry, NEVER clicks)
        # Pass verified input box as explicit --caret-scope alongside server 120s carry-over.
        evidence.log_event("pre_send_caret_scope", {"caret_scope": active_input_box.as_list()})
        try:
            mp, _ = click_chain.pre_send_point_with_caret_retry(
                send_box.center,
                send_box,
                step_name="pre_send_point",
                max_attempts=self.config.pre_send_point_retries,
                initial_screen_id=post_obs.screen_id,
                caret_scope=active_input_box,
            )
        except PointerNotConfirmedError as exc:
            return report(
                SendStatus.ABORTED_POINTER,
                str(exc),
                paste_receipt=redacted_receipt,
            )
        except DriverError as exc:
            return report(
                SendStatus.ABORTED_POINTER,
                f"Pre-send pointer positioning failed: {exc}",
                paste_receipt=redacted_receipt,
            )
        steps.append("pre_send_pointer_confirmed")

        # 13. Single-shot Send click — NEVER retry click_observed under any circumstances
        try:
            send_click_res = click_chain.single_shot_send_click(
                mp,
                send_box,
                send_label,
                step_name="send_click",
                caret_scope=active_input_box,
            )
        except DriverError as exc:
            return report(
                SendStatus.SEND_UNCONFIRMED,
                f"Send click errored ({exc}); click will NOT be retried. Re-observe screen manually.",
                paste_receipt=redacted_receipt,
            )
        steps.append("send_click_dispatched")

        # 14. Post-send verification: must see a NEW complete green outgoing bubble before skill-confirm-send
        try:
            if send_click_res.image_bytes:
                final_bytes = send_click_res.image_bytes
            else:
                after_obs = self.driver.observe(image_output=evidence.image_output_path("05_post_send"))
                final_bytes = after_obs.image_bytes
            final_ocr = self.ocr.read_boxes(final_bytes)
            final_check = self.skill_guard.check_screen_text(final_ocr)
            if final_check.stop:
                return report(
                    SendStatus.ABORTED_RISK_PHRASE,
                    f"Circuit breaker triggered after send: {final_check.reason}",
                    paste_receipt=redacted_receipt,
                )

            final_bubbles = extract_outgoing_bubbles(
                final_bytes,
                final_ocr,
                chat_area=Box(0.0, 0.13, 1.0, 0.87),
            )
            bubble_ok = has_new_outgoing_bubble(baseline_bubbles, final_bubbles, clean_text)
            if not bubble_ok:
                # One passive re-observe in case mirror lag delayed bubble rendering; NEVER re-click Send.
                reobs = self.driver.observe(image_output=evidence.image_output_path("06_post_send_reobs"))
                final_ocr = self.ocr.read_boxes(reobs.image_bytes)
                recheck = self.skill_guard.check_screen_text(final_ocr)
                if recheck.stop:
                    return report(
                        SendStatus.ABORTED_RISK_PHRASE,
                        f"Circuit breaker triggered on post-send re-observe: {recheck.reason}",
                        paste_receipt=redacted_receipt,
                    )
                final_bubbles = extract_outgoing_bubbles(
                    reobs.image_bytes,
                    final_ocr,
                    chat_area=Box(0.0, 0.13, 1.0, 0.87),
                )
                bubble_ok = has_new_outgoing_bubble(baseline_bubbles, final_bubbles, clean_text)
        except DriverError as exc:
            return report(
                SendStatus.SEND_UNCONFIRMED,
                f"Post-send verification failed ({exc}); do NOT auto-retry Send.",
                paste_receipt=redacted_receipt,
            )

        if not bubble_ok:
            return report(
                SendStatus.SEND_UNCONFIRMED,
                "New outgoing green bubble was NOT confirmed on screen (input clearing or hardware ok does not prove delivery).",
                paste_receipt=redacted_receipt,
            )

        steps.append("new_bubble_visible")
        try:
            confirm = self.skill_guard.confirm_send(bubble_visible=True, ocr_boxes=final_ocr)
        except DriverError as exc:
            return report(
                SendStatus.SEND_UNCONFIRMED,
                f"Visible bubble seen, but skill-confirm-send failed: {exc}",
                paste_receipt=redacted_receipt,
            )

        if confirm.stop:
            return report(
                SendStatus.ABORTED_RISK_PHRASE,
                f"Circuit breaker triggered on confirm-send: {confirm.reason}",
                paste_receipt=redacted_receipt,
                confirm_result=confirm.raw,
            )
        if not confirm.counted:
            return report(
                SendStatus.SEND_UNCONFIRMED,
                f"Bubble seen, but confirm-send returned counted=false: {confirm.reason}",
                paste_receipt=redacted_receipt,
                confirm_result=confirm.raw,
            )

        steps.append("send_confirmed")
        return report(
            SendStatus.SENT_CONFIRMED,
            "Message sent and confirmed by new outgoing green bubble.",
            paste_receipt=redacted_receipt,
            confirm_result=confirm.raw,
        )
