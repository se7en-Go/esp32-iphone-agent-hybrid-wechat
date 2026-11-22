"""Unit tests for ClickChain and strict measured_pointer object parsing."""

from pathlib import Path

import pytest

from tests.fakes import FakePhoneDriver
from wechat_agent.adapters.driver import ConflictError
from wechat_agent.evidence import RunEvidence
from wechat_agent.models import Box, MeasuredPointer, Observation, PointResult
from wechat_agent.workflows.click_chain import (
    ClickChain,
    PointerConflictError,
    PointerNotConfirmedError,
    SendPreconditionError,
)


def test_measured_pointer_strict_dict_parsing() -> None:
    assert MeasuredPointer.from_point_response({"measured_pointer": None}) is None
    mp = MeasuredPointer.from_point_response({"measured_pointer": {"x": 0.41, "y": 0.62}})
    assert mp == MeasuredPointer(0.41, 0.62)

    with pytest.raises(ValueError, match="JSON object"):
        MeasuredPointer.from_point_response({"measured_pointer": [0.41, 0.62]})
    with pytest.raises(ValueError, match="numbers"):
        MeasuredPointer.from_point_response({"measured_pointer": {"x": True, "y": 0.5}})


def test_atomic_click_chain_uses_measured_pointer_not_aim(tmp_path: Path) -> None:
    driver = FakePhoneDriver()
    target_box = Box(0.75, 0.85, 0.98, 0.93)
    # Measured pointer differs slightly from requested aim (0.865, 0.89), still inside box.
    driver.point_handler = lambda x, y, sid, tb, rh: PointResult(
        screen_id="pt-1",
        measured_pointer=MeasuredPointer(0.84, 0.88),
        pointer_seen=True,
    )
    driver.observe_for_click_queue.append(Observation(screen_id="obs-fc-1"))

    ev = RunEvidence(tmp_path)
    chain = ClickChain(driver, ev, rehome_retries=1)
    chain.point_and_click("obs-0", (0.865, 0.89), target_box, step_name="test_click", target_label="发送")

    names = [name for name, _ in driver.calls]
    assert names == ["point", "observe", "click_observed"]
    assert driver.calls[1][1]["for_click"] is True
    click_args = driver.calls[2][1]
    assert (click_args["pointer_x"], click_args["pointer_y"]) == (0.84, 0.88)
    assert click_args["screen_id"] == "obs-fc-1"
    assert click_args["target_label"] == "发送"


def test_point_rehome_on_null_or_out_of_box(tmp_path: Path) -> None:
    driver = FakePhoneDriver()
    target_box = Box(0.20, 0.20, 0.40, 0.30)
    calls = {"n": 0}

    def handle_point(x, y, sid, tb, rehome):
        calls["n"] += 1
        if calls["n"] == 1:
            assert rehome is False
            return PointResult(screen_id="pt-1", measured_pointer=None, pointer_seen=False)
        assert rehome is True
        return PointResult(screen_id="pt-2", measured_pointer=MeasuredPointer(0.30, 0.25), pointer_seen=True)

    driver.point_handler = handle_point
    driver.observe_queue.append(Observation(screen_id="obs-rehome"))
    driver.observe_for_click_queue.append(Observation(screen_id="obs-fc"))

    chain = ClickChain(driver, RunEvidence(tmp_path), rehome_retries=1)
    res = chain.point_and_click("obs-0", (0.30, 0.25), target_box, step_name="rehome_test")
    assert res.clicked is True
    assert [name for name, _ in driver.calls] == ["point", "observe", "point", "observe", "click_observed"]
    assert driver.calls[2][1]["rehome"] is True


def test_point_never_clicks_when_measured_pointer_remains_outside(tmp_path: Path) -> None:
    driver = FakePhoneDriver()
    target_box = Box(0.20, 0.20, 0.40, 0.30)
    driver.point_handler = lambda x, y, sid, tb, rh: PointResult(
        screen_id="pt",
        measured_pointer=MeasuredPointer(0.90, 0.90),
        pointer_seen=True,
    )
    chain = ClickChain(driver, RunEvidence(tmp_path), rehome_retries=1)
    with pytest.raises(PointerNotConfirmedError):
        chain.point_and_click("obs-0", (0.30, 0.25), target_box, step_name="bad_pt")
    assert not any(name == "click_observed" for name, _ in driver.calls)


def test_pre_send_409_caret_retries_pointer_only_never_clicks(tmp_path: Path) -> None:
    driver = FakePhoneDriver()
    target_box = Box(0.75, 0.85, 0.98, 0.93)
    attempts = {"n": 0}

    def handle_point(x, y, sid, tb, rh):
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise ConflictError("HTTP 409: caret blinked", status_code=409)
        return PointResult(screen_id="pt-ok", measured_pointer=MeasuredPointer(0.86, 0.89), pointer_seen=True)

    driver.point_handler = handle_point
    chain = ClickChain(driver, RunEvidence(tmp_path))
    mp, _ = chain.pre_send_point_with_caret_retry(
        (0.86, 0.89),
        target_box,
        step_name="pre_send",
        max_attempts=3,
        initial_screen_id="obs-initial",
    )
    assert mp == MeasuredPointer(0.86, 0.89)
    assert [name for name, _ in driver.calls] == ["point", "observe", "point", "observe", "point"]
    assert not any(name == "click_observed" for name, _ in driver.calls)


def test_single_shot_send_click_never_retries_on_error(tmp_path: Path) -> None:
    driver = FakePhoneDriver()
    target_box = Box(0.75, 0.85, 0.98, 0.93)
    driver.click_handler = lambda *args: (_ for _ in ()).throw(
        ConflictError("HTTP 409: screen changed", status_code=409)
    )
    chain = ClickChain(driver, RunEvidence(tmp_path))
    with pytest.raises(ConflictError):
        chain.single_shot_send_click(MeasuredPointer(0.86, 0.89), target_box, "发送")
    click_calls = [c for c in driver.calls if c[0] == "click_observed"]
    assert len(click_calls) == 1


def test_pre_send_and_send_click_forward_caret_scope(tmp_path: Path) -> None:
    driver = FakePhoneDriver()
    target_box = Box(0.75, 0.85, 0.98, 0.93)
    caret_scope = Box(0.132, 0.528, 0.764, 0.576)
    chain = ClickChain(driver, RunEvidence(tmp_path))

    mp, _ = chain.pre_send_point_with_caret_retry(
        (0.86, 0.89),
        target_box,
        step_name="pre_send",
        initial_screen_id="obs-1",
        caret_scope=caret_scope,
    )
    chain.single_shot_send_click(mp, target_box, "发送", caret_scope=caret_scope)

    pt_call = [args for name, args in driver.calls if name == "point"][0]
    clk_call = [args for name, args in driver.calls if name == "click_observed"][0]
    assert pt_call["caret_scope"] == caret_scope
    assert clk_call["caret_scope"] == caret_scope


def test_point_409_raises_pointer_conflict_while_click_409_stays_plain_conflict(tmp_path: Path) -> None:
    target_box = Box(0.20, 0.20, 0.40, 0.30)

    driver1 = FakePhoneDriver()
    driver1.point_handler = lambda *args: (_ for _ in ()).throw(
        ConflictError("HTTP 409: screen changed", status_code=409)
    )
    chain1 = ClickChain(driver1, RunEvidence(tmp_path / "r1"))
    with pytest.raises(PointerConflictError):
        chain1.point_and_click("obs-0", (0.30, 0.25), target_box, step_name="pt409")
    assert not any(name == "click_observed" for name, _ in driver1.calls)

    driver2 = FakePhoneDriver()
    driver2.click_handler = lambda *args: (_ for _ in ()).throw(
        ConflictError("HTTP 409: screen changed", status_code=409)
    )
    chain2 = ClickChain(driver2, RunEvidence(tmp_path / "r2"))
    with pytest.raises(ConflictError) as exc_info:
        chain2.point_and_click("obs-0", (0.30, 0.25), target_box, step_name="clk409")
    assert not isinstance(exc_info.value, PointerConflictError)
    assert len([c for c in driver2.calls if c[0] == "click_observed"]) == 1


def test_single_shot_send_click_for_click_409_and_shifted_button_abort_without_clicking(tmp_path: Path) -> None:
    target_box = Box(0.75, 0.85, 0.98, 0.93)
    mp = MeasuredPointer(0.86, 0.89)

    # 1. observe(for_click=True) returning 409 raises PointerConflictError with zero click_observed calls
    driver_409 = FakePhoneDriver()
    driver_409.observe_for_click_queue.append(ConflictError("HTTP 409: frame stale", status_code=409))
    chain_409 = ClickChain(driver_409, RunEvidence(tmp_path / "fc409"))
    with pytest.raises(PointerConflictError):
        chain_409.single_shot_send_click(mp, target_box, "发送")
    assert not any(name == "click_observed" for name, _ in driver_409.calls)

    # 2. verify_for_click reports Send button shifted away from measured_pointer -> aborts with zero click
    driver_shift = FakePhoneDriver()
    driver_shift.observe_for_click_queue.append(Observation(screen_id="fc-shifted", image_bytes=b"frame"))
    chain_shift = ClickChain(driver_shift, RunEvidence(tmp_path / "shift"))
    shifted_box = Box(0.75, 0.60, 0.98, 0.68)
    with pytest.raises(SendPreconditionError, match="shifted"):
        chain_shift.single_shot_send_click(
            mp,
            target_box,
            "发送",
            verify_for_click=lambda obs: (shifted_box, "发送", None),
        )
    assert not any(name == "click_observed" for name, _ in driver_shift.calls)

    # 3. verify_for_click updates fresh box & caret_scope passed into click_observed
    driver_ok = FakePhoneDriver()
    driver_ok.observe_for_click_queue.append(Observation(screen_id="fc-fresh", image_bytes=b"frame"))
    chain_ok = ClickChain(driver_ok, RunEvidence(tmp_path / "fresh"))
    fresh_box = Box(0.76, 0.855, 0.975, 0.925)
    fresh_caret = Box(0.14, 0.50, 0.76, 0.57)
    chain_ok.single_shot_send_click(
        mp,
        target_box,
        "发送",
        verify_for_click=lambda obs: (fresh_box, "Send", fresh_caret),
    )
    clk_call = [args for name, args in driver_ok.calls if name == "click_observed"][0]
    assert clk_call["target_box"] == fresh_box
    assert clk_call["target_label"] == "Send"
    assert clk_call["caret_scope"] == fresh_caret


