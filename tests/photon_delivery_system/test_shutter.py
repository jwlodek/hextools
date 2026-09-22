import asyncio
import time

import bluesky.plan_stubs as bps
import pytest
from bluesky.run_engine import RunEngine
from ophyd_async.core import callback_on_mock_execute, init_devices, set_mock_value

from hextools.photon_delivery_system.shutter import (
    Shutter,
    ShutterStatus,
    ensure_shutter_state,
)


@pytest.fixture
def shutter() -> Shutter:
    with init_devices(mock=True):
        ps = Shutter("TEST:SHUTTER:", name="test_shutter")
    return ps


async def _delayed_readback(ps: Shutter, value: bool, delay: float):
    await asyncio.sleep(delay)
    set_mock_value(ps.status, ShutterStatus.OPEN if value else ShutterStatus.CLOSED)


@pytest.mark.parametrize("delay", [0.0, 0.05, 0.1, 0.15])
async def test_shutter_open_close_behavior(
    RE: RunEngine, shutter: Shutter, delay: float
):
    set_mock_value(shutter.status, ShutterStatus.CLOSED)

    callback_on_mock_execute(
        shutter.open_cmd,
        lambda *_: asyncio.ensure_future(_delayed_readback(shutter, True, delay)),
    )
    callback_on_mock_execute(
        shutter.close_cmd,
        lambda *_: asyncio.ensure_future(_delayed_readback(shutter, False, delay)),
    )

    t0 = time.monotonic()
    RE(bps.mv(shutter, True))
    open_duration = time.monotonic() - t0
    assert await shutter.status.get_value() is ShutterStatus.OPEN
    assert open_duration >= delay
    assert open_duration < delay + 0.1

    t0 = time.monotonic()
    RE(bps.mv(shutter, False))
    close_duration = time.monotonic() - t0
    assert await shutter.status.get_value() is ShutterStatus.CLOSED
    assert close_duration >= delay
    assert close_duration < delay + 0.1


@pytest.mark.parametrize(
    ("initial_state", "desired_state", "allow_actuation", "expected_executions"),
    (
        # Already in the desired state, so no actuation is needed
        (True, True, False, 0),
        (True, True, True, 0),
        (False, False, False, 0),
        (False, False, True, 0),
        # Not in the desired state, and actuation is allowed
        (False, True, True, 1),
        (True, False, True, 1),
        # Not in the desired state, and actuation is not allowed
        (False, True, False, None),
        (True, False, False, None),
    ),
)
async def test_ensure_shutter_state(
    RE: RunEngine,
    shutter: Shutter,
    initial_state: bool,
    desired_state: bool,
    allow_actuation: bool,
    expected_executions: int | None,
):
    set_mock_value(
        shutter.status, ShutterStatus.OPEN if initial_state else ShutterStatus.CLOSED
    )

    executions: list[bool] = []

    def _on_execute(value: bool):
        def _callback(*_):
            executions.append(value)
            asyncio.ensure_future(_delayed_readback(shutter, value, 0.0))

        return _callback

    callback_on_mock_execute(shutter.open_cmd, _on_execute(True))
    callback_on_mock_execute(shutter.close_cmd, _on_execute(False))

    if expected_executions is None:
        expected_status = "open" if initial_state else "not open"
        with pytest.raises(
            ValueError,
            match=(
                f"{shutter.name} is {expected_status}, but the desired state is "
                f"{'open' if desired_state else 'closed'}!"
            ),
        ):
            RE(ensure_shutter_state(shutter, desired_state))
        assert executions == []
        # The shutter should not have been actuated
        assert await shutter.status.get_value() is (
            ShutterStatus.OPEN if initial_state else ShutterStatus.CLOSED
        )
    else:
        RE(
            ensure_shutter_state(
                shutter, desired_state, allow_actuation=allow_actuation
            )
        )
        assert executions == [desired_state] * expected_executions
        assert await shutter.status.get_value() is (
            ShutterStatus.OPEN if desired_state else ShutterStatus.CLOSED
        )
