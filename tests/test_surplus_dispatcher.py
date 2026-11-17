"""Unit tests for SurplusDispatcher — start/stop rules and hysteresis."""
from smartboiler.surplus_dispatcher import SurplusDispatcher

HEATER_W = 2000.0


def _run(d, readings, allowed=True):
    return [d.update(r, allowed) for r in readings]


def test_starts_after_two_readings_with_enough_surplus():
    d = SurplusDispatcher(HEATER_W)
    assert _run(d, [1700, 1700]) == [False, True]   # 1700 ≥ 0.8 × 2000
    assert d.active


def test_single_spike_does_not_start():
    d = SurplusDispatcher(HEATER_W)
    assert _run(d, [2500, 300, 2500]) == [False, False, False]


def test_too_little_surplus_does_not_start():
    d = SurplusDispatcher(HEATER_W)
    assert _run(d, [1500, 1500, 1500]) == [False, False, False]   # < 1600 W


def test_keeps_running_while_partly_importing():
    d = SurplusDispatcher(HEATER_W)
    _run(d, [2000, 2000])
    # Heater draw now included in the reading: -800 W = 60 % still solar
    assert _run(d, [-800, -900, 200]) == [True, True, True]


def test_stops_after_two_readings_of_heavy_import():
    d = SurplusDispatcher(HEATER_W)
    _run(d, [2000, 2000])
    assert _run(d, [-1200, -1300]) == [True, False]   # > 1000 W import
    assert not d.active


def test_single_import_dip_does_not_stop():
    d = SurplusDispatcher(HEATER_W)
    _run(d, [2000, 2000])
    assert _run(d, [-1500, 100, -1500]) == [True, True, True]


def test_not_allowed_resets_immediately():
    d = SurplusDispatcher(HEATER_W)
    _run(d, [2000, 2000])
    assert d.update(2000, allowed=False) is False
    assert not d.active
    # Needs two fresh readings again
    assert _run(d, [2000, 2000]) == [False, True]


def test_missing_reading_resets():
    d = SurplusDispatcher(HEATER_W)
    _run(d, [2000, 2000])
    assert d.update(None, allowed=True) is False
    assert not d.active


def test_start_counter_resets_on_missing_reading():
    d = SurplusDispatcher(HEATER_W)
    assert _run(d, [2000, None, 2000]) == [False, False, False]
