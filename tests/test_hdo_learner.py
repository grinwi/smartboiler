"""
Tests for HDOLearner — HDO (ripple control) pattern detection and scheduling.

HDO blocking signal: the boiler switch entity state == "unavailable" in HA.
This happens because the HDO relay physically cuts the circuit.

relay available + power == 0  →  boiler thermostat reached target temp — NOT HDO.

Resolution: 5-minute slots (288 per day).

Trust policy: a slot is only considered blocked when observations span at least
MIN_WEEKS_TO_TRUST distinct ISO calendar weeks.

Covers:
- Explicit schedule parsing: whole-hour, sub-hour (12:05-13:35), midnight-wrap,
  multi-segment, edge cases
- Observation recording: relay_unavailable=True/False, old-data pruning
- is_blocked_at() real-time check (5-min resolution)
- is_blocked(weekday, hour) backward-compat check (coarsened to hour)
- MIN_WEEKS_TO_TRUST enforcement
- Confidence calculation with exponential decay
- get_blocked_hours_next_24h() day-boundary logic
- get_weekly_schedule() full grid
- Bootstrap simulation with new unavailable-state signal
"""
from datetime import datetime, timedelta

import pytest

from smartboiler.hdo_learner import (
    HDOLearner,
    HISTORY_WEEKS,
    MIN_CONFIDENCE_TO_BLOCK,
    MIN_WEEKS_TO_TRUST,
    SLOT_MINUTES,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_dt(weekday: int, hour: int, minute: int = 0, weeks_ago: float = 0.0) -> datetime:
    """Return a naive local datetime on the given weekday/hour/minute, weeks_ago in the past."""
    now = datetime.now().replace(minute=0, second=0, microsecond=0)
    days_back = (now.weekday() - weekday) % 7
    base = now - timedelta(days=days_back, weeks=weeks_ago)
    return base.replace(hour=hour, minute=minute, second=0, microsecond=0)


def _obs_blocked(learner: HDOLearner, dt: datetime) -> None:
    learner.observe(dt, relay_unavailable=True)


def _obs_clear(learner: HDOLearner, dt: datetime) -> None:
    learner.observe(dt, relay_unavailable=False)


def _feed_two_weeks(learner: HDOLearner, weekday: int, hour: int, minute: int = 0,
                    blocked: bool = True) -> None:
    """Feed one observation per calendar-week for 2 different weeks (satisfies MIN_WEEKS_TO_TRUST)."""
    for w in range(MIN_WEEKS_TO_TRUST):
        dt = _make_dt(weekday, hour, minute, weeks_ago=w)
        learner.observe(dt, relay_unavailable=blocked)


# ---------------------------------------------------------------------------
# Explicit schedule parsing
# ---------------------------------------------------------------------------

class TestSetExplicitSchedule:
    def test_simple_daytime_window(self):
        l = HDOLearner()
        l.set_explicit_schedule("13:00-15:00")
        for wd in range(7):
            assert l.is_blocked(wd, 13)
            assert l.is_blocked(wd, 14)
            assert not l.is_blocked(wd, 15)

    def test_sub_hour_window(self):
        """12:05-13:35 must block the correct 5-min slots, not whole hours."""
        l = HDOLearner()
        l.set_explicit_schedule("12:05-13:35")
        for wd in range(7):
            # 12:00 slot is NOT in the window (window starts at 12:05)
            assert not l.is_blocked_at(_make_dt(wd, 12, 0)), f"12:00 should NOT be blocked wd={wd}"
            # 12:05 IS in the window
            assert l.is_blocked_at(_make_dt(wd, 12, 5)), f"12:05 should be blocked wd={wd}"
            # 13:30 IS in the window
            assert l.is_blocked_at(_make_dt(wd, 13, 30)), f"13:30 should be blocked wd={wd}"
            # 13:35 is the end → NOT blocked
            assert not l.is_blocked_at(_make_dt(wd, 13, 35)), f"13:35 should NOT be blocked wd={wd}"

    def test_hour_blocked_if_any_slot_blocked(self):
        """is_blocked(wd, hour) returns True if ANY 5-min slot in that hour is blocked."""
        l = HDOLearner()
        l.set_explicit_schedule("12:05-12:10")  # only 1 slot inside hour 12
        for wd in range(7):
            assert l.is_blocked(wd, 12)   # hour 12 has at least one blocked slot
            assert not l.is_blocked(wd, 11)

    def test_midnight_wrap(self):
        l = HDOLearner()
        l.set_explicit_schedule("22:00-06:00")
        for wd in range(7):
            for h in [22, 23, 0, 1, 2, 3, 4, 5]:
                assert l.is_blocked(wd, h), f"expected blocked at wd={wd} h={h}"
            assert not l.is_blocked(wd, 6)

    def test_multi_segment(self):
        l = HDOLearner()
        l.set_explicit_schedule("22:00-06:00,13:00-15:00")
        for wd in range(7):
            assert l.is_blocked(wd, 22)
            assert l.is_blocked(wd, 3)
            assert l.is_blocked(wd, 13)
            assert l.is_blocked(wd, 14)
            assert not l.is_blocked(wd, 7)

    def test_empty_string_clears_schedule(self):
        l = HDOLearner()
        l.set_explicit_schedule("13:00-15:00")
        l.set_explicit_schedule("")
        for wd in range(7):
            assert not l.is_blocked(wd, 13)
        assert not l._explicit_blocked

    def test_explicit_keyword_empty(self):
        l = HDOLearner()
        l.set_explicit_schedule("empty")
        assert not l._explicit_blocked

    def test_none_value(self):
        l = HDOLearner()
        l.set_explicit_schedule(None)
        assert not l._explicit_blocked

    def test_malformed_segment_ignored(self):
        l = HDOLearner()
        l.set_explicit_schedule("garbage,13:00-15:00")
        for wd in range(7):
            assert l.is_blocked(wd, 13)  # valid segment still parsed
            assert not l.is_blocked(wd, 9)

    def test_zero_length_segment_ignored(self):
        l = HDOLearner()
        l.set_explicit_schedule("13:00-13:00")
        for wd in range(7):
            assert not l.is_blocked(wd, 13)

    def test_setting_schedule_twice_replaces_old(self):
        l = HDOLearner()
        l.set_explicit_schedule("22:00-23:00")
        l.set_explicit_schedule("08:00-09:00")
        for wd in range(7):
            assert l.is_blocked(wd, 8)
            assert not l.is_blocked(wd, 22)


# ---------------------------------------------------------------------------
# observe() — basic behaviour
# ---------------------------------------------------------------------------

class TestObserve:
    def test_unavailable_is_recorded(self):
        l = HDOLearner()
        _obs_blocked(l, _make_dt(0, 10))
        assert l.observation_count() == 1

    def test_available_is_also_recorded(self):
        """relay available (on/off) must also be recorded — it's evidence against HDO."""
        l = HDOLearner()
        _obs_clear(l, _make_dt(0, 10))
        assert l.observation_count() == 1

    def test_both_states_counted_separately(self):
        l = HDOLearner()
        dt = _make_dt(0, 10)
        l.observe(dt, relay_unavailable=True)
        l.observe(dt, relay_unavailable=False)
        assert l.observation_count() == 2

    def test_old_observations_pruned(self):
        l = HDOLearner()
        # Inject an observation older than HISTORY_WEEKS directly
        old_dt = datetime.now() - timedelta(weeks=HISTORY_WEEKS + 1)
        slot_idx = (old_dt.hour * 60 + old_dt.minute) // SLOT_MINUTES
        key = (old_dt.weekday(), slot_idx)
        iso = old_dt.isocalendar()
        l._observations[key] = [(old_dt.timestamp(), True, (int(iso[0]), int(iso[1])))]
        # Trigger pruning via a new observe() on the same slot
        recent_dt = _make_dt(old_dt.weekday(), old_dt.hour, old_dt.minute)
        l.observe(recent_dt, relay_unavailable=True)
        assert len(l._observations[key]) == 1  # only the recent one remains

    def test_5min_slot_granularity(self):
        """12:05 and 12:10 land in different slots."""
        l = HDOLearner()
        dt1 = _make_dt(0, 12, 5)
        dt2 = _make_dt(0, 12, 10)
        l.observe(dt1, relay_unavailable=True)
        l.observe(dt2, relay_unavailable=True)
        slot1 = (12 * 60 + 5) // SLOT_MINUTES    # 145
        slot2 = (12 * 60 + 10) // SLOT_MINUTES   # 146
        assert slot1 != slot2
        assert len(l._observations.get((0, slot1), [])) == 1
        assert len(l._observations.get((0, slot2), [])) == 1


# ---------------------------------------------------------------------------
# Trust policy: MIN_WEEKS_TO_TRUST
# ---------------------------------------------------------------------------

class TestMinWeeksTrust:
    def test_single_week_not_trusted(self):
        """All observations in one calendar week → not trusted."""
        l = HDOLearner()
        dt = _make_dt(0, 10)
        for _ in range(20):
            l.observe(dt, relay_unavailable=True)
        assert not l.is_blocked(0, 10)

    def test_two_weeks_trusted(self):
        """Observations spanning 2 distinct weeks → trusted."""
        l = HDOLearner()
        _feed_two_weeks(l, 0, 10, blocked=True)
        assert l.is_blocked(0, 10)

    def test_two_weeks_not_blocked_when_clear(self):
        l = HDOLearner()
        _feed_two_weeks(l, 0, 10, blocked=False)
        assert not l.is_blocked(0, 10)


# ---------------------------------------------------------------------------
# is_blocked() — confidence logic
# ---------------------------------------------------------------------------

class TestIsBlocked:
    def test_no_data_returns_false(self):
        assert not HDOLearner().is_blocked(0, 10)

    def test_all_blocked_across_two_weeks(self):
        l = HDOLearner()
        _feed_two_weeks(l, 0, 10, blocked=True)
        assert l.is_blocked(0, 10)

    def test_all_clear_across_two_weeks(self):
        l = HDOLearner()
        _feed_two_weeks(l, 0, 10, blocked=False)
        assert not l.is_blocked(0, 10)

    def test_explicit_overrides_learned_not_blocked(self):
        """Explicit schedule takes priority even if learned data says not blocked."""
        l = HDOLearner()
        l.set_explicit_schedule("10:00-11:00")
        _feed_two_weeks(l, 0, 10, blocked=False)
        assert l.is_blocked(0, 10)  # explicit wins

    def test_exponential_decay_down_weights_old_blocked(self):
        """Old blocked observations heavily decayed; recent clear observations dominate."""
        l = HDOLearner(decay_weeks=1)
        wd, h = 1, 14
        # 10 blocked observations from 3 weeks ago (heavy decay)
        old = _make_dt(wd, h, weeks_ago=3)
        old_iso = old.isocalendar()
        slot_idx = (h * 60) // SLOT_MINUTES
        key = (wd, slot_idx)
        for _ in range(10):
            l._observations.setdefault(key, []).append(
                (old.timestamp(), True, (int(old_iso[0]), int(old_iso[1])))
            )
        # 5 recent clear observations from 2 different weeks
        for w in range(2):
            recent = _make_dt(wd, h, weeks_ago=w)
            for _ in range(3):
                l.observe(recent, relay_unavailable=False)
        # Recent clear data must dominate
        assert not l.is_blocked(wd, h)

    def test_mixed_confidence_below_threshold(self):
        """1 blocked + 4 clear across 2 weeks → 20% confidence → not blocked."""
        l = HDOLearner(decay_weeks=999)
        w0 = _make_dt(0, 10, weeks_ago=0)
        w1 = _make_dt(0, 10, weeks_ago=1)
        l.observe(w0, relay_unavailable=True)
        for _ in range(2):
            l.observe(w0, relay_unavailable=False)
            l.observe(w1, relay_unavailable=False)
        assert not l.is_blocked(0, 10)

    def test_mixed_confidence_above_threshold(self):
        """4 blocked + 1 clear across 2 weeks → 80% confidence → blocked."""
        l = HDOLearner(decay_weeks=999)
        w0 = _make_dt(0, 10, weeks_ago=0)
        w1 = _make_dt(0, 10, weeks_ago=1)
        for _ in range(2):
            l.observe(w0, relay_unavailable=True)
            l.observe(w1, relay_unavailable=True)
        l.observe(w0, relay_unavailable=False)
        assert l.is_blocked(0, 10)


# ---------------------------------------------------------------------------
# is_blocked_at() — precise 5-min datetime query
# ---------------------------------------------------------------------------

class TestIsBlockedAt:
    def test_exact_slot_blocked(self):
        """12:05 is blocked via explicit schedule."""
        l = HDOLearner()
        l.set_explicit_schedule("12:05-12:10")
        dt = _make_dt(0, 12, 5)
        assert l.is_blocked_at(dt)

    def test_adjacent_slot_not_blocked(self):
        """12:00 is NOT blocked when schedule starts at 12:05."""
        l = HDOLearner()
        l.set_explicit_schedule("12:05-12:10")
        dt = _make_dt(0, 12, 0)
        assert not l.is_blocked_at(dt)

    def test_learned_slot_blocked_at(self):
        l = HDOLearner()
        _feed_two_weeks(l, 0, 22, minute=30, blocked=True)
        dt = _make_dt(0, 22, 30)
        assert l.is_blocked_at(dt)


# ---------------------------------------------------------------------------
# get_blocked_hours_next_24h()
# ---------------------------------------------------------------------------

class TestGetBlockedHoursNext24h:
    def test_returns_24_bools(self):
        l = HDOLearner()
        result = l.get_blocked_hours_next_24h(datetime(2024, 3, 11, 8, 0))
        assert len(result) == 24
        assert all(isinstance(x, bool) for x in result)

    def test_explicit_night_window_reflected(self):
        l = HDOLearner()
        l.set_explicit_schedule("22:00-06:00")
        from_dt = datetime(2024, 3, 11, 20, 0)  # Monday 20:00
        result = l.get_blocked_hours_next_24h(from_dt)
        assert result[2] is True   # 22:00 Mon
        assert result[3] is True   # 23:00 Mon
        for i in range(4, 10):
            assert result[i] is True, f"expected index {i} blocked"
        assert result[10] is False  # 06:00 Tue

    def test_no_blocked_when_no_explicit_and_no_data(self):
        l = HDOLearner()
        assert not any(l.get_blocked_hours_next_24h())

    def test_uses_now_when_from_dt_is_none(self):
        l = HDOLearner()
        assert len(l.get_blocked_hours_next_24h(None)) == 24


# ---------------------------------------------------------------------------
# get_weekly_schedule()
# ---------------------------------------------------------------------------

class TestGetWeeklySchedule:
    def test_returns_all_seven_days(self):
        sched = HDOLearner().get_weekly_schedule()
        assert set(sched.keys()) == {"Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"}

    def test_explicit_schedule_reflected_in_weekly(self):
        l = HDOLearner()
        l.set_explicit_schedule("22:00-23:00")
        sched = l.get_weekly_schedule()
        for day in sched:
            assert 22 in sched[day]
            assert 23 not in sched[day]

    def test_empty_when_no_schedule_and_no_data(self):
        sched = HDOLearner().get_weekly_schedule()
        for day in sched:
            assert sched[day] == []


# ---------------------------------------------------------------------------
# observation_count()
# ---------------------------------------------------------------------------

class TestObservationCount:
    def test_zero_initially(self):
        assert HDOLearner().observation_count() == 0

    def test_counts_both_blocked_and_clear(self):
        l = HDOLearner()
        l.observe(_make_dt(0, 10), relay_unavailable=True)
        l.observe(_make_dt(1, 11), relay_unavailable=False)
        assert l.observation_count() == 2


# ---------------------------------------------------------------------------
# SIMULATION: Bootstrap from HA DB history (new unavailable-state signal)
# ---------------------------------------------------------------------------

def _simulate_ha_week_unavailable(
    week_offset_days: int,
    hdo_night_hours: list,
) -> list:
    """
    Simulate one week of boiler relay state as HA history objects.

    HDO hours → state "unavailable"  (physical relay cut by utility).
    Heating hours → state "on".
    Thermostat-off hours → state "off".

    Returns a list of HA history dicts for the relay entity.
    """
    relay_states = []
    today_midnight = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    week_start = today_midnight - timedelta(days=week_offset_days + 7)

    for day in range(7):
        for hour in range(24):
            dt = week_start + timedelta(days=day, hours=hour)
            if hour in hdo_night_hours:
                state = "unavailable"
            elif hour in [6, 7, 8, 17, 18, 19]:
                state = "on"
            else:
                state = "off"
            relay_states.append({
                "entity_id": "switch.boiler",
                "state": state,
                "last_changed": dt.isoformat() + "+00:00",
                "last_updated": dt.isoformat() + "+00:00",
                "attributes": {},
            })

    return relay_states


class TestHDOBootstrapFromHAHistory:
    """Bootstrap HDO learning from synthetic HA history using the unavailable-state signal."""

    HDO_HOURS = list(range(22, 24)) + list(range(0, 6))  # 22:00–06:00

    def _run_bootstrap(self, weeks: int = 5) -> HDOLearner:
        import pandas as pd

        learner = HDOLearner(decay_weeks=4)
        for w in range(weeks):
            relay_hist = _simulate_ha_week_unavailable(
                week_offset_days=w * 7,
                hdo_night_hours=self.HDO_HOURS,
            )
            for relay_obj in relay_hist:
                ts = pd.to_datetime(relay_obj["last_changed"], utc=True).tz_localize(None)
                relay_unavailable = relay_obj["state"] == "unavailable"
                learner.observe(ts.to_pydatetime(), relay_unavailable=relay_unavailable)
        return learner

    def test_infers_night_hours_as_blocked(self):
        learner = self._run_bootstrap(weeks=5)
        for wd in range(7):
            for h in self.HDO_HOURS:
                assert learner.is_blocked(wd, h), \
                    f"Expected HDO blocked at weekday={wd} hour={h}"

    def test_does_not_block_daytime_hours(self):
        learner = self._run_bootstrap(weeks=5)
        for wd in range(7):
            for h in [7, 8, 10, 12, 15, 17, 18]:
                assert not learner.is_blocked(wd, h), \
                    f"Expected NOT blocked at weekday={wd} hour={h}"

    def test_weekly_schedule_matches_simulated_hdo(self):
        learner = self._run_bootstrap(weeks=5)
        sched = learner.get_weekly_schedule()
        for day_hours in sched.values():
            for h in self.HDO_HOURS:
                assert h in day_hours

    def test_observation_count_grows_with_more_weeks(self):
        l2 = self._run_bootstrap(weeks=2)
        l5 = self._run_bootstrap(weeks=5)
        assert l5.observation_count() > l2.observation_count()

    def test_24h_forecast_correctly_flags_upcoming_hdo(self):
        learner = self._run_bootstrap(weeks=5)
        from_dt = datetime(2024, 3, 11, 20, 0)  # Monday 20:00
        blocked = learner.get_blocked_hours_next_24h(from_dt)
        assert blocked[2] is True   # 22:00 Mon
        assert blocked[3] is True   # 23:00 Mon
        assert blocked[0] is False  # 20:00 Mon
        assert blocked[1] is False  # 21:00 Mon

    def test_no_explicit_schedule_required_for_learned_detection(self):
        learner = self._run_bootstrap(weeks=5)
        assert not learner._explicit_blocked
        assert learner.is_blocked(0, 0)

    def test_unavailable_not_confused_with_thermostat_off(self):
        """relay available + off (thermostat) must NOT trigger HDO detection."""
        learner = HDOLearner()
        # 20 observations of state="off" (thermostat cut) — should never be blocked
        for w in range(5):
            dt = _make_dt(0, 10, weeks_ago=w)
            for _ in range(4):
                learner.observe(dt, relay_unavailable=False)  # "off" or "on" = available
        assert not learner.is_blocked(0, 10)


# ---------------------------------------------------------------------------
# Bootstrap with explicit schedule overlay
# ---------------------------------------------------------------------------

class TestHDOExplicitPlusLearned:
    def test_explicit_and_learned_union(self):
        learner = HDOLearner()
        _feed_two_weeks(learner, 0, 10, blocked=True)
        learner.set_explicit_schedule("22:00-23:00")
        assert learner.is_blocked(0, 10)   # learned
        assert learner.is_blocked(3, 22)   # explicit

    def test_explicit_overrides_even_when_data_says_clear(self):
        learner = HDOLearner()
        learner.set_explicit_schedule("10:00-11:00")
        _feed_two_weeks(learner, 0, 10, blocked=False)
        assert learner.is_blocked(0, 10)  # explicit wins
