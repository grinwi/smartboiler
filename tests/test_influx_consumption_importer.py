from datetime import datetime, timedelta
from unittest.mock import patch

import pandas as pd

from smartboiler.hdo_learner import HDOLearner
from smartboiler.influx_consumption_importer import seed_hdo_learner


class _Recorder:
    def __init__(self):
        self.obs = []

    def observe(self, dt, relay_unavailable: bool) -> None:
        self.obs.append((dt, relay_unavailable))


def test_seed_hdo_learner_infers_hdo_from_auxiliary_telemetry_gap():
    # 3-week window so the single 40-minute gap appears in all 3 ISO weeks
    # (we place the gap at a fixed weekday×minute-of-day by anchoring to a
    # Monday and repeating the same gap offset in all three weeks).
    import math

    end = datetime.now().replace(second=0, microsecond=0)
    start = end - timedelta(weeks=3)

    # Build an aux-sensor series that has a 40-min gap (minutes 41–80 relative
    # to the start of each week) repeated in all 3 weeks.
    online_parts = []
    for week in range(3):
        week_start = start + timedelta(weeks=week)
        before = pd.date_range(
            week_start + timedelta(minutes=1),
            week_start + timedelta(minutes=40),
            freq="1min",
        )
        after = pd.date_range(
            week_start + timedelta(minutes=81),
            week_start + timedelta(minutes=160),
            freq="1min",
        )
        online_parts.append(pd.Series(0.0, index=before))
        online_parts.append(pd.Series(0.0, index=after))
    power_series = pd.concat(online_parts)

    def fake_query_series(client, meas, entity, field, agg, t0, t1, fill="null", dtype="float"):
        if entity == "sensor.power":
            return power_series
        return pd.Series(dtype=float)

    learner = HDOLearner(decay_weeks=4)
    with patch("smartboiler.influx_consumption_importer.query_series", side_effect=fake_query_series):
        count = seed_hdo_learner(
            client=object(),
            relay_entity="switch.boiler",
            meas_state="state",
            hdo_learner=learner,
            start=start,
            end=end,
            presence_sources=[("W", "sensor.power")],
        )

    # The 40-minute gap (minutes 41–80) must be detected as HDO in every week,
    # so the corresponding (weekday, hour) slots should be blocked.
    gap_start_ts = start + timedelta(minutes=41)
    assert count > 0
    assert learner.is_blocked(gap_start_ts.weekday(), gap_start_ts.hour), (
        "Expected HDO block for the repeated 40-min gap"
    )


def test_seed_hdo_learner_does_not_infer_gaps_without_auxiliary_presence():
    # Without auxiliary presence sources, no gap analysis is possible —
    # the `known` mask comes only from relay data, which is empty here.
    # The new algorithm finds no bounded gaps → injects 0 observations.
    end = datetime.now().replace(second=0, microsecond=0)
    start = end - timedelta(hours=2)

    def fake_query_series(client, meas, entity, field, agg, t0, t1, fill="null", dtype="float"):
        return pd.Series(dtype=float)

    learner = _Recorder()
    with patch("smartboiler.influx_consumption_importer.query_series", side_effect=fake_query_series):
        count = seed_hdo_learner(
            client=object(),
            relay_entity="switch.boiler",
            meas_state="state",
            hdo_learner=learner,
            start=start,
            end=end,
        )

    assert count == 0
    assert learner.obs == []


def test_seed_hdo_learner_simulation_learns_weekly_schedule_from_repeated_gaps():
    end = datetime.now().replace(second=0, microsecond=0)
    start = end - timedelta(weeks=3)
    all_minutes = pd.date_range(start + timedelta(minutes=1), end, freq="1min")
    online_mask = ~((all_minutes.hour >= 22) | (all_minutes.hour < 6))
    power_series = pd.Series(0.0, index=all_minutes[online_mask], dtype=float)

    def fake_query_series(client, meas, entity, field, agg, t0, t1, fill="null", dtype="float"):
        if entity == "sensor.power":
            return power_series
        return pd.Series(dtype=float)

    learner = HDOLearner(decay_weeks=4)
    with patch("smartboiler.influx_consumption_importer.query_series", side_effect=fake_query_series):
        count = seed_hdo_learner(
            client=object(),
            relay_entity="switch.boiler",
            meas_state="state",
            hdo_learner=learner,
            start=start,
            end=end,
            presence_sources=[("W", "sensor.power")],
        )

    assert count > 0
    for wd in range(7):
        for hour in (22, 23, 0, 1, 2, 3, 4, 5):
            assert learner.is_blocked(wd, hour), f"Expected HDO block at weekday={wd} hour={hour}"
        for hour in (7, 12, 18):
            assert not learner.is_blocked(wd, hour), f"Unexpected HDO block at weekday={wd} hour={hour}"
