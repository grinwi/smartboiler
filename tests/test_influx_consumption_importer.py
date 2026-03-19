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
    end = datetime.now().replace(second=0, microsecond=0)
    start = end - timedelta(hours=2)
    power_idx = (
        pd.date_range(start + timedelta(minutes=1), start + timedelta(minutes=40), freq="1min")
        .append(pd.date_range(start + timedelta(minutes=81), end, freq="1min"))
    )
    power_series = pd.Series(0.0, index=power_idx, dtype=float)

    def fake_query_series(client, meas, entity, field, agg, t0, t1, fill="null", dtype="float"):
        if entity == "sensor.power":
            return power_series
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
            presence_sources=[("W", "sensor.power")],
        )

    observed = {dt: blocked for dt, blocked in learner.obs}
    assert count == len(learner.obs)
    assert observed[start + timedelta(minutes=20)] is False
    assert observed[start + timedelta(minutes=60)] is True
    assert observed[start + timedelta(minutes=75)] is True
    assert observed[start + timedelta(minutes=110)] is False


def test_seed_hdo_learner_does_not_infer_gaps_without_auxiliary_presence():
    end = datetime.now().replace(second=0, microsecond=0)
    start = end - timedelta(hours=2)
    relay_series = pd.Series(
        [False, False],
        index=pd.to_datetime([start + timedelta(minutes=5), start + timedelta(minutes=115)]),
        dtype=object,
    )

    def fake_query_series(client, meas, entity, field, agg, t0, t1, fill="null", dtype="float"):
        if entity == "switch.boiler" and dtype == "bool":
            return relay_series
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

    assert count == 2
    assert all(blocked is False for _, blocked in learner.obs)


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
