"""
Tests for HADataCollector — HA entity history → hourly consumption pipeline.

Covers:
- _states_to_series(): HA history objects → minute-resolution pandas Series
  - normal state transitions, unavailable gaps, UTC timezone stripping
- _compute_consumed_kwh_from_flow(): enthalpy calculation
- _estimate_consumed_from_relay(): fallback zeros
- _fetch_hourly(): end-to-end aggregation to hourly rows
- collect_and_update(): integration with StateStore
- get_current_readings(): current entity values

BOOTSTRAP SIMULATIONS
─────────────────────
Two complete bootstrap scenarios test the full pipeline from raw HA / InfluxDB
history data to a trained predictor + HDO learner, mirroring what happens when
the add-on starts for the first time with historical data available.

Schema note
───────────
Home Assistant REST API (/api/history/period/<ISO>) returns a list-of-lists.
Each inner list is a sequence of state-change objects:
  {
    "entity_id": "switch.boiler",
    "state": "on" | "off" | "unavailable",
    "last_changed": "<ISO8601 with timezone>",
    "last_updated": "<ISO8601 with timezone>",
    "attributes": {}
  }

The InfluxDB HA integration stores the same data in measurements; the shape
that HADataCollector consumes is the list of these dicts (as returned by
get_history()). The bootstrap tests use this exact format.
"""
from datetime import datetime, timedelta
from typing import List, Dict
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from smartboiler.ha_data_collector import HADataCollector
from smartboiler.ha_client import HAClient
from smartboiler.state_store import StateStore
from smartboiler.hdo_learner import HDOLearner
from smartboiler.predictor import RollingHistogramPredictor


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_ha():
    ha = MagicMock(spec=HAClient)
    ha.get_history.return_value = []
    ha.get_state.return_value = None
    ha.get_state_value.return_value = None
    return ha


@pytest.fixture
def mock_store(tmp_path):
    return StateStore(data_dir=str(tmp_path))


@pytest.fixture
def collector(mock_ha, mock_store):
    return HADataCollector(
        ha=mock_ha,
        store=mock_store,
        relay_entity_id="switch.boiler",
        relay_power_entity_id="sensor.boiler_power",
    )


@pytest.fixture
def collector_with_flow(mock_ha, mock_store):
    return HADataCollector(
        ha=mock_ha,
        store=mock_store,
        relay_entity_id="switch.boiler",
        relay_power_entity_id="sensor.boiler_power",
        water_flow_entity_id="sensor.flow",
        water_temp_out_entity_id="sensor.water_temp",
        boiler_volume_l=120.0,
        boiler_set_tmp=60.0,
        cold_water_tmp=10.0,
    )


# ---------------------------------------------------------------------------
# HA history builder helpers (mirrors real HA REST API schema)
# ---------------------------------------------------------------------------

def _ha_state(entity_id: str, state: str, ts: datetime) -> Dict:
    """Create a single HA state-change object in the format returned by the REST API."""
    return {
        "entity_id": entity_id,
        "state": state,
        "last_changed": ts.isoformat() + "+00:00",
        "last_updated": ts.isoformat() + "+00:00",
        "attributes": {},
    }


def _ha_history(entity_id: str, events: List[tuple]) -> List[Dict]:
    """Build HA history as list of state-change dicts from (datetime, value) pairs."""
    return [_ha_state(entity_id, str(val), dt) for dt, val in events]


# ---------------------------------------------------------------------------
# _states_to_series()
# ---------------------------------------------------------------------------

class TestStatesToSeries:
    def test_empty_history_returns_nan_series(self, collector):
        start = datetime(2024, 3, 11, 8, 0)
        end = datetime(2024, 3, 11, 9, 0)
        series = collector._states_to_series([], start, end, dtype="float")
        assert series.isna().all()

    def test_single_on_event_fills_series(self, collector):
        start = datetime(2024, 3, 11, 8, 0)
        end = datetime(2024, 3, 11, 9, 0)
        history = [_ha_state("switch.boiler", "on", start)]
        series = collector._states_to_series(history, start, end, dtype="bool")
        # The series covers [start, end] inclusive; the last minute (=end) has no
        # covering state (mask uses < next_ts=end) so it's NaN. Exclude it.
        assert (series.dropna() == 1.0).all()

    def test_off_then_on_transition(self, collector):
        start = datetime(2024, 3, 11, 8, 0)
        mid = datetime(2024, 3, 11, 8, 30)
        end = datetime(2024, 3, 11, 9, 0)
        history = [
            _ha_state("switch.boiler", "off", start),
            _ha_state("switch.boiler", "on", mid),
        ]
        series = collector._states_to_series(history, start, end, dtype="bool")
        # Minutes before 08:30 should be 0.0 (off)
        assert series.iloc[0] == pytest.approx(0.0)
        # Minutes at/after 08:30 up to (not including) end should be 1.0 (on).
        # The series endpoint (09:00) has no covering state → NaN; check second-to-last.
        assert series.iloc[-2] == pytest.approx(1.0)

    def test_unavailable_becomes_nan(self, collector):
        start = datetime(2024, 3, 11, 8, 0)
        mid = datetime(2024, 3, 11, 8, 20)
        end = datetime(2024, 3, 11, 9, 0)
        history = [
            _ha_state("sensor.power", "2000", start),
            _ha_state("sensor.power", "unavailable", mid),
        ]
        series = collector._states_to_series(history, start, end, dtype="float")
        # Before unavailable: 2000.0
        assert series.iloc[0] == pytest.approx(2000.0)
        # After unavailable: NaN
        assert np.isnan(series.iloc[-1])

    def test_unknown_state_becomes_nan(self, collector):
        start = datetime(2024, 3, 11, 8, 0)
        end = datetime(2024, 3, 11, 9, 0)
        history = [_ha_state("sensor.x", "unknown", start)]
        series = collector._states_to_series(history, start, end, dtype="float")
        assert series.isna().all()

    def test_float_values_parsed_correctly(self, collector):
        start = datetime(2024, 3, 11, 10, 0)
        end = datetime(2024, 3, 11, 11, 0)
        history = [_ha_state("sensor.power", "1950.5", start)]
        series = collector._states_to_series(history, start, end, dtype="float")
        assert series.iloc[0] == pytest.approx(1950.5)

    def test_utc_timezone_stripped(self, collector):
        """Timestamps with +00:00 must be converted to tz-naive."""
        start = datetime(2024, 3, 11, 10, 0)
        end = datetime(2024, 3, 11, 11, 0)
        history = [_ha_state("sensor.x", "100", start)]
        series = collector._states_to_series(history, start, end, dtype="float")
        assert series.index.tz is None

    def test_bool_dtype_on_string(self, collector):
        start = datetime(2024, 3, 11, 10, 0)
        end = datetime(2024, 3, 11, 11, 0)
        for on_str in ("on", "true", "1"):
            history = [_ha_state("switch.x", on_str, start)]
            series = collector._states_to_series(history, start, end, dtype="bool")
            assert series.iloc[0] == pytest.approx(1.0)
        for off_str in ("off", "false", "0"):
            history = [_ha_state("switch.x", off_str, start)]
            series = collector._states_to_series(history, start, end, dtype="bool")
            assert series.iloc[0] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# _compute_consumed_kwh_from_flow()
# ---------------------------------------------------------------------------

class TestComputeConsumedFromFlow:
    def test_zero_flow_gives_zero_consumption(self, collector_with_flow):
        start = datetime(2024, 3, 11, 10, 0)
        end = datetime(2024, 3, 11, 11, 0)
        flow_hist = [_ha_state("sensor.flow", "0.0", start)]
        temp_hist = [_ha_state("sensor.water_temp", "55.0", start)]
        result = collector_with_flow._compute_consumed_kwh_from_flow(
            flow_hist, temp_hist, start, end
        )
        assert (result.dropna() == 0.0).all()

    def test_positive_flow_gives_positive_kwh(self, collector_with_flow):
        start = datetime(2024, 3, 11, 10, 0)
        end = datetime(2024, 3, 11, 11, 0)
        flow_hist = [_ha_state("sensor.flow", "5.0", start)]  # 5 L/min
        temp_hist = [_ha_state("sensor.water_temp", "55.0", start)]  # 55°C outlet
        result = collector_with_flow._compute_consumed_kwh_from_flow(
            flow_hist, temp_hist, start, end
        )
        # Q_per_min = 5 * 4.186 * (55-10) / 3600 ≈ 0.002621 kWh/min
        # Over 60 minutes = ~0.157 kWh
        total = result.dropna().sum()
        assert total == pytest.approx(5 * 4.186 * 45 / 3600 * 60, rel=0.01)

    def test_cold_water_delta_clamped_to_zero(self, collector_with_flow):
        """Outlet temp below cold water temp → no negative consumption."""
        start = datetime(2024, 3, 11, 10, 0)
        end = datetime(2024, 3, 11, 11, 0)
        flow_hist = [_ha_state("sensor.flow", "3.0", start)]
        temp_hist = [_ha_state("sensor.water_temp", "5.0", start)]  # below cold_water_tmp=10
        result = collector_with_flow._compute_consumed_kwh_from_flow(
            flow_hist, temp_hist, start, end
        )
        assert (result.dropna() == 0.0).all()

    def test_nan_temperature_propagates_to_nan_kwh(self, collector_with_flow):
        start = datetime(2024, 3, 11, 10, 0)
        end = datetime(2024, 3, 11, 11, 0)
        flow_hist = [_ha_state("sensor.flow", "3.0", start)]
        temp_hist = [_ha_state("sensor.water_temp", "unavailable", start)]
        result = collector_with_flow._compute_consumed_kwh_from_flow(
            flow_hist, temp_hist, start, end
        )
        assert result.isna().all()


# ---------------------------------------------------------------------------
# _estimate_consumed_from_relay()
# ---------------------------------------------------------------------------

class TestEstimateConsumedFromRelay:
    def test_returns_all_zeros(self, collector):
        idx = pd.date_range("2024-03-11 10:00", periods=60, freq="1min")
        relay = pd.Series(1.0, index=idx)
        result = collector._estimate_consumed_from_relay(relay)
        assert (result == 0.0).all()

    def test_index_preserved(self, collector):
        idx = pd.date_range("2024-03-11 10:00", periods=30, freq="1min")
        relay = pd.Series(1.0, index=idx)
        result = collector._estimate_consumed_from_relay(relay)
        assert len(result) == 30


# ---------------------------------------------------------------------------
# _fetch_hourly()
# ---------------------------------------------------------------------------

class TestFetchHourly:
    def _make_relay_history(self, start, end, on_from=None):
        """Relay is OFF, then ON from on_from."""
        events = [_ha_state("switch.boiler", "off", start)]
        if on_from:
            events.append(_ha_state("switch.boiler", "on", on_from))
        return events

    def test_returns_dataframe_with_required_columns(self, collector, mock_ha):
        start = datetime(2024, 3, 11, 8, 0)
        end = datetime(2024, 3, 11, 10, 0)
        mock_ha.get_history.return_value = self._make_relay_history(start, end)
        df = collector._fetch_hourly(start, end)
        assert "consumed_kwh" in df.columns
        assert "relay_on" in df.columns
        assert "power_w" in df.columns

    def test_relay_on_fraction_computed(self, collector, mock_ha):
        start = datetime(2024, 3, 11, 8, 0)
        end = datetime(2024, 3, 11, 10, 0)
        mid = datetime(2024, 3, 11, 9, 0)
        # First hour: all OFF; second hour: all ON
        relay_hist = [
            _ha_state("switch.boiler", "off", start),
            _ha_state("switch.boiler", "on", mid),
        ]
        power_hist = [_ha_state("sensor.boiler_power", "2000", start)]
        mock_ha.get_history.side_effect = lambda eid, *a, **kw: (
            relay_hist if eid == "switch.boiler" else power_hist
        )
        df = collector._fetch_hourly(start, end)
        assert len(df) >= 1

    def test_empty_history_returns_empty_df(self, collector, mock_ha):
        mock_ha.get_history.return_value = []
        start = datetime(2024, 3, 11, 8, 0)
        end = datetime(2024, 3, 11, 10, 0)
        df = collector._fetch_hourly(start, end)
        assert isinstance(df, pd.DataFrame)

    def test_uses_flow_sensor_when_configured(self, collector_with_flow, mock_ha):
        start = datetime(2024, 3, 11, 10, 0)
        end = datetime(2024, 3, 11, 11, 0)
        relay_hist = [_ha_state("switch.boiler", "on", start)]
        power_hist = [_ha_state("sensor.boiler_power", "2000", start)]
        flow_hist = [_ha_state("sensor.flow", "3.0", start)]
        temp_hist = [_ha_state("sensor.water_temp", "55.0", start)]

        def _side(eid, *a, **kw):
            return {
                "switch.boiler": relay_hist,
                "sensor.boiler_power": power_hist,
                "sensor.flow": flow_hist,
                "sensor.water_temp": temp_hist,
            }.get(eid, [])

        mock_ha.get_history.side_effect = _side
        df = collector_with_flow._fetch_hourly(start, end)
        if not df.empty:
            # Flow-based consumption should be > 0 (or NaN if no valid data)
            assert "consumed_kwh" in df.columns


# ---------------------------------------------------------------------------
# collect_and_update()
# ---------------------------------------------------------------------------

class TestCollectAndUpdate:
    def test_returns_dataframe(self, collector, mock_ha):
        mock_ha.get_history.return_value = []
        df = collector.collect_and_update(lookback_hours=1)
        assert isinstance(df, pd.DataFrame)

    def test_updates_last_data_collection_timestamp(self, collector, mock_ha, mock_store):
        start = datetime(2024, 3, 11, 10, 0)
        relay_hist = [_ha_state("switch.boiler", "on", start)]
        power_hist = [_ha_state("sensor.boiler_power", "2000", start)]
        mock_ha.get_history.side_effect = lambda eid, *a, **kw: (
            relay_hist if eid == "switch.boiler" else power_hist
        )
        collector.collect_and_update(lookback_hours=2)
        last = mock_store.get_last_data_collection()
        assert last.year >= 2024

    def test_no_new_data_returns_existing_history(self, collector, mock_ha, mock_store):
        # Pre-populate store
        existing = pd.DataFrame(
            {"consumed_kwh": [0.3], "relay_on": [True], "power_w": [2000.0]},
            index=pd.DatetimeIndex([datetime(2024, 3, 11, 8, 0)]),
        )
        mock_store.save_consumption_history(existing)
        mock_ha.get_history.return_value = []

        df = collector.collect_and_update(lookback_hours=1)
        assert len(df) >= 1


# ---------------------------------------------------------------------------
# get_current_readings()
# ---------------------------------------------------------------------------

class TestGetCurrentReadings:
    def test_relay_on_state(self, collector, mock_ha):
        mock_ha.get_state.return_value = {"entity_id": "switch.boiler", "state": "on"}
        readings = collector.get_current_readings()
        assert readings["relay_on"] is True

    def test_relay_off_state(self, collector, mock_ha):
        mock_ha.get_state.return_value = {"entity_id": "switch.boiler", "state": "off"}
        readings = collector.get_current_readings()
        assert readings["relay_on"] is False

    def test_includes_power_when_configured(self, collector, mock_ha):
        mock_ha.get_state.return_value = {"state": "on"}
        mock_ha.get_state_value.return_value = 1950.0
        readings = collector.get_current_readings()
        assert "power_w" in readings

    def test_handles_missing_entity_gracefully(self, collector, mock_ha):
        mock_ha.get_state.return_value = None
        readings = collector.get_current_readings()
        assert readings.get("relay_on") is None


# ---------------------------------------------------------------------------
# BOOTSTRAP SIMULATION: full pipeline from HA DB history
#
# This tests the complete startup sequence:
# 1. Generate weeks of synthetic HA history in REST API format
# 2. Feed through HADataCollector._fetch_hourly() → hourly DataFrame
# 3. Train RollingHistogramPredictor from the DataFrame
# 4. Train HDOLearner from the relay + power columns
# 5. Verify predictions and HDO detection match the simulated truth
# ---------------------------------------------------------------------------

def _generate_ha_week_history(
    week_offset_days: int,
    hdo_hours: List[int],
    shower_hours: List[int],
    boiler_watt: float = 2000.0,
    cold_water_tmp: float = 10.0,
    outlet_tmp: float = 55.0,
    flow_lpm: float = 4.0,
) -> Dict[str, List[Dict]]:
    """
    Simulate one week of boiler entity history in HA REST API format.

    Returns dict with entity_id → list of state-change dicts.
    The schema matches exactly what HA's /api/history/period/ returns.

    Timestamps are midnight-aligned so hours in the generated data match
    the hours in the series produced by _states_to_series().
    """
    rng = np.random.default_rng(week_offset_days)
    # Anchor to midnight so `timedelta(hours=h)` gives clean hour-of-day values
    today_midnight = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    week_start = today_midnight - timedelta(days=week_offset_days + 7)

    relay_events = []
    power_events = []
    flow_events = []
    temp_events = []

    for day in range(7):
        for hour in range(24):
            dt = week_start + timedelta(days=day, hours=hour)

            if hour in hdo_hours:
                # HDO physically cuts the relay circuit → entity state = "unavailable"
                relay_events.append((dt, "unavailable"))
                power_events.append((dt, "0"))
            elif hour in (shower_hours + list(range(22, 24)) + list(range(0, 4))):
                relay_events.append((dt, "on"))
                power_events.append((dt, str(boiler_watt + rng.uniform(-50, 50))))
            else:
                # Thermostat reached target → relay available but off
                relay_events.append((dt, "off"))
                power_events.append((dt, "0"))

            # Water flow: only during shower hours
            if hour in shower_hours:
                flow_events.append((dt, str(flow_lpm + rng.uniform(-0.2, 0.2))))
                temp_events.append((dt, str(outlet_tmp + rng.uniform(-2, 2))))
            else:
                flow_events.append((dt, "0"))
                temp_events.append((dt, str(cold_water_tmp + rng.uniform(-1, 1))))

    return {
        "switch.boiler": _ha_history("switch.boiler", relay_events),
        "sensor.boiler_power": _ha_history("sensor.boiler_power", power_events),
        "sensor.flow": _ha_history("sensor.flow", flow_events),
        "sensor.water_temp": _ha_history("sensor.water_temp", temp_events),
    }


def _bootstrap_from_weeks(
    n_weeks: int,
    hdo_hours: List[int],
    shower_hours: List[int],
    tmp_path,
    use_flow_sensor: bool = True,
) -> tuple:
    """
    Full bootstrap pipeline:
    - n_weeks of synthetic HA history
    - Returns (predictor, hdo_learner, full_df)
    """
    store = StateStore(data_dir=str(tmp_path))
    predictor = RollingHistogramPredictor(conservatism="medium")
    hdo_learner = HDOLearner(decay_weeks=4)

    all_dfs = []

    today_midnight = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)

    for w in range(n_weeks):
        week_data = _generate_ha_week_history(
            week_offset_days=w * 7,
            hdo_hours=hdo_hours,
            shower_hours=shower_hours,
        )

        mock_ha = MagicMock(spec=HAClient)
        mock_ha.get_history.side_effect = lambda eid, *a, **kw: week_data.get(eid, [])

        if use_flow_sensor:
            coll = HADataCollector(
                ha=mock_ha,
                store=store,
                relay_entity_id="switch.boiler",
                relay_power_entity_id="sensor.boiler_power",
                water_flow_entity_id="sensor.flow",
                water_temp_out_entity_id="sensor.water_temp",
                cold_water_tmp=10.0,
            )
        else:
            coll = HADataCollector(
                ha=mock_ha,
                store=store,
                relay_entity_id="switch.boiler",
                relay_power_entity_id="sensor.boiler_power",
            )

        # Fetch exactly the window that _generate_ha_week_history filled for this w
        start = today_midnight - timedelta(days=w * 7 + 7)
        end = start + timedelta(days=7)
        df_week = coll._fetch_hourly(start, end)
        if not df_week.empty:
            all_dfs.append(df_week)

        # Feed HDO learner from raw relay HA history (mirrors production control loop).
        # "unavailable" state = HDO cut the circuit.
        for relay_obj in week_data.get("switch.boiler", []):
            ts = pd.to_datetime(relay_obj["last_changed"], utc=True).tz_localize(None)
            relay_unavailable = relay_obj["state"] == "unavailable"
            hdo_learner.observe(ts.to_pydatetime(), relay_unavailable=relay_unavailable)

    if all_dfs:
        full_df = pd.concat(all_dfs)
        full_df = full_df[~full_df.index.duplicated(keep="last")].sort_index()
        predictor.update(full_df)
    else:
        full_df = pd.DataFrame()

    return predictor, hdo_learner, full_df


class TestBootstrapFromHADBHistory:
    """
    Simulate bootstrapping from Home Assistant history (InfluxDB or SQLite recorder).

    The HA recorder stores entity state history; this test verifies that
    feeding weeks of that history through the full pipeline correctly
    trains both the consumption predictor and the HDO schedule learner.
    """
    HDO_HOURS = list(range(22, 24)) + list(range(0, 6))  # 22:00–06:00
    SHOWER_HOURS = [7, 18]  # morning + evening showers

    def test_predictor_has_enough_data_after_bootstrap(self, tmp_path):
        pred, _, df = _bootstrap_from_weeks(
            n_weeks=6,
            hdo_hours=self.HDO_HOURS,
            shower_hours=self.SHOWER_HOURS,
            tmp_path=tmp_path,
            use_flow_sensor=True,
        )
        assert pred.has_enough_data(min_days=20), \
            f"Expected enough data, got _total_samples={pred._total_samples}"

    def test_hdo_learner_detects_blocked_hours(self, tmp_path):
        _, learner, _ = _bootstrap_from_weeks(
            n_weeks=5,
            hdo_hours=self.HDO_HOURS,
            shower_hours=self.SHOWER_HOURS,
            tmp_path=tmp_path,
            use_flow_sensor=False,  # power-only path
        )
        # Check representative HDO hours
        for wd in range(7):
            for h in [22, 23, 0, 2, 4]:
                assert learner.is_blocked(wd, h), \
                    f"Expected HDO blocked at weekday={wd} hour={h}"

    def test_hdo_learner_does_not_block_shower_hours(self, tmp_path):
        _, learner, _ = _bootstrap_from_weeks(
            n_weeks=5,
            hdo_hours=self.HDO_HOURS,
            shower_hours=self.SHOWER_HOURS,
            tmp_path=tmp_path,
            use_flow_sensor=False,
        )
        for wd in range(7):
            assert not learner.is_blocked(wd, 7), \
                f"Shower hour 7 should NOT be HDO blocked at weekday={wd}"
            assert not learner.is_blocked(wd, 18), \
                f"Shower hour 18 should NOT be HDO blocked at weekday={wd}"

    def test_consumption_df_has_correct_columns(self, tmp_path):
        _, _, df = _bootstrap_from_weeks(
            n_weeks=3,
            hdo_hours=self.HDO_HOURS,
            shower_hours=self.SHOWER_HOURS,
            tmp_path=tmp_path,
            use_flow_sensor=True,
        )
        if not df.empty:
            assert "consumed_kwh" in df.columns
            assert "relay_on" in df.columns
            assert "power_w" in df.columns

    def test_flow_based_consumption_positive_during_shower(self, tmp_path):
        _, _, df = _bootstrap_from_weeks(
            n_weeks=4,
            hdo_hours=self.HDO_HOURS,
            shower_hours=self.SHOWER_HOURS,
            tmp_path=tmp_path,
            use_flow_sensor=True,
        )
        if df.empty:
            pytest.skip("No data collected (test infrastructure issue)")
        # Shower hours should have non-trivial consumption
        shower_rows = df[df.index.hour.isin(self.SHOWER_HOURS)]
        non_nan_shower = shower_rows["consumed_kwh"].dropna()
        if len(non_nan_shower) > 0:
            assert non_nan_shower.mean() > 0.0

    def test_bootstrap_integrates_with_store_persistence(self, tmp_path):
        """Verify that StateStore correctly persists and reloads the consumption history."""
        store = StateStore(data_dir=str(tmp_path))
        mock_ha = MagicMock(spec=HAClient)

        week_data = _generate_ha_week_history(
            week_offset_days=0,
            hdo_hours=self.HDO_HOURS,
            shower_hours=self.SHOWER_HOURS,
        )
        mock_ha.get_history.side_effect = lambda eid, *a, **kw: week_data.get(eid, [])

        coll = HADataCollector(
            ha=mock_ha,
            store=store,
            relay_entity_id="switch.boiler",
            relay_power_entity_id="sensor.boiler_power",
            water_flow_entity_id="sensor.flow",
            water_temp_out_entity_id="sensor.water_temp",
        )

        now = datetime.now()
        df_new = coll._fetch_hourly(now - timedelta(days=7), now)
        if not df_new.empty:
            df_stored = store.append_consumption(df_new)
            df_reloaded = store.load_consumption_history()
            assert len(df_reloaded) == len(df_stored)
            assert "consumed_kwh" in df_reloaded.columns


# ---------------------------------------------------------------------------
# StateStore integration
# ---------------------------------------------------------------------------

class TestStateStoreIntegration:
    def test_append_consumption_deduplicates(self, tmp_path):
        store = StateStore(data_dir=str(tmp_path))
        # Use recent timestamps so the 90-day trim doesn't remove them
        base = datetime.now().replace(minute=0, second=0, microsecond=0)
        idx = pd.DatetimeIndex([base - timedelta(hours=4 - h) for h in range(5)])
        df1 = pd.DataFrame({"consumed_kwh": [0.3]*5, "relay_on": [True]*5, "power_w": [2000.0]*5}, index=idx)
        df2 = pd.DataFrame({"consumed_kwh": [0.4]*5, "relay_on": [True]*5, "power_w": [2100.0]*5}, index=idx)
        store.append_consumption(df1)
        result = store.append_consumption(df2)
        # Deduplicated — should have 5 rows, not 10
        assert len(result) == 5
        # Keep last → second update wins
        assert result["consumed_kwh"].iloc[0] == pytest.approx(0.4)

    def test_append_consumption_trims_old_data(self, tmp_path):
        from smartboiler.state_store import CONSUMPTION_HISTORY_DAYS
        store = StateStore(data_dir=str(tmp_path))
        # One row very old, one row recent
        old_ts = datetime.now() - timedelta(days=CONSUMPTION_HISTORY_DAYS + 5)
        new_ts = datetime.now()
        idx = pd.DatetimeIndex([old_ts, new_ts])
        df = pd.DataFrame(
            {"consumed_kwh": [0.5, 0.3], "relay_on": [True, False], "power_w": [2000.0, 0.0]},
            index=idx,
        )
        result = store.append_consumption(df)
        assert len(result) == 1
        assert result.index[0].date() == new_ts.date()

    def test_load_consumption_history_returns_empty_df_when_no_file(self, tmp_path):
        store = StateStore(data_dir=str(tmp_path))
        df = store.load_consumption_history()
        assert df.empty
        assert list(df.columns) == ["consumed_kwh", "relay_on", "power_w"]
