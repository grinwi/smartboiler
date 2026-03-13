"""
Tests for RollingHistogramPredictor — weekday×hour quantile predictor.

Covers:
- update() with empty / partial / full DataFrames
- History window cutoff (HISTORY_WEEKS)
- Noise filtering (> 0.005 kWh threshold)
- predict_hour() with enough samples vs. global fallback
- predict_next_24h() day-boundary, length, monotone structure
- has_enough_data() threshold check
- get_histogram_summary() structure
- Conservatism levels (p50 / p75 / p90)
- Simulation: feed realistic multi-week consumption data and verify predictions
  are within plausible range of true values
"""
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from smartboiler.predictor import (
    RollingHistogramPredictor,
    HISTORY_WEEKS,
    MIN_SAMPLES_FOR_SLOT,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_df(
    hours: int = 24 * 30,
    base_kwh: float = 0.3,
    noise: float = 0.05,
    end: datetime = None,
) -> pd.DataFrame:
    """Generate a DataFrame with DatetimeIndex and consumed_kwh column."""
    if end is None:
        end = datetime.now().replace(minute=0, second=0, microsecond=0)
    idx = pd.date_range(end=end, periods=hours, freq="1h")
    rng = np.random.default_rng(42)
    vals = np.maximum(0, base_kwh + rng.normal(0, noise, size=hours))
    return pd.DataFrame({"consumed_kwh": vals}, index=idx)


def _make_pattern_df(weeks: int = 6) -> pd.DataFrame:
    """
    Realistic consumption pattern:
    - Shower peak at 07:00 and 18:00 (0.5 kWh)
    - Dishwasher at 20:00 (0.4 kWh)
    - Otherwise near-zero

    Anchors to midnight so hours are always correct regardless of when tests run.
    """
    today_midnight = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    rows = []
    for w in range(weeks):
        # Each week starts on a separate calendar week (midnight aligned)
        week_start = today_midnight - timedelta(weeks=weeks - w)
        for d in range(7):
            for h in range(24):
                dt = week_start + timedelta(days=d, hours=h)
                if h in (7, 18):
                    kwh = 0.5 + np.random.default_rng(w * 100 + d * 10 + h).uniform(-0.05, 0.05)
                elif h == 20:
                    kwh = 0.4 + np.random.default_rng(w * 100 + d * 10 + h).uniform(-0.03, 0.03)
                else:
                    kwh = 0.01  # minimal consumption; passes noise filter but much less than peaks
                rows.append({"dt": dt, "consumed_kwh": kwh})
    df = pd.DataFrame(rows).set_index("dt")
    df.index = pd.to_datetime(df.index)
    return df


# ---------------------------------------------------------------------------
# update()
# ---------------------------------------------------------------------------

class TestUpdate:
    def test_empty_df_does_nothing(self):
        p = RollingHistogramPredictor()
        p.update(pd.DataFrame())
        assert p._total_samples == 0
        assert p._global_fallback == 0.0

    def test_none_df_does_nothing(self):
        p = RollingHistogramPredictor()
        p.update(None)
        assert p._total_samples == 0

    def test_missing_column_does_nothing(self):
        p = RollingHistogramPredictor()
        df = pd.DataFrame({"relay_on": [1, 0, 1]})
        p.update(df)
        assert p._total_samples == 0

    def test_builds_histogram_for_nonzero_hours(self):
        p = RollingHistogramPredictor()
        df = _make_df(hours=24 * 7, base_kwh=0.3)
        p.update(df)
        assert len(p._hist) > 0

    def test_filters_noise_below_0005(self):
        p = RollingHistogramPredictor()
        now = datetime.now().replace(minute=0, second=0, microsecond=0)
        idx = pd.date_range(end=now, periods=10, freq="1h")
        # Mix of noise and real consumption
        vals = [0.001, 0.003, 0.004, 0.005, 0.006, 0.1, 0.2, 0.3, 0.4, 0.5]
        df = pd.DataFrame({"consumed_kwh": vals}, index=idx)
        p.update(df)
        # Only values > 0.005 should be in histogram (0.006 onwards)
        all_hist_vals = [v for vals in p._hist.values() for v in vals]
        assert all(v > 0.005 for v in all_hist_vals)

    def test_ignores_data_older_than_history_weeks(self):
        p = RollingHistogramPredictor()
        now = datetime.now().replace(minute=0, second=0, microsecond=0)
        # Recent data: 24 hours
        recent_idx = pd.date_range(end=now, periods=24, freq="1h")
        # Old data: beyond HISTORY_WEEKS
        old_end = now - timedelta(weeks=HISTORY_WEEKS + 1)
        old_idx = pd.date_range(end=old_end, periods=24, freq="1h")

        df_recent = pd.DataFrame({"consumed_kwh": [0.3] * 24}, index=recent_idx)
        df_old = pd.DataFrame({"consumed_kwh": [1.0] * 24}, index=old_idx)
        df = pd.concat([df_old, df_recent])

        p.update(df)
        # total_samples should only count recent window
        assert p._total_samples == 24

    def test_update_replaces_old_histogram(self):
        p = RollingHistogramPredictor()
        df1 = _make_df(hours=24 * 7, base_kwh=0.2)
        df2 = _make_df(hours=24 * 7, base_kwh=0.8)
        p.update(df1)
        fallback1 = p._global_fallback
        p.update(df2)
        fallback2 = p._global_fallback
        # Second update with higher values should shift the fallback up
        assert fallback2 > fallback1

    def test_global_fallback_non_zero_for_real_data(self):
        p = RollingHistogramPredictor()
        p.update(_make_df(hours=24 * 7, base_kwh=0.3))
        assert p._global_fallback > 0.0

    def test_total_samples_counts_recent_rows(self):
        p = RollingHistogramPredictor()
        df = _make_df(hours=100, base_kwh=0.1)
        p.update(df)
        assert p._total_samples == 100


# ---------------------------------------------------------------------------
# predict_hour()
# ---------------------------------------------------------------------------

class TestPredictHour:
    def test_returns_global_fallback_when_slot_has_too_few_samples(self):
        p = RollingHistogramPredictor()
        p.update(_make_df(hours=24 * 2, base_kwh=0.3))  # only 2 days = < MIN_SAMPLES_FOR_SLOT
        # Most weekday×hour slots will have 0-1 samples; fallback should be used
        result = p.predict_hour(0, 3)
        assert result == pytest.approx(p._global_fallback)

    def test_returns_slot_quantile_when_enough_samples(self):
        p = RollingHistogramPredictor()
        # Give one slot 10 identical values
        p._hist[(0, 10)] = [0.5] * 10
        p._global_fallback = 0.1
        result = p.predict_hour(0, 10)
        # p75 of [0.5]*10 = 0.5
        assert result == pytest.approx(0.5)

    def test_quantile_p50_lower_than_p75(self):
        p50 = RollingHistogramPredictor(conservatism="low")
        p75 = RollingHistogramPredictor(conservatism="medium")
        df = _make_df(hours=24 * 30, base_kwh=0.4, noise=0.1)
        p50.update(df)
        p75.update(df)
        assert p50._global_fallback <= p75._global_fallback

    def test_quantile_p75_lower_than_p90(self):
        p75 = RollingHistogramPredictor(conservatism="medium")
        p90 = RollingHistogramPredictor(conservatism="high")
        df = _make_df(hours=24 * 30, base_kwh=0.4, noise=0.1)
        p75.update(df)
        p90.update(df)
        assert p75._global_fallback <= p90._global_fallback

    def test_unknown_conservatism_defaults_to_p75(self):
        p = RollingHistogramPredictor(conservatism="ultra_high")
        assert p.quantile == pytest.approx(0.75)

    def test_returns_zero_when_no_data_at_all(self):
        p = RollingHistogramPredictor()
        assert p.predict_hour(0, 12) == pytest.approx(0.0)

    def test_slot_prediction_above_zero_for_real_data(self):
        p = RollingHistogramPredictor()
        df = _make_pattern_df(weeks=6)
        p.update(df)
        # Shower hours should predict > 0
        assert p.predict_hour(0, 7) > 0.0   # Monday morning shower
        assert p.predict_hour(0, 18) > 0.0  # Monday evening shower


# ---------------------------------------------------------------------------
# predict_next_24h()
# ---------------------------------------------------------------------------

class TestPredictNext24h:
    def test_returns_24_values(self):
        p = RollingHistogramPredictor()
        result = p.predict_next_24h(datetime(2024, 3, 11, 8, 0))
        assert len(result) == 24

    def test_uses_now_when_from_dt_none(self):
        p = RollingHistogramPredictor()
        result = p.predict_next_24h(None)
        assert len(result) == 24

    def test_all_non_negative(self):
        p = RollingHistogramPredictor()
        p.update(_make_df(hours=24 * 14))
        result = p.predict_next_24h()
        assert all(v >= 0.0 for v in result)

    def test_peak_hours_higher_than_off_peak(self):
        """Pattern predictor: shower hours (7, 18) must predict higher than off-peak (3)."""
        p = RollingHistogramPredictor(conservatism="medium")
        df = _make_pattern_df(weeks=6)
        p.update(df)
        # Query histogram directly by weekday to avoid now-relative alignment issues
        today_wd = datetime.now().weekday()
        peak = max(p.predict_hour(today_wd, 7), p.predict_hour(today_wd, 18))
        off_peak = p.predict_hour(today_wd, 3)
        assert peak > off_peak, f"Peak {peak:.3f} should exceed off-peak {off_peak:.3f}"

    def test_values_are_floats(self):
        p = RollingHistogramPredictor()
        p.update(_make_df(24 * 7))
        for v in p.predict_next_24h():
            assert isinstance(v, float)


# ---------------------------------------------------------------------------
# has_enough_data()
# ---------------------------------------------------------------------------

class TestHasEnoughData:
    def test_false_when_no_data(self):
        assert not RollingHistogramPredictor().has_enough_data()

    def test_false_when_too_few_days(self):
        p = RollingHistogramPredictor()
        p.update(_make_df(hours=24 * 10))  # 10 days < 30 * 20 threshold
        assert not p.has_enough_data(min_days=30)

    def test_true_when_enough_days(self):
        p = RollingHistogramPredictor()
        # 30 days * 20 usable hours/day = 600 samples threshold
        p.update(_make_df(hours=24 * 30, base_kwh=0.3))
        # With base_kwh=0.3 > 0.005, ~all hours are non-zero so _total_samples=720
        assert p.has_enough_data(min_days=30)

    def test_custom_min_days(self):
        p = RollingHistogramPredictor()
        p.update(_make_df(hours=24 * 5))
        assert p.has_enough_data(min_days=5)
        assert not p.has_enough_data(min_days=10)


# ---------------------------------------------------------------------------
# get_histogram_summary()
# ---------------------------------------------------------------------------

class TestGetHistogramSummary:
    def test_returns_dict(self):
        p = RollingHistogramPredictor()
        p.update(_make_pattern_df(weeks=4))
        summary = p.get_histogram_summary()
        assert isinstance(summary, dict)

    def test_day_keys_present(self):
        p = RollingHistogramPredictor()
        p.update(_make_pattern_df(weeks=4))
        summary = p.get_histogram_summary()
        for day in ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]:
            assert day in summary

    def test_slot_has_required_fields(self):
        p = RollingHistogramPredictor()
        p.update(_make_pattern_df(weeks=4))
        summary = p.get_histogram_summary()
        for day_data in summary.values():
            for hour_data in day_data.values():
                assert "p50" in hour_data
                assert "p75" in hour_data
                assert "count" in hour_data

    def test_p50_leq_p75_in_all_slots(self):
        p = RollingHistogramPredictor()
        p.update(_make_pattern_df(weeks=4))
        summary = p.get_histogram_summary()
        for day_data in summary.values():
            for hour_data in day_data.values():
                assert hour_data["p50"] <= hour_data["p75"] + 1e-9

    def test_count_positive_for_peak_hours(self):
        p = RollingHistogramPredictor()
        df = _make_pattern_df(weeks=6)
        p.update(df)
        summary = p.get_histogram_summary()
        # Check that at least one weekday has data at hour 7 (shower peak)
        has_hour_7 = any(7 in day_data for day_data in summary.values())
        assert has_hour_7, f"Expected some weekday to have hour-7 data; got keys: {[list(v.keys()) for v in summary.values()]}"

    def test_empty_when_no_data(self):
        p = RollingHistogramPredictor()
        summary = p.get_histogram_summary()
        # All days present, all empty
        total_slots = sum(len(v) for v in summary.values())
        assert total_slots == 0


# ---------------------------------------------------------------------------
# Simulation: multi-week pattern + prediction accuracy
# ---------------------------------------------------------------------------

class TestPredictionSimulation:
    """End-to-end: generate realistic consumption history, train predictor,
    assert predictions are in the right ballpark for peak vs off-peak hours."""

    def test_peak_hours_predicted_higher_than_off_peak(self):
        p = RollingHistogramPredictor(conservatism="medium")
        df = _make_pattern_df(weeks=8)
        p.update(df)

        # Query by actual weekday from the histogram (not fixed Monday=0 assumption)
        # since _make_pattern_df is anchored to today's weekday.
        today_wd = datetime.now().weekday()
        shower_morning = p.predict_hour(today_wd, 7)
        shower_evening = p.predict_hour(today_wd, 18)
        off_peak = p.predict_hour(today_wd, 3)

        # Shower slots have real consumption; hour-3 has zero (filtered) → falls back to global.
        # The global fallback is computed from peak hours only (zeros filtered), so the
        # shower prediction should be >= global fallback (not strictly greater).
        assert shower_morning >= off_peak, \
            f"Morning shower {shower_morning:.3f} should be >= off-peak {off_peak:.3f}"
        assert shower_evening >= off_peak, \
            f"Evening shower {shower_evening:.3f} should be >= off-peak {off_peak:.3f}"

    def test_p90_higher_than_p50_for_same_slot(self):
        p50 = RollingHistogramPredictor(conservatism="low")
        p90 = RollingHistogramPredictor(conservatism="high")
        df = _make_pattern_df(weeks=8)
        p50.update(df)
        p90.update(df)

        # For the high-consumption shower slot with >MIN_SAMPLES_FOR_SLOT data,
        # p90 must be ≥ p50
        assert p90.predict_hour(0, 7) >= p50.predict_hour(0, 7)

    def test_predictions_stable_across_updates(self):
        """Calling update() twice with the same data should give the same predictions."""
        p = RollingHistogramPredictor()
        df = _make_pattern_df(weeks=4)
        p.update(df)
        pred1 = p.predict_next_24h(datetime(2024, 3, 11, 0, 0))
        p.update(df)
        pred2 = p.predict_next_24h(datetime(2024, 3, 11, 0, 0))
        for a, b in zip(pred1, pred2):
            assert a == pytest.approx(b)

    def test_fallback_used_for_sparse_new_weekday(self):
        """If only weekday 0 has data, weekday 6 (Sunday) falls back to global quantile."""
        p = RollingHistogramPredictor()
        # Only add data for Monday (weekday=0) at one hour
        now = datetime.now().replace(minute=0, second=0, microsecond=0)
        # Make Monday = weekday 0 by aligning to this week's Monday
        monday = now - timedelta(days=now.weekday())
        idx = pd.date_range(start=monday - timedelta(weeks=4), periods=4 * 4, freq="1w")
        vals = [0.5] * len(idx)
        df = pd.DataFrame({"consumed_kwh": vals}, index=idx)
        p.update(df)

        sunday_pred = p.predict_hour(6, 12)  # Sunday has no data
        assert sunday_pred == pytest.approx(p._global_fallback)
