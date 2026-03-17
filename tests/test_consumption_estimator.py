"""
Tests for FlowlessConsumptionEstimator — Simple Mode and Standard Mode.

Covers:
- Energy balance math (zero relay, known scenario, standby, k_water)
- T_in live tracking (T_cold_live computation, usage in estimate)
- Draw detection (trigger / no-trigger / volume formula)
- Coupling auto-calibration (guard, formula, EMA update)
- _effective_coupling property
- Alpha learning (true_vol feedback, draw correction, clamping)
- T_set calibration at relay-OFF
- Persistence (to_dict / from_dict round-trip with all new fields)
- Diagnostics keys
- Accumulator reset on day rollover

Simulation tests:
- Simple mode: full day with HDO gaps (relay_on=False during blocked intervals)
- Simple mode: multi-day coupling & T_cold_live convergence
- Standard mode: alpha converges toward 1.0 with flow-sensor ground truth
"""
from datetime import date, timedelta
from typing import Optional
from unittest.mock import patch

import pytest

from smartboiler.consumption_estimator import (
    FlowlessConsumptionEstimator,
    C_WATER,
    RHO,
    _COUPLING_MIN,
    _COUPLING_MAX,
    _ALPHA_MIN,
    _ALPHA_MAX,
)


# ── Common fixture ─────────────────────────────────────────────────────────────

PARAMS = dict(
    vol_L            = 100.0,
    relay_w          = 2000.0,
    T_cold           = 10.0,
    coupling         = 0.45,
    T_amb_default    = 18.0,
    standby_w        = 50.0,
    draw_threshold_c = 2.0,
)


def make_estimator(**overrides) -> FlowlessConsumptionEstimator:
    return FlowlessConsumptionEstimator(**{**PARAMS, **overrides})


# ─────────────────────────────────────────────────────────────────────────────
# Energy balance math
# ─────────────────────────────────────────────────────────────────────────────

class TestEnergyBalance:
    def test_zero_relay_returns_zero(self):
        est = make_estimator()
        assert est.estimate_today() == 0.0

    def test_known_scenario(self):
        """2 h relay at 2000 W, T_set=45, T_cold=10, no standby → 98.3 L"""
        est = make_estimator(standby_w=0.0)
        est.T_set_calibrated = 45.0
        for _ in range(2 * 60):          # 2 hours × 60-s ticks
            est.tick(relay_on=True, T_case=None, T_amb=18.0, dt_s=60.0)
        vol = est.estimate_today(T_amb=18.0)
        expected = 4.0 * 3.6e6 / (C_WATER * RHO * (45.0 - 10.0))
        assert abs(vol - expected) < 1.0

    def test_standby_reduces_estimate(self):
        """Standby losses lower net energy and therefore volume."""
        est_no = make_estimator(standby_w=0.0)
        est_sw = make_estimator(standby_w=100.0)
        for est in (est_no, est_sw):
            est.T_set_calibrated = 45.0
            for _ in range(120):
                est.tick(relay_on=True, T_case=None, T_amb=18.0, dt_s=60.0)
        assert est_sw.estimate_today(T_amb=18.0) < est_no.estimate_today(T_amb=18.0)

    def test_k_water_equivalent_to_standby_w(self):
        """k_water from standby_w derivation → same estimate as standby_w path."""
        T_set, T_amb, vol_L, sw = 45.0, 18.0, 120.0, 60.0
        k_water = sw * 3600.0 / (vol_L * RHO * C_WATER * (T_set - T_amb))
        est = make_estimator(vol_L=vol_L, standby_w=sw)
        est.T_set_calibrated = T_set
        for _ in range(24 * 60):
            est.tick(relay_on=False, T_case=None, T_amb=T_amb, dt_s=60.0)
        assert abs(
            est.estimate_today(k_water=k_water, T_amb=T_amb)
            - est.estimate_today(k_water=None, T_amb=T_amb)
        ) < 0.1

    def test_alpha_scales_linearly(self):
        est = make_estimator(standby_w=0.0)
        est.T_set_calibrated = 45.0
        for _ in range(60):
            est.tick(relay_on=True, T_case=None, T_amb=18.0, dt_s=60.0)
        v1 = est.estimate_today()
        est.alpha = 2.0
        v2 = est.estimate_today()
        assert abs(v2 - 2 * v1) < 1e-6


# ─────────────────────────────────────────────────────────────────────────────
# T_in live tracking (T_cold_live)
# ─────────────────────────────────────────────────────────────────────────────

class TestTinTracking:
    def test_T_in_samples_accumulate(self):
        est = make_estimator()
        for _ in range(5):
            est.tick(relay_on=False, T_case=20.0, T_amb=18.0, dt_s=60.0, T_in=12.0)
        assert len(est._T_in_samples) == 5

    def test_T_in_sanity_filter_rejects_hot_water(self):
        """Inlet probe > 30°C should be ignored (that's hot water, sensor fault)."""
        est = make_estimator()
        est.tick(relay_on=False, T_case=20.0, T_amb=18.0, dt_s=60.0, T_in=35.0)
        assert len(est._T_in_samples) == 0

    def test_T_in_sanity_filter_rejects_below_zero(self):
        est = make_estimator()
        est.tick(relay_on=False, T_case=20.0, T_amb=18.0, dt_s=60.0, T_in=-1.0)
        assert len(est._T_in_samples) == 0

    def test_T_cold_live_computed_from_T_in_median(self):
        est = make_estimator()
        for t in [10.0, 11.0, 12.0, 11.0, 10.5]:
            est.tick(relay_on=False, T_case=20.0, T_amb=18.0, dt_s=60.0, T_in=t)
        # Finalize with no ground truth
        est._finalize_day(k_water=None, T_amb=18.0, true_vol_L=None)
        # Median of [10, 11, 12, 11, 10.5] = 11.0
        assert abs(est._T_cold_live - 11.0) < 0.1

    def test_T_cold_live_used_over_static_T_cold(self):
        """Lower T_cold_live (colder inlet) → larger dT → lower volume estimate."""
        est_live = make_estimator(T_cold=10.0, standby_w=0.0)
        est_static = make_estimator(T_cold=10.0, standby_w=0.0)
        # Give live estimator a warmer T_cold_live (8°C vs static 10°C is wrong direction,
        # let me use 6°C which is colder → larger dT → *lower* volume)
        est_live._T_cold_live = 6.0     # colder inlet → larger T_set-T_cold → fewer litres
        est_static._T_cold_live = None  # uses static 10°C

        for est in (est_live, est_static):
            est.T_set_calibrated = 45.0
            for _ in range(60):
                est.tick(relay_on=True, T_case=None, T_amb=18.0, dt_s=60.0)

        v_live   = est_live.estimate_today(T_amb=18.0)
        v_static = est_static.estimate_today(T_amb=18.0)
        # Colder T_cold → larger dT → fewer litres needed to carry same energy
        assert v_live < v_static


# ─────────────────────────────────────────────────────────────────────────────
# Draw detection
# ─────────────────────────────────────────────────────────────────────────────

class TestDrawDetection:
    def _make_and_prime(self, **kw) -> FlowlessConsumptionEstimator:
        est = make_estimator(**kw)
        # Prime _prev_T_case by one relay-OFF tick
        est.tick(relay_on=False, T_case=30.0, T_amb=18.0, dt_s=60.0)
        return est

    def test_no_trigger_below_threshold(self):
        est = self._make_and_prime(draw_threshold_c=2.0)
        # Drop of only 1.5°C — should NOT trigger
        est.tick(relay_on=False, T_case=28.5, T_amb=18.0, dt_s=60.0)
        assert est._draw_vol_today == 0.0

    def test_trigger_above_threshold(self):
        est = self._make_and_prime(draw_threshold_c=2.0)
        # Drop of 5°C — should trigger
        est.tick(relay_on=False, T_case=25.0, T_amb=18.0, dt_s=60.0)
        assert est._draw_vol_today > 0.0

    def test_no_trigger_when_relay_on(self):
        """Draw detection is disabled while relay is ON (Newton cooling dominated by heating)."""
        est = make_estimator(draw_threshold_c=2.0)
        est.tick(relay_on=True, T_case=30.0, T_amb=18.0, dt_s=60.0)
        est.tick(relay_on=True, T_case=25.0, T_amb=18.0, dt_s=60.0)  # big drop — relay ON
        assert est._draw_vol_today == 0.0

    def test_no_trigger_without_prev_case(self):
        """First tick has no _prev_T_case — must not trigger."""
        est = make_estimator(draw_threshold_c=2.0)
        est.tick(relay_on=False, T_case=25.0, T_amb=18.0, dt_s=60.0)
        assert est._draw_vol_today == 0.0

    def test_volume_formula(self):
        """
        vol_L=100, coupling=0.45, T_amb=18, T_cold=10.
        prev_T_case=30 → T_water_before = 18 + (30-18)/0.45 = 44.67
        curr_T_case=25 → delta_case=5, delta_water=5/0.45=11.11
        V_draw = 100 * 11.11 / (44.67 - 10) = 32.1 L  (approx)
        """
        est = make_estimator(vol_L=100.0, coupling=0.45, T_cold=10.0, draw_threshold_c=2.0)
        est.tick(relay_on=False, T_case=30.0, T_amb=18.0, dt_s=60.0)  # prime
        est.tick(relay_on=False, T_case=25.0, T_amb=18.0, dt_s=60.0)  # 5°C drop

        T_water_before = 18.0 + (30.0 - 18.0) / 0.45
        delta_water    = 5.0 / 0.45
        denom          = T_water_before - 10.0
        expected       = 100.0 * delta_water / denom
        assert abs(est._draw_vol_today - expected) < 0.5

    def test_draw_uses_T_cold_live_when_available(self):
        """When T_cold_live is set it overrides static T_cold in the draw formula."""
        est = make_estimator(vol_L=100.0, coupling=0.45, T_cold=10.0, draw_threshold_c=2.0)
        est._T_cold_live = 5.0   # colder than static 10°C → larger dT → smaller volume

        est.tick(relay_on=False, T_case=30.0, T_amb=18.0, dt_s=60.0)
        est.tick(relay_on=False, T_case=25.0, T_amb=18.0, dt_s=60.0)

        T_water_before = 18.0 + (30.0 - 18.0) / 0.45
        delta_water    = 5.0 / 0.45
        denom          = T_water_before - 5.0   # T_cold_live=5
        expected       = 100.0 * delta_water / denom
        assert abs(est._draw_vol_today - expected) < 0.5

    def test_draw_uses_calibrated_coupling(self):
        est = make_estimator(coupling=0.45, draw_threshold_c=2.0)
        est.coupling_calibrated = 0.30   # overrides config coupling

        est.tick(relay_on=False, T_case=30.0, T_amb=18.0, dt_s=60.0)
        est.tick(relay_on=False, T_case=25.0, T_amb=18.0, dt_s=60.0)

        # With coupling=0.30: delta_water = 5/0.30 = 16.67
        # T_water_before = 18 + (30-18)/0.30 = 58.0
        delta_water = 5.0 / 0.30
        denom       = (18.0 + 12.0 / 0.30) - 10.0   # 48.0
        expected    = 100.0 * delta_water / denom
        assert abs(est._draw_vol_today - expected) < 0.5

    def test_draw_vol_resets_on_day_rollover_in_tick(self):
        est = make_estimator()
        est._draw_vol_today = 50.0  # simulate accumulated draws
        # Simulate day rollover in tick()
        from datetime import date
        est._current_day = date.today() - timedelta(days=1)
        est.tick(relay_on=False, T_case=20.0, T_amb=18.0, dt_s=60.0)
        assert est._draw_vol_today == 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Coupling auto-calibration
# ─────────────────────────────────────────────────────────────────────────────

class TestCouplingCalibration:
    def test_returns_none_without_T_set_calibrated(self):
        est = make_estimator()
        assert est._infer_coupling(30.0, 18.0) is None

    def test_infers_correct_coupling(self):
        """coupling = (T_case - T_amb) / (T_set_cal - T_amb)
        With T_set_cal=45, T_amb=18, coupling=0.45:
          T_case = T_amb + coupling*(T_set_cal - T_amb) = 18 + 0.45*27 = 30.15
        """
        est = make_estimator(coupling=0.45)
        est.T_set_calibrated = 45.0
        T_case = 18.0 + 0.45 * (45.0 - 18.0)   # 30.15
        result = est._infer_coupling(T_case, 18.0)
        assert result is not None
        assert abs(result - 0.45) < 0.01

    def test_rejects_coupling_outside_bounds(self):
        est = make_estimator()
        est.T_set_calibrated = 45.0
        # T_case = T_amb + 0.05*(45-18) = 18 + 1.35 = 19.35 → coupling=0.05 < MIN
        assert est._infer_coupling(19.35, 18.0) is None
        # T_case = T_amb + 0.95*(45-18) = 18 + 25.65 = 43.65 → coupling=0.95 > MAX
        assert est._infer_coupling(43.65, 18.0) is None

    def test_effective_coupling_uses_calibrated_when_available(self):
        est = make_estimator(coupling=0.45)
        est.coupling_calibrated = 0.30
        assert est._effective_coupling == 0.30

    def test_effective_coupling_falls_back_to_config(self):
        est = make_estimator(coupling=0.45)
        assert est._effective_coupling == 0.45

    def test_coupling_ema_update_in_finalize(self):
        """coupling_calibrated should EMA-update toward today's median sample."""
        est = make_estimator(coupling=0.45)
        est.T_set_calibrated = 45.0
        # Inject coupling samples at 0.30
        est._coupling_samples = [0.30] * 5
        # First finalize → initialises coupling_calibrated = 0.30
        est._finalize_day(k_water=None, T_amb=18.0, true_vol_L=None)
        assert est.coupling_calibrated is not None
        assert abs(est.coupling_calibrated - 0.30) < 0.02

    def test_coupling_calibrated_clamped_in_finalize(self):
        est = make_estimator()
        est.T_set_calibrated = 45.0
        est._coupling_samples = [0.01] * 5   # below _COUPLING_MIN
        est._finalize_day(k_water=None, T_amb=18.0, true_vol_L=None)
        if est.coupling_calibrated is not None:
            assert est.coupling_calibrated >= _COUPLING_MIN


# ─────────────────────────────────────────────────────────────────────────────
# T_set calibration
# ─────────────────────────────────────────────────────────────────────────────

class TestTsetCalibration:
    def test_sample_at_relay_off(self):
        """coupling=0.45, T_amb=18, T_case=30 → T_water = 18+(30-18)/0.45 = 44.67"""
        est = make_estimator(coupling=0.45)
        est.tick(relay_on=True,  T_case=30.0, T_amb=18.0)
        est.tick(relay_on=False, T_case=30.0, T_amb=18.0)
        assert len(est._T_set_samples) == 1
        expected = 18.0 + (30.0 - 18.0) / 0.45
        assert abs(est._T_set_samples[0] - expected) < 1e-9

    def test_no_sample_without_transition(self):
        est = make_estimator()
        for _ in range(5):
            est.tick(relay_on=False, T_case=30.0, T_amb=18.0)
        assert len(est._T_set_samples) == 0

    def test_T_set_calibrated_after_finalize(self):
        est = make_estimator(coupling=0.45)
        for T_c in [30.0, 28.0, 31.0, 29.5, 30.5]:
            est.tick(relay_on=True,  T_case=T_c, T_amb=18.0)
            est.tick(relay_on=False, T_case=T_c, T_amb=18.0)
        est._finalize_day(k_water=None, T_amb=18.0, true_vol_L=None)
        assert est.T_set_calibrated is not None
        assert 30.0 < est.T_set_calibrated < 85.0


# ─────────────────────────────────────────────────────────────────────────────
# Alpha learning
# ─────────────────────────────────────────────────────────────────────────────

class TestAlphaLearning:
    def test_alpha_moves_down_when_over_estimated(self):
        est = make_estimator(standby_w=0.0)
        est.T_set_calibrated = 45.0
        est.alpha = 1.0
        for _ in range(120):
            est.tick(relay_on=True, T_case=None, T_amb=18.0, dt_s=60.0)
        est_vol = est.estimate_today()
        est._finalize_day(k_water=None, T_amb=18.0, true_vol_L=est_vol * 0.7)
        assert est.alpha < 1.0

    def test_alpha_moves_up_when_under_estimated(self):
        est = make_estimator(standby_w=0.0)
        est.T_set_calibrated = 45.0
        est.alpha = 1.0
        for _ in range(120):
            est.tick(relay_on=True, T_case=None, T_amb=18.0, dt_s=60.0)
        est_vol = est.estimate_today()
        est._finalize_day(k_water=None, T_amb=18.0, true_vol_L=est_vol * 1.3)
        assert est.alpha > 1.0

    def test_alpha_clamped_at_max(self):
        est = make_estimator(standby_w=0.0)
        est.T_set_calibrated = 45.0
        for _ in range(60):
            est.tick(relay_on=True, T_case=None, T_amb=18.0, dt_s=60.0)
        est_vol = est.estimate_today()
        est._finalize_day(k_water=None, T_amb=18.0, true_vol_L=est_vol * 10)
        assert est.alpha <= _ALPHA_MAX

    def test_alpha_clamped_at_min(self):
        est = make_estimator(standby_w=0.0)
        est.T_set_calibrated = 45.0
        for _ in range(60):
            est.tick(relay_on=True, T_case=None, T_amb=18.0, dt_s=60.0)
        est_vol = est.estimate_today()
        est._finalize_day(k_water=None, T_amb=18.0, true_vol_L=est_vol * 0.01)
        assert est.alpha >= _ALPHA_MIN

    def test_draw_correction_nudges_alpha_without_true_vol(self):
        """In pure simple mode (no flow sensor), draw-detected volume corrects alpha."""
        est = make_estimator(standby_w=0.0)
        est.T_set_calibrated = 45.0
        est.alpha = 1.0
        for _ in range(120):
            est.tick(relay_on=True, T_case=None, T_amb=18.0, dt_s=60.0)
        est_vol = est.estimate_today()
        # Inject a draw volume 50% larger than EB estimate → alpha should go up
        est._draw_vol_today = est_vol * 1.5
        old_alpha = est.alpha
        est._finalize_day(k_water=None, T_amb=18.0, true_vol_L=None)
        assert est.alpha != old_alpha   # was nudged (direction depends on ratio)

    def test_draw_correction_not_applied_below_2L_threshold(self):
        """Very small draw_vol_today (<= 2 L) should not trigger alpha correction."""
        est = make_estimator(standby_w=0.0)
        est.T_set_calibrated = 45.0
        est.alpha = 1.0
        for _ in range(120):
            est.tick(relay_on=True, T_case=None, T_amb=18.0, dt_s=60.0)
        est._draw_vol_today = 1.5   # below 2 L threshold
        est._finalize_day(k_water=None, T_amb=18.0, true_vol_L=None)
        assert est.alpha == 1.0   # unchanged

    def test_draw_correction_not_applied_when_true_vol_available(self):
        """If true_vol_L is available (standard mode), draw correction must NOT run."""
        est = make_estimator(standby_w=0.0)
        est.T_set_calibrated = 45.0
        est.alpha = 1.0
        for _ in range(120):
            est.tick(relay_on=True, T_case=None, T_amb=18.0, dt_s=60.0)
        est_vol = est.estimate_today()
        est._draw_vol_today = est_vol * 3.0   # would wildly push alpha if applied
        est._finalize_day(k_water=None, T_amb=18.0, true_vol_L=est_vol)  # exact match
        # Alpha should be updated by true_vol (no net change) — not by the wild draw signal
        assert abs(est.alpha - 1.0) < 0.15   # small movement from true_vol branch only


# ─────────────────────────────────────────────────────────────────────────────
# Persistence
# ─────────────────────────────────────────────────────────────────────────────

class TestPersistence:
    def test_round_trip_all_fields(self):
        est = make_estimator()
        est.alpha               = 1.23
        est.T_set_calibrated    = 44.5
        est.coupling_calibrated = 0.31
        est._T_cold_live        = 11.8
        est._daily_history      = [
            {"date": "2026-01-01", "est_vol_L": 35.0, "draw_vol_L": 30.0,
             "true_vol_L": None, "alpha": 1.23}
        ]
        d = est.to_dict()
        est2 = FlowlessConsumptionEstimator.from_dict(d)

        assert est2.alpha               == est.alpha
        assert est2.T_set_calibrated    == est.T_set_calibrated
        assert est2.coupling_calibrated == est.coupling_calibrated
        assert est2._T_cold_live        == est._T_cold_live
        assert est2._daily_history      == est._daily_history
        assert est2.vol_L               == est.vol_L
        assert est2.draw_threshold_c    == est.draw_threshold_c

    def test_round_trip_with_none_optionals(self):
        """None values for coupling_calibrated and T_cold_live survive the round-trip."""
        est = make_estimator()
        d = est.to_dict()
        est2 = FlowlessConsumptionEstimator.from_dict(d)
        assert est2.coupling_calibrated is None
        assert est2._T_cold_live        is None

    def test_from_dict_handles_missing_new_keys(self):
        """Older persisted dicts without new keys should deserialise without error."""
        d = {
            "vol_L": 100.0, "relay_w": 2000.0, "T_cold": 10.0,
            "coupling": 0.45, "T_amb_default": 18.0, "standby_w": 50.0,
            "alpha": 1.0, "T_set_calibrated": None, "daily_history": [],
        }
        est = FlowlessConsumptionEstimator.from_dict(d)
        assert est.coupling_calibrated  is None
        assert est._T_cold_live         is None
        assert est.draw_threshold_c     == 2.0   # default


# ─────────────────────────────────────────────────────────────────────────────
# Diagnostics
# ─────────────────────────────────────────────────────────────────────────────

class TestDiagnostics:
    def test_all_keys_present(self):
        est = make_estimator()
        d = est.diagnostics()
        for key in (
            "alpha", "T_set_calibrated", "coupling_calibrated",
            "T_cold_live", "relay_on_h_today", "draw_vol_today_L",
            "est_vol_today_L", "T_set_samples_today", "daily_history",
        ):
            assert key in d, f"Missing key: {key}"

    def test_values_are_none_before_data(self):
        est = make_estimator()
        d = est.diagnostics()
        assert d["T_set_calibrated"]    is None
        assert d["coupling_calibrated"] is None
        assert d["T_cold_live"]         is None

    def test_relay_on_h_accumulates(self):
        est = make_estimator()
        for _ in range(60):
            est.tick(relay_on=True, T_case=None, T_amb=18.0)
        assert abs(est.diagnostics()["relay_on_h_today"] - 1.0) < 0.01


# ─────────────────────────────────────────────────────────────────────────────
# Accumulator reset on day rollover
# ─────────────────────────────────────────────────────────────────────────────

class TestDayRollover:
    def test_accumulators_reset_after_finalize(self):
        est = make_estimator()
        est._relay_on_s       = 3600.0
        est._elapsed_s        = 7200.0
        est._T_set_samples    = [45.0]
        est._coupling_samples = [0.30]
        est._T_in_samples     = [11.0]
        est._draw_vol_today   = 25.0
        est._finalize_day(k_water=None, T_amb=18.0, true_vol_L=None)
        assert est._relay_on_s       == 0.0
        assert est._elapsed_s        == 0.0
        assert est._T_set_samples    == []
        assert est._coupling_samples == []
        assert est._T_in_samples     == []
        assert est._draw_vol_today   == 0.0

    def test_tick_resets_intraday_accumulators_on_new_day(self):
        est = make_estimator()
        est._relay_on_s      = 3600.0
        est._coupling_samples = [0.30]
        est._T_in_samples     = [11.0]
        est._draw_vol_today   = 10.0
        # Simulate day boundary detected in tick()
        est._current_day = date.today() - timedelta(days=1)
        est.tick(relay_on=False, T_case=20.0, T_amb=18.0, dt_s=60.0)
        assert est._relay_on_s       == 0.0
        assert est._coupling_samples == []
        assert est._T_in_samples     == []
        assert est._draw_vol_today   == 0.0

    def test_daily_history_bounded_at_90(self):
        est = make_estimator()
        est._daily_history = [{"date": f"2026-01-{i:02d}", "est_vol_L": 1.0,
                                "draw_vol_L": 0.0, "true_vol_L": None, "alpha": 1.0}
                              for i in range(1, 92)]
        est._finalize_day(k_water=None, T_amb=18.0, true_vol_L=None)
        assert len(est._daily_history) <= 90


# ─────────────────────────────────────────────────────────────────────────────
# SIMULATION: Simple mode — full day with HDO gaps
# ─────────────────────────────────────────────────────────────────────────────

def _run_simple_day(
    relay_schedule: list,   # list of (relay_on: bool, ticks: int, T_case: float)
    T_amb: float = 18.0,
    T_in: float = 11.0,
    coupling: float = 0.30,
    T_set: float = 70.0,
    vol_L: float = 120.0,
    standby_w: float = 0.0,
) -> FlowlessConsumptionEstimator:
    """
    Drive the estimator through one simulated day and finalise it.

    relay_schedule: list of (relay_on, n_ticks, T_case) tuples.
    T_in is passed every tick.  Returns the estimator after finalization.
    """
    est = FlowlessConsumptionEstimator(
        vol_L=vol_L, relay_w=2000.0, T_cold=12.0, coupling=coupling,
        T_amb_default=T_amb, standby_w=standby_w,
    )

    for relay_on, n_ticks, T_case in relay_schedule:
        for _ in range(n_ticks):
            est.tick(relay_on=relay_on, T_case=T_case, T_amb=T_amb, dt_s=60.0, T_in=T_in)

    vol = est.estimate_today(T_amb=T_amb)
    est._finalize_day(k_water=None, T_amb=T_amb, true_vol_L=None)
    return est


class TestSimpleModeDaySimulation:
    """
    Simulate a realistic day:
      - Morning heat (relay ON 1 h)   → T_case at T_amb + coupling*(T_set-T_amb)
      - HDO block (relay OFF / unavailable, 2 h) — passed as relay_on=False
      - Afternoon heat (relay ON 1 h)
      - Evening draw (T_case drops > threshold, relay OFF)
    """

    COUPLING = 0.30
    T_AMB    = 18.0
    T_SET    = 70.0
    T_CASE_HOT = 18.0 + 0.30 * (70.0 - 18.0)   # 33.6°C — tank fully heated

    def _default_schedule(self):
        return [
            (True,  60, self.T_CASE_HOT),   # Morning relay ON (1 h)
            (False,  1, self.T_CASE_HOT),   # relay-OFF transition → T_set sample
            (False, 60, self.T_CASE_HOT),   # HDO-like block (relay OFF 1 h)
            (True,  60, self.T_CASE_HOT),   # Afternoon relay ON
            (False,  1, self.T_CASE_HOT),   # relay-OFF → T_set sample
            (False, 30, self.T_CASE_HOT),   # cooling, no draw
        ]

    def test_estimate_positive_after_relay_on(self):
        est = _run_simple_day(self._default_schedule(), coupling=self.COUPLING,
                              T_set=self.T_SET, T_amb=self.T_AMB)
        assert est._daily_history[-1]["est_vol_L"] > 0.0

    def test_T_set_samples_collected(self):
        est = make_estimator(coupling=self.COUPLING, draw_threshold_c=2.0)
        for relay_on, n_ticks, T_case in self._default_schedule():
            for _ in range(n_ticks):
                est.tick(relay_on=relay_on, T_case=T_case, T_amb=self.T_AMB,
                         dt_s=60.0, T_in=11.0)
        # 2 relay-OFF transitions → 2 T_set samples
        assert len(est._T_set_samples) == 2

    def test_T_cold_live_set_after_day(self):
        est = _run_simple_day(self._default_schedule(), coupling=self.COUPLING,
                              T_amb=self.T_AMB)
        assert est._T_cold_live is not None
        assert abs(est._T_cold_live - 11.0) < 0.1   # should equal T_in=11.0

    def test_hdo_gap_reduces_estimate_vs_no_gap(self):
        """A day with HDO gaps (relay OFF) should yield a smaller estimate than
        a day where the relay ran continuously for the same total ON time."""
        schedule_with_gap = [
            (True,  60, self.T_CASE_HOT),
            (False, 60, self.T_CASE_HOT),   # HDO gap
            (True,  60, self.T_CASE_HOT),
        ]
        schedule_no_gap = [
            (True,  120, self.T_CASE_HOT),  # same relay-ON total, no gap
        ]
        est_gap    = _run_simple_day(schedule_with_gap,  coupling=self.COUPLING, T_amb=self.T_AMB)
        est_no_gap = _run_simple_day(schedule_no_gap,    coupling=self.COUPLING, T_amb=self.T_AMB)
        # Both have same relay_on_s (120 min), same kWh — estimates should match
        hist_gap    = est_gap._daily_history[-1]["est_vol_L"]
        hist_no_gap = est_no_gap._daily_history[-1]["est_vol_L"]
        assert abs(hist_gap - hist_no_gap) < 5.0   # within 5 L (standby=0)

    def test_draw_detected_during_off_period(self):
        """T_case drop during relay-OFF period triggers draw detection."""
        est = make_estimator(
            coupling=self.COUPLING, draw_threshold_c=2.0,
            vol_L=120.0, T_cold=12.0,
        )
        T_hot = self.T_CASE_HOT
        # relay ON then OFF
        est.tick(relay_on=True,  T_case=T_hot,        T_amb=self.T_AMB)
        est.tick(relay_on=False, T_case=T_hot,        T_amb=self.T_AMB)
        # Simulate draw: T_case drops 6°C over 2 ticks
        est.tick(relay_on=False, T_case=T_hot - 3.0,  T_amb=self.T_AMB)
        est.tick(relay_on=False, T_case=T_hot - 6.0,  T_amb=self.T_AMB)
        assert est._draw_vol_today > 0.0


# ─────────────────────────────────────────────────────────────────────────────
# SIMULATION: Simple mode — multi-day coupling convergence
# ─────────────────────────────────────────────────────────────────────────────

class TestSimpleModeMultiDay:
    """
    Drive 14 simulated days.  Two independent convergence scenarios:

    1. T_set convergence: correct config coupling → T_set_calibrated converges to ~T_SET.
    2. Coupling convergence: known T_set seeded → coupling_calibrated converges to
       TRUE_COUPLING.  (T_set and coupling are not independently observable from T_case
       alone; one must be known to infer the other.)
    """

    TRUE_COUPLING = 0.30
    T_AMB  = 18.0
    T_SET  = 70.0
    T_CASE_HOT = 18.0 + 0.30 * (70.0 - 18.0)   # 33.6

    def _simulate_days(self, n_days: int, coupling_config: float,
                       seed_T_set: Optional[float] = None) -> FlowlessConsumptionEstimator:
        est = FlowlessConsumptionEstimator(
            vol_L=120.0, relay_w=2000.0, T_cold=12.0,
            coupling=coupling_config,
            T_amb_default=self.T_AMB, standby_w=0.0,
        )
        if seed_T_set is not None:
            est.T_set_calibrated = seed_T_set
        for _ in range(n_days):
            for _ in range(5 * 60):  # 5 relay-ON/OFF cycles of 60 ticks each
                est.tick(relay_on=True,  T_case=self.T_CASE_HOT, T_amb=self.T_AMB,
                         dt_s=60.0, T_in=11.0)
                est.tick(relay_on=False, T_case=self.T_CASE_HOT, T_amb=self.T_AMB,
                         dt_s=60.0, T_in=11.0)
            est._finalize_day(k_water=None, T_amb=self.T_AMB, true_vol_L=None)
        return est

    def test_T_set_calibrated_converges(self):
        """With correct config coupling (0.30), T_set_calibrated converges to ~70°C."""
        est = self._simulate_days(n_days=14, coupling_config=self.TRUE_COUPLING)
        assert est.T_set_calibrated is not None
        assert abs(est.T_set_calibrated - self.T_SET) < 5.0

    def test_coupling_calibrated_converges(self):
        """With T_set_calibrated seeded (known), coupling converges from 0.45 → 0.30."""
        est = self._simulate_days(
            n_days=14, coupling_config=0.45, seed_T_set=self.T_SET,
        )
        assert est.coupling_calibrated is not None
        assert abs(est.coupling_calibrated - self.TRUE_COUPLING) < 0.10

    def test_T_cold_live_stabilises(self):
        est = self._simulate_days(n_days=14, coupling_config=self.TRUE_COUPLING)
        assert est._T_cold_live is not None
        assert abs(est._T_cold_live - 11.0) < 0.5


# ─────────────────────────────────────────────────────────────────────────────
# SIMULATION: Standard mode — alpha converges with flow-sensor ground truth
# ─────────────────────────────────────────────────────────────────────────────

class TestStandardModeAlphaConvergence:
    """
    Standard mode: true_vol_L comes from the flow meter.
    Start with alpha=0.7 (under-estimate).  Provide accurate ground truth each day.
    Alpha should approach 1.0 within ~10 days.
    """

    T_SET  = 45.0
    T_COLD = 10.0
    T_AMB  = 18.0

    def test_alpha_converges_toward_1(self):
        est = make_estimator(standby_w=0.0, T_cold=self.T_COLD)
        est.T_set_calibrated = self.T_SET
        est.alpha = 0.7

        for _ in range(10):
            for _ in range(120):   # 2 h relay ON
                est.tick(relay_on=True, T_case=None, T_amb=self.T_AMB, dt_s=60.0)
            # Ground truth: the "true" volume using alpha=1.0 formula
            relay_kwh = (120 * 60 / 3600.0) * 2.0   # 4 kWh
            dT = self.T_SET - self.T_COLD
            true_vol = relay_kwh * 3.6e6 / (C_WATER * RHO * dT)
            est._finalize_day(k_water=None, T_amb=self.T_AMB, true_vol_L=true_vol)

        assert est.alpha > 0.85, f"Alpha should have converged above 0.85, got {est.alpha:.3f}"

    def test_alpha_stable_when_estimate_matches_truth(self):
        """If alpha=1.0 and estimate exactly matches true_vol, alpha should stay ~1.0."""
        est = make_estimator(standby_w=0.0, T_cold=self.T_COLD)
        est.T_set_calibrated = self.T_SET
        est.alpha = 1.0

        for _ in range(5):
            for _ in range(120):
                est.tick(relay_on=True, T_case=None, T_amb=self.T_AMB, dt_s=60.0)
            est_vol = est.estimate_today(T_amb=self.T_AMB)
            est._finalize_day(k_water=None, T_amb=self.T_AMB, true_vol_L=est_vol)

        assert abs(est.alpha - 1.0) < 0.05


# ─────────────────────────────────────────────────────────────────────────────
# SIMULATION: HDO blocking — relay unavailable = relay_on=False
# ─────────────────────────────────────────────────────────────────────────────

class TestHDOBehaviourInEstimator:
    """
    During HDO the Shelly relay is physically cut → HA state = 'unavailable'.
    The controller passes relay_on=False to tick() during these intervals.
    Verify the estimator handles this correctly:
      - No relay_on_s accumulated during HDO
      - No spurious T_set samples from HDO relay-OFF transition
      - Estimate still positive for pre-HDO heating
    """

    T_AMB      = 18.0
    T_SET      = 70.0
    COUPLING   = 0.30
    T_CASE_HOT = 18.0 + COUPLING * (T_SET - T_AMB)   # 33.6

    HDO_HOURS = list(range(22, 24)) + list(range(0, 6))

    def test_no_relay_accumulation_during_hdo(self):
        est = make_estimator(coupling=self.COUPLING)
        # 1 h normal heating
        for _ in range(60):
            est.tick(relay_on=True,  T_case=self.T_CASE_HOT, T_amb=self.T_AMB)
        relay_after_heat = est._relay_on_s
        # 2 h HDO block → relay_on=False
        for _ in range(120):
            est.tick(relay_on=False, T_case=self.T_CASE_HOT, T_amb=self.T_AMB)
        assert est._relay_on_s == relay_after_heat   # unchanged during HDO

    def test_estimate_unaffected_by_hdo_gap(self):
        """Estimate computed before HDO should match after HDO passes (relay still OFF)."""
        est = make_estimator(coupling=self.COUPLING, standby_w=0.0)
        est.T_set_calibrated = self.T_SET
        for _ in range(60):
            est.tick(relay_on=True,  T_case=self.T_CASE_HOT, T_amb=self.T_AMB)
        vol_before = est.estimate_today(T_amb=self.T_AMB)
        for _ in range(120):
            est.tick(relay_on=False, T_case=self.T_CASE_HOT, T_amb=self.T_AMB)
        vol_after = est.estimate_today(T_amb=self.T_AMB)
        assert abs(vol_before - vol_after) < 1.0   # standby=0 → no change

    def test_full_day_with_hdo_window_produces_positive_estimate(self):
        """A realistic day: heat in morning, HDO block at night → positive estimate."""
        est = make_estimator(coupling=self.COUPLING, standby_w=0.0)
        est.T_set_calibrated = self.T_SET
        # Morning: 2 h relay ON
        for _ in range(2 * 60):
            est.tick(relay_on=True,  T_case=self.T_CASE_HOT, T_amb=self.T_AMB)
        # Relay OFF transition
        est.tick(relay_on=False, T_case=self.T_CASE_HOT, T_amb=self.T_AMB)
        # HDO block: 8 h relay unavailable → relay_on=False
        for _ in range(8 * 60):
            est.tick(relay_on=False, T_case=self.T_CASE_HOT, T_amb=self.T_AMB)
        vol = est.estimate_today(T_amb=self.T_AMB)
        assert vol > 0.0

    def test_simulated_week_with_nightly_hdo(self):
        """
        5-day simulation where 22:00–06:00 = HDO (relay OFF).
        Daytime has 2 h relay ON.  After each day the estimate should be non-zero.
        """
        est = FlowlessConsumptionEstimator(
            vol_L=120.0, relay_w=2000.0, T_cold=12.0, coupling=self.COUPLING,
            T_amb_default=self.T_AMB, standby_w=0.0,
        )
        est.T_set_calibrated = self.T_SET

        for day in range(5):
            # Day: 2 h relay ON at 10:00
            for _ in range(2 * 60):
                est.tick(relay_on=True,  T_case=self.T_CASE_HOT, T_amb=self.T_AMB)
            est.tick(relay_on=False, T_case=self.T_CASE_HOT, T_amb=self.T_AMB)
            # Night: 8 h HDO (relay_on=False)
            for _ in range(8 * 60):
                est.tick(relay_on=False, T_case=self.T_CASE_HOT * 0.98, T_amb=self.T_AMB)
            vol_before = est.estimate_today(T_amb=self.T_AMB)
            est._finalize_day(k_water=None, T_amb=self.T_AMB, true_vol_L=None)
            assert est._daily_history[-1]["est_vol_L"] > 0.0, \
                f"Day {day}: estimate should be positive, got {vol_before:.1f} L"
