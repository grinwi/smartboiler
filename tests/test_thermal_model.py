"""
Simulation tests for ThermalModel.

All tests are self-contained: no network, no HA, no filesystem.
Synthetic ground-truth cooling data is generated from the same Newton's-law
equations the model fits, allowing exact tolerance checks.

Physics recap
─────────────
  T_water(dt) = T_amb + (T_set - T_amb) * exp(-k_w * dt_h)
  T_case(dt)  = T_amb + (C0   - T_amb) * exp(-k_c * dt_h)

  k_case  = 0.50 /h  (case shell cools with ~1.4h half-life)
  k_water = 0.15 /h  (water cools with ~4.6h half-life)
  mass_ratio = k_water / k_case = 0.30  (default)
"""

import math
import sys
import time as _time_module
from typing import List, Tuple

import pytest

# thermal_model has no heavy deps; import directly
sys.path.insert(0, "src")
from smartboiler.thermal_model import ThermalModel, SAMPLE_MIN_INTERVAL_S, _CalibEvent

# ── Ground-truth simulation parameters ───────────────────────────────────────

K_CASE_TRUE  = 0.50   # /h
K_WATER_TRUE = 0.15   # /h  →  mass_ratio = 0.30
T_SET        = 60.0
T_AMB        = 18.0
C0           = 42.0   # case temperature at the moment thermostat trips (< T_set)

SAMPLE_INTERVAL_H = 0.25   # 15 min — matches SAMPLE_MIN_INTERVAL_S
# Use current real time as base so _prune never discards "now" events.
# All timestamps in tests are T0 + positive offsets, staying inside the window.
T0 = _time_module.time()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _true_T_water(dt_h: float, T_amb: float = T_AMB) -> float:
    return T_amb + (T_SET - T_amb) * math.exp(-K_WATER_TRUE * dt_h)


def _true_T_case(dt_h: float, C0_: float = C0, T_amb: float = T_AMB) -> float:
    return T_amb + (C0_ - T_amb) * math.exp(-K_CASE_TRUE * dt_h)


def _simulate_cooling(
    t0: float,
    n_samples: int,
    T_amb: float = T_AMB,
    C0_: float = C0,
) -> List[Tuple[float, float, float]]:
    """
    Return list of (timestamp, T_case_true, T_water_true) for n_samples,
    spaced SAMPLE_INTERVAL_H apart, starting just after t0.
    """
    result = []
    for i in range(1, n_samples + 1):
        dt_h = i * SAMPLE_INTERVAL_H
        ts = t0 + dt_h * 3600.0
        T_c = _true_T_case(dt_h, C0_=C0_, T_amb=T_amb)
        T_w = _true_T_water(dt_h, T_amb=T_amb)
        result.append((ts, T_c, T_w))
    return result


def _fresh_model(
    mass_ratio: float = 0.3,
    window_days: float = 7.0,
    t_now: float = T0,
) -> ThermalModel:
    """Create a fresh model whose clock is frozen at t_now (avoids real-time drift)."""
    return ThermalModel(window_days=window_days, mass_ratio=mass_ratio,
                        clock=lambda: t_now)


def _calibrate_and_feed(
    model: ThermalModel,
    n_samples: int = 20,
    T_amb: float = T_AMB,
    C0_: float = C0,
    t0: float = T0,
    fit: bool = True,
) -> List[Tuple[float, float, float]]:
    """Calibrate model, feed n_samples, optionally trigger fit. Returns sample list."""
    model.observe_calibration(T_SET, C0_, T_amb, timestamp=t0)
    samples = _simulate_cooling(t0, n_samples, T_amb=T_amb, C0_=C0_)
    for ts, T_c, _ in samples:
        model.observe_case_tmp(T_c, T_amb, timestamp=ts)
    if fit:
        model._try_fit()
    return samples


# ── Tests: pre-calibration behaviour ─────────────────────────────────────────

class TestNoCalibraton:
    def test_returns_none_before_any_calibration(self):
        model = _fresh_model()
        assert model.estimate_water_tmp(35.0, T_AMB) is None

    def test_is_fitted_false(self):
        model = _fresh_model()
        assert not model.is_fitted

    def test_last_calibration_ts_is_none(self):
        model = _fresh_model()
        assert model.last_calibration_ts is None

    def test_diagnostics_returns_dict(self):
        model = _fresh_model()
        d = model.diagnostics()
        assert d["fitted"] is False
        assert d["n_calib_events"] == 0


# ── Tests: calibration event ──────────────────────────────────────────────────

class TestCalibrationEvent:
    def test_immediately_after_calibration_returns_T_set(self):
        """Right at t0 (dt=0), T_water must equal T_set."""
        model = _fresh_model()
        model.observe_calibration(T_SET, C0, T_AMB, timestamp=T0)
        # estimate at exactly t0
        est = model.estimate_water_tmp(C0, T_AMB, timestamp=T0)
        assert est is not None
        assert abs(est - T_SET) < 0.1

    def test_calibration_stored(self):
        model = _fresh_model()
        model.observe_calibration(T_SET, C0, T_AMB, timestamp=T0)
        assert model.last_calibration_ts == T0
        assert len(model._calib_events) == 1

    def test_second_calibration_updates_reference(self):
        model = _fresh_model()
        model.observe_calibration(T_SET, C0, T_AMB, timestamp=T0)
        t1 = T0 + 3600
        model.observe_calibration(T_SET, C0 - 2, T_AMB, timestamp=t1)
        assert model.last_calibration_ts == t1
        # Estimate at t1: should still be T_set
        est = model.estimate_water_tmp(C0 - 2, T_AMB, timestamp=t1)
        assert abs(est - T_SET) < 0.1


# ── Tests: proportional fallback (no fit yet) ─────────────────────────────────

class TestProportionalFallback:
    """Before fitting, the model uses T_water = T_amb + ratio*(T_case - T_amb)."""

    def test_fallback_is_between_T_amb_and_T_set(self):
        model = _fresh_model()
        model.observe_calibration(T_SET, C0, T_AMB, timestamp=T0)
        # 4 h into cooling — not enough samples for fit
        dt_h = 4.0
        T_c = _true_T_case(dt_h)
        est = model.estimate_water_tmp(T_c, T_AMB, timestamp=T0 + dt_h * 3600)
        assert T_AMB <= est <= T_SET

    def test_fallback_monotonically_decreasing_over_time(self):
        """As case cools, estimate should decrease (even without fit)."""
        model = _fresh_model()
        model.observe_calibration(T_SET, C0, T_AMB, timestamp=T0)
        estimates = []
        for dt_h in [0.5, 1.0, 2.0, 4.0, 8.0]:
            T_c = _true_T_case(dt_h)
            ts = T0 + dt_h * 3600
            est = model.estimate_water_tmp(T_c, T_AMB, timestamp=ts)
            estimates.append(est)
        # each estimate should be <= the previous
        for prev, curr in zip(estimates, estimates[1:]):
            assert curr <= prev + 0.1, f"estimate went up: {prev:.2f} → {curr:.2f}"

    def test_fallback_at_T_amb_returns_T_amb(self):
        """When case reaches ambient, estimate should clamp to ambient."""
        model = _fresh_model()
        model.observe_calibration(T_SET, C0, T_AMB, timestamp=T0)
        est = model.estimate_water_tmp(T_AMB, T_AMB, timestamp=T0 + 100 * 3600)
        assert abs(est - T_AMB) < 0.5


# ── Tests: fitting ────────────────────────────────────────────────────────────

class TestFitting:
    def test_fit_succeeds_with_enough_samples(self):
        model = _fresh_model()
        _calibrate_and_feed(model, n_samples=20, fit=True)
        assert model.is_fitted
        assert model.k_case is not None
        assert model.k_water is not None

    def test_fit_fails_with_too_few_samples(self):
        model = _fresh_model()
        _calibrate_and_feed(model, n_samples=4, fit=True)
        # 4 < _MIN_SAMPLES_FOR_FIT (8) → should not fit
        assert not model.is_fitted

    def test_fit_recovers_k_case_within_20_percent(self):
        """Fitted k_case should be within 20% of the true value."""
        model = _fresh_model(mass_ratio=K_WATER_TRUE / K_CASE_TRUE)
        _calibrate_and_feed(model, n_samples=24, fit=True)  # 6 h of data
        assert model.is_fitted
        rel_err = abs(model.k_case - K_CASE_TRUE) / K_CASE_TRUE
        assert rel_err < 0.20, f"k_case={model.k_case:.4f} vs truth={K_CASE_TRUE}"

    def test_fit_derives_k_water_from_mass_ratio(self):
        model = _fresh_model(mass_ratio=0.3)
        _calibrate_and_feed(model, n_samples=20, fit=True)
        assert model.is_fitted
        assert abs(model.k_water - model.k_case * 0.3) < 1e-9

    def test_fit_rejects_negative_k(self, monkeypatch):
        """Corrupted data that produces k <= 0 must be silently rejected."""
        model = _fresh_model()
        model.observe_calibration(T_SET, C0, T_AMB, timestamp=T0)
        # Feed samples where T_case is INCREASING (impossible physically → bad k)
        for i in range(1, 20):
            ts = T0 + i * SAMPLE_MIN_INTERVAL_S
            rising_T_case = C0 + i * 0.5   # heating up — nonsensical for cooling
            model.observe_case_tmp(rising_T_case, T_AMB, timestamp=ts)
        result = model._try_fit()
        assert not model.is_fitted

    def test_fit_ignores_samples_above_C0(self):
        """Samples where T_case > C0 (boiler reheated) must be skipped."""
        model = _fresh_model()
        model.observe_calibration(T_SET, C0, T_AMB, timestamp=T0)
        # Feed one real cooling sample, then some "reheated" samples
        model.observe_case_tmp(_true_T_case(0.25), T_AMB, timestamp=T0 + 0.25 * 3600)
        for i in range(2, 22):
            model.observe_case_tmp(C0 + 5.0, T_AMB, timestamp=T0 + i * SAMPLE_MIN_INTERVAL_S)
        # Those bad samples are filtered; fit should fail (< 8 valid)
        model._try_fit()
        assert not model.is_fitted


# ── Tests: estimation accuracy after fitting ──────────────────────────────────

class TestEstimationAccuracy:
    """Core accuracy test: simulate a real cooling scenario, fit, then check RMSE."""

    def _run_accuracy_scenario(
        self,
        T_amb: float = T_AMB,
        C0_: float = C0,
        tolerance_deg: float = 3.0,
    ) -> float:
        """Fit model on first 20 samples, evaluate on next 16. Returns RMSE."""
        model = ThermalModel(window_days=7.0, mass_ratio=K_WATER_TRUE / K_CASE_TRUE)
        t0 = T0
        model.observe_calibration(T_SET, C0_, T_amb, timestamp=t0)

        all_samples = _simulate_cooling(t0, n_samples=36, T_amb=T_amb, C0_=C0_)
        # Feed first 20 for training
        for ts, T_c, _ in all_samples[:20]:
            model.observe_case_tmp(T_c, T_amb, timestamp=ts)
        model._try_fit()
        assert model.is_fitted, "Model did not fit — increase n_samples"

        # Evaluate on remaining 16 (4–9 h into cooling)
        errors = []
        for ts, T_c, T_w_true in all_samples[20:]:
            est = model.estimate_water_tmp(T_c, T_amb, timestamp=ts)
            assert est is not None
            errors.append(abs(est - T_w_true))

        rmse = math.sqrt(sum(e**2 for e in errors) / len(errors))
        return rmse

    def test_rmse_below_3_degrees_standard_conditions(self):
        rmse = self._run_accuracy_scenario()
        assert rmse < 3.0, f"RMSE={rmse:.2f}°C exceeds tolerance"

    def test_rmse_below_3_degrees_cold_ambient(self):
        """Boiler in unheated garage in winter."""
        rmse = self._run_accuracy_scenario(T_amb=5.0)
        assert rmse < 3.0, f"RMSE={rmse:.2f}°C (cold ambient)"

    def test_rmse_below_3_degrees_warm_ambient(self):
        """Boiler in warm utility room in summer."""
        rmse = self._run_accuracy_scenario(T_amb=28.0)
        assert rmse < 3.0, f"RMSE={rmse:.2f}°C (warm ambient)"

    def test_estimate_bounded_by_T_amb_and_T_set(self):
        model = _fresh_model(mass_ratio=K_WATER_TRUE / K_CASE_TRUE)
        samples = _calibrate_and_feed(model, n_samples=20, fit=True)
        for ts, T_c, _ in samples:
            est = model.estimate_water_tmp(T_c, T_AMB, timestamp=ts)
            assert T_AMB <= est <= T_SET, f"Out of bounds: {est:.2f}"

    def test_estimate_is_strictly_decreasing_after_fit(self):
        """After fitting, estimates over 24 h should monotonically decrease."""
        model = _fresh_model(mass_ratio=K_WATER_TRUE / K_CASE_TRUE)
        _calibrate_and_feed(model, n_samples=20, fit=True)
        estimates = []
        for dt_h in [0.25, 0.5, 1.0, 2.0, 4.0, 6.0, 10.0, 16.0, 24.0]:
            T_c = _true_T_case(dt_h)
            ts = T0 + dt_h * 3600
            est = model.estimate_water_tmp(T_c, T_AMB, timestamp=ts)
            estimates.append((dt_h, est))
        for (h_a, e_a), (h_b, e_b) in zip(estimates, estimates[1:]):
            assert e_b <= e_a + 0.05, (
                f"Estimate rose: {e_a:.2f}°C at {h_a}h → {e_b:.2f}°C at {h_b}h"
            )


# ── Tests: diagnostic snapshot for dashboard ────────────────────────────────

class TestDiagnosticSnapshot:
    def test_snapshot_uses_proportional_fallback_before_fit(self):
        model = _fresh_model()
        model.observe_calibration(T_SET, C0, T_AMB, timestamp=T0)

        dt_h = 1.0
        T_c = _true_T_case(dt_h)
        snap = model.debug_snapshot(T_c, T_AMB, timestamp=T0 + dt_h * 3600.0)

        assert snap["available"] is True
        assert snap["mode"] == "proportional_fallback"
        assert snap["estimate"] is not None
        assert snap["intermediates"]["ratio"] > 0.0
        assert len(snap["equations"]) == 2

    def test_snapshot_uses_fitted_case_decay_after_fit(self):
        model = _fresh_model(mass_ratio=K_WATER_TRUE / K_CASE_TRUE)
        _calibrate_and_feed(model, n_samples=24, fit=True)

        dt_h = 2.0
        T_c = _true_T_case(dt_h)
        true_water = _true_T_water(dt_h)
        snap = model.debug_snapshot(T_c, T_AMB, timestamp=T0 + dt_h * 3600.0)

        assert snap["available"] is True
        assert snap["mode"] == "fitted_case_decay"
        assert snap["estimate"] == pytest.approx(true_water, abs=3.0)
        assert snap["intermediates"]["elapsed_h"] == pytest.approx(dt_h, abs=0.4)
        assert snap["calibration"]["case_tmp"] == pytest.approx(C0, abs=0.1)

    def test_snapshot_contains_current_cycle_samples(self):
        model = _fresh_model()
        _calibrate_and_feed(model, n_samples=10, fit=False)

        snap = model.debug_snapshot(_true_T_case(1.0), T_AMB, timestamp=T0 + 3600.0)

        assert len(snap["current_cycle_samples"]) > 0
        first = snap["current_cycle_samples"][0]
        assert "timestamp" in first
        assert "estimated_water_tmp" in first


# ── Tests: seasonal adaptation ────────────────────────────────────────────────

class TestSeasonalAdaptation:
    """
    Simulate summer→winter ambient shift.
    After the rolling window expires the old summer data is pruned and the
    model re-fits with winter data only.
    """

    def test_model_adapts_after_window_expires(self):
        window_days = 3.0
        T_amb_summer = 25.0
        T_amb_winter = 5.0

        # ── Phase 1: summer — clock is at t_summer ────────────────────────
        t_summer = T0
        t_winter = t_summer + window_days * 86400 + 1  # just past the window

        # Model clock starts at summer
        model = ThermalModel(window_days=window_days, mass_ratio=K_WATER_TRUE / K_CASE_TRUE,
                             clock=lambda: t_summer)
        C0_s = C0
        model.observe_calibration(T_SET, C0_s, T_amb_summer, timestamp=t_summer)
        for i in range(1, 21):
            ts = t_summer + i * SAMPLE_MIN_INTERVAL_S
            dt_h = i * SAMPLE_INTERVAL_H
            T_c = T_amb_summer + (C0_s - T_amb_summer) * math.exp(-K_CASE_TRUE * dt_h)
            model.observe_case_tmp(T_c, T_amb_summer, timestamp=ts)
        model._try_fit()
        assert model.is_fitted

        # ── Phase 2: advance clock to winter; summer data should be pruned ─
        model._clock = lambda: t_winter
        C0_w = T_amb_winter + (T_SET - T_amb_winter) * 0.6
        model.observe_calibration(T_SET, C0_w, T_amb_winter, timestamp=t_winter)

        for i in range(1, 25):
            ts = t_winter + i * SAMPLE_MIN_INTERVAL_S
            dt_h = i * SAMPLE_INTERVAL_H
            T_c = T_amb_winter + (C0_w - T_amb_winter) * math.exp(-K_CASE_TRUE * dt_h)
            model.observe_case_tmp(T_c, T_amb_winter, timestamp=ts)

        model._try_fit()
        assert model.is_fitted

        # Summer calib events must be pruned
        assert all(e.T_amb == T_amb_winter for e in model._calib_events), \
            "Summer calibration was not pruned from the window"

        # Estimates on winter data must be within 4°C
        eval_dt_h = 3.0
        T_c_eval = T_amb_winter + (C0_w - T_amb_winter) * math.exp(-K_CASE_TRUE * eval_dt_h)
        T_w_true = T_amb_winter + (T_SET - T_amb_winter) * math.exp(-K_WATER_TRUE * eval_dt_h)
        est = model.estimate_water_tmp(T_c_eval, T_amb_winter,
                                        timestamp=t_winter + eval_dt_h * 3600)
        assert abs(est - T_w_true) < 4.0, f"Winter estimate off: {est:.2f} vs {T_w_true:.2f}"

    def test_prune_removes_old_data(self):
        window_days = 3.0
        model = _fresh_model(window_days=window_days)
        # Insert an event that is older than the window
        old_ts = T0 - (window_days + 1) * 86400
        model._calib_events.append(
            _CalibEvent(ts=old_ts, T_set=T_SET, T_case=C0, T_amb=T_AMB)
        )
        assert len(model._calib_events) == 1
        # Trigger prune by adding a new calibration (clock is frozen at T0)
        model.observe_calibration(T_SET, C0, T_AMB, timestamp=T0)
        # Old event (3 days before T0 - 7day_window = well outside) must be gone
        assert all(e.ts >= T0 - window_days * 86400 for e in model._calib_events)


# ── Tests: sample rate limiting ───────────────────────────────────────────────

class TestSampleRateLimiting:
    def test_samples_not_stored_before_interval(self):
        model = _fresh_model()
        model.observe_calibration(T_SET, C0, T_AMB, timestamp=T0)
        # Feed 5 samples spaced only 60 s apart — well below the 15-min limit
        for i in range(5):
            model.observe_case_tmp(C0 - i, T_AMB, timestamp=T0 + i * 60)
        # Only the first one should be stored (second is 60s < 900s after first)
        assert len(model._samples) == 1

    def test_samples_stored_after_interval(self):
        model = _fresh_model()
        model.observe_calibration(T_SET, C0, T_AMB, timestamp=T0)
        for i in range(5):
            model.observe_case_tmp(C0 - i, T_AMB, timestamp=T0 + i * SAMPLE_MIN_INTERVAL_S)
        assert len(model._samples) == 5

    def test_no_samples_without_calibration(self):
        model = _fresh_model()
        for i in range(10):
            model.observe_case_tmp(30.0, T_AMB, timestamp=T0 + i * SAMPLE_MIN_INTERVAL_S)
        assert len(model._samples) == 0


# ── Tests: persistence round-trip ────────────────────────────────────────────

class TestPersistence:
    def test_roundtrip_preserves_fitted_params(self):
        model = _fresh_model(mass_ratio=K_WATER_TRUE / K_CASE_TRUE)
        _calibrate_and_feed(model, n_samples=20, fit=True)
        assert model.is_fitted

        d = model.to_dict()
        restored = ThermalModel.from_dict(d)

        assert restored.is_fitted
        assert abs(restored.k_case  - model.k_case)  < 1e-9
        assert abs(restored.k_water - model.k_water) < 1e-9
        assert restored.window_days == model.window_days
        assert restored.mass_ratio  == model.mass_ratio

    def test_roundtrip_preserves_calib_events(self):
        model = _fresh_model()
        model.observe_calibration(T_SET, C0, T_AMB, timestamp=T0)
        model.observe_calibration(T_SET, C0 - 1, T_AMB, timestamp=T0 + 3600)

        restored = ThermalModel.from_dict(model.to_dict())
        assert len(restored._calib_events) == 2
        assert restored.last_calibration_ts == T0 + 3600

    def test_roundtrip_preserves_samples(self):
        model = _fresh_model()
        _calibrate_and_feed(model, n_samples=10, fit=False)

        restored = ThermalModel.from_dict(model.to_dict())
        assert len(restored._samples) == 10

    def test_unfitted_roundtrip_still_estimates(self):
        """After restore without fit, proportional fallback must still work."""
        model = _fresh_model()
        model.observe_calibration(T_SET, C0, T_AMB, timestamp=T0)
        restored = ThermalModel.from_dict(model.to_dict())
        # No fit yet; proportional fallback should kick in
        dt_h = 2.0
        T_c = _true_T_case(dt_h)
        est = restored.estimate_water_tmp(T_c, T_AMB, timestamp=T0 + dt_h * 3600)
        assert est is not None
        assert T_AMB <= est <= T_SET


# ── Tests: maybe_refit scheduling ────────────────────────────────────────────

class TestMaybeRefit:
    def test_maybe_refit_does_not_refit_before_window(self):
        model = _fresh_model()
        _calibrate_and_feed(model, n_samples=20, fit=True)
        assert model.is_fitted
        old_k = model.k_case

        # last_fit_ts was just set by _try_fit to T0 (clock frozen at T0)
        # Advance clock by only 1 day (< window_days=7): should not re-fit
        model._clock = lambda: T0 + 1 * 86400
        model._samples.clear()   # clear samples so re-fit would fail if attempted
        model.maybe_refit()
        assert model.k_case == old_k

    def test_maybe_refit_triggers_after_window(self, monkeypatch):
        model = _fresh_model()
        _calibrate_and_feed(model, n_samples=20, fit=True)
        assert model.is_fitted

        # Advance clock by 8 days (> window_days=7): should trigger
        t_refitted = T0 + 8 * 86400
        model._clock = lambda: t_refitted

        # Add fresh calibration + samples at the new time
        model.observe_calibration(T_SET, C0, T_AMB, timestamp=t_refitted)
        for i in range(1, 21):
            ts = t_refitted + i * SAMPLE_MIN_INTERVAL_S
            dt_h = i * SAMPLE_INTERVAL_H
            T_c = _true_T_case(dt_h)
            model.observe_case_tmp(T_c, T_AMB, timestamp=ts)

        called = []
        original_try_fit = model._try_fit
        def _patched():
            called.append(True)
            return original_try_fit()
        monkeypatch.setattr(model, "_try_fit", _patched)

        model.maybe_refit()
        assert called, "maybe_refit did not call _try_fit after window expired"


# ── Tests: edge cases ────────────────────────────────────────────────────────

class TestEdgeCases:
    def test_case_at_exact_ambient_clamps_to_ambient(self):
        model = _fresh_model()
        model.observe_calibration(T_SET, C0, T_AMB, timestamp=T0)
        est = model.estimate_water_tmp(T_AMB, T_AMB, timestamp=T0 + 100 * 3600)
        assert abs(est - T_AMB) < 0.5

    def test_case_above_C0_uses_timestamp_fallback(self):
        """If T_case > C0 (boiler reheating), model uses time-based fallback not log."""
        model = _fresh_model(mass_ratio=K_WATER_TRUE / K_CASE_TRUE)
        _calibrate_and_feed(model, n_samples=20, fit=True)
        # T_case above C0 is physically inconsistent; should not crash
        est = model.estimate_water_tmp(C0 + 5.0, T_AMB, timestamp=T0 + 0.5 * 3600)
        assert est is not None
        assert T_AMB <= est <= T_SET

    def test_clock_jump_backwards_returns_T_set(self):
        """If timestamp is before the most-recent calibration, return T_set."""
        model = _fresh_model()
        calib_ts = T0 + 3600   # calibration 1 h into simulated time
        model.observe_calibration(T_SET, C0, T_AMB, timestamp=calib_ts)
        # Query with a timestamp *before* the calibration
        est = model.estimate_water_tmp(C0, T_AMB, timestamp=calib_ts - 100)
        assert est is not None
        assert abs(est - T_SET) < 0.1

    def test_diagnostics_after_fit(self):
        model = _fresh_model()
        _calibrate_and_feed(model, n_samples=20, fit=True)
        d = model.diagnostics()
        assert d["fitted"] is True
        assert d["k_case"] > 0
        assert d["k_water"] > 0
        assert d["n_calib_events"] >= 1
        assert d["n_fit_samples"] >= 8

    def test_very_low_C0_minus_T_amb(self):
        """When C0 ≈ T_amb, fallback must not divide by zero."""
        model = _fresh_model()
        T_amb_close = T_AMB
        C0_close = T_amb_close + 0.3   # nearly at ambient
        model.observe_calibration(T_SET, C0_close, T_amb_close, timestamp=T0)
        # Should not raise; may return None or a reasonable value
        est = model.estimate_water_tmp(T_amb_close + 0.1, T_amb_close, timestamp=T0 + 3600)
        # If returned, must be within bounds
        if est is not None:
            assert T_AMB - 1 <= est <= T_SET + 1


# ── Full end-to-end simulation ────────────────────────────────────────────────

class TestFullCycleSimulation:
    """
    Simulate 3 complete heat/cool cycles (as a boiler would experience in
    2–3 days of normal operation) and verify that estimation quality improves
    after each cycle as more calibration data accumulates.
    """

    def _make_model(self, t_start: float) -> ThermalModel:
        """Model whose clock advances based on the current cycle offset."""
        # Use a mutable container so the lambda can reference the latest ts
        state = [t_start]
        m = ThermalModel(window_days=7.0, mass_ratio=K_WATER_TRUE / K_CASE_TRUE,
                         clock=lambda: state[0])
        m._state_ref = state   # keep alive
        return m

    def test_three_cycle_simulation(self):
        cycle_duration_h = 8.0
        errors_per_cycle = []
        model = self._make_model(T0)

        for cycle in range(3):
            t_calib = T0 + cycle * cycle_duration_h * 3600
            # Advance model clock to the latest timestamp used this cycle
            model._state_ref[0] = t_calib + 32 * SAMPLE_MIN_INTERVAL_S

            model.observe_calibration(T_SET, C0, T_AMB, timestamp=t_calib)
            samples = _simulate_cooling(t_calib, n_samples=32)
            for ts, T_c, _ in samples:
                model.observe_case_tmp(T_c, T_AMB, timestamp=ts)

            model._try_fit()

            if model.is_fitted:
                errs = []
                for ts, T_c, T_w_true in samples[-12:]:
                    est = model.estimate_water_tmp(T_c, T_AMB, timestamp=ts)
                    if est is not None:
                        errs.append(abs(est - T_w_true))
                if errs:
                    errors_per_cycle.append(sum(errs) / len(errs))

        assert len(errors_per_cycle) >= 1, "Model never fitted during 3 cycles"
        final_mae = errors_per_cycle[-1]
        assert final_mae < 3.0, (
            f"Final MAE={final_mae:.2f}°C after 3 cycles (errors={errors_per_cycle})"
        )

    def test_estimation_improves_or_stays_stable_across_cycles(self):
        """MAE in cycle 3 should not be significantly worse than cycle 1."""
        model = self._make_model(T0)
        maes = []

        for cycle in range(3):
            t_calib = T0 + cycle * 8 * 3600
            model._state_ref[0] = t_calib + 32 * SAMPLE_MIN_INTERVAL_S

            model.observe_calibration(T_SET, C0, T_AMB, timestamp=t_calib)
            samples = _simulate_cooling(t_calib, n_samples=32)
            for ts, T_c, _ in samples:
                model.observe_case_tmp(T_c, T_AMB, timestamp=ts)
            model._try_fit()

            if model.is_fitted:
                errs = [
                    abs(model.estimate_water_tmp(T_c, T_AMB, timestamp=ts) - T_w)
                    for ts, T_c, T_w in samples[-8:]
                    if model.estimate_water_tmp(T_c, T_AMB, timestamp=ts) is not None
                ]
                maes.append(sum(errs) / len(errs) if errs else float("inf"))

        if len(maes) >= 2:
            assert maes[-1] <= maes[0] + 1.0, (
                f"MAE degraded: cycle1={maes[0]:.2f} cycle3={maes[-1]:.2f}"
            )
