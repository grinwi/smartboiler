# Tests for FlowlessConsumptionEstimator
# Run with:  python -m pytest smartboiler/src/smartboiler/tests/test_consumption_estimator.py -v

import sys
import os
from datetime import date, datetime, timedelta
from unittest.mock import patch

# Allow running from repo root without installing the package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from consumption_estimator import FlowlessConsumptionEstimator, C_WATER, RHO

# ── Fixtures ──────────────────────────────────────────────────────────────────

PARAMS = dict(
    vol_L        = 100.0,
    relay_w      = 2000.0,
    T_cold       = 10.0,
    coupling     = 0.45,
    T_amb_default = 18.0,
    standby_w    = 50.0,
)


def make_estimator(**overrides) -> FlowlessConsumptionEstimator:
    p = {**PARAMS, **overrides}
    return FlowlessConsumptionEstimator(**p)


# ── Energy balance math ───────────────────────────────────────────────────────

def test_energy_balance_zero_relay():
    """With no relay time, estimate should be 0."""
    est = make_estimator()
    assert est.estimate_today() == 0.0


def test_energy_balance_known_scenario():
    """
    2 hours of relay-on at 2000 W, T_set=45, T_cold=10, standby_w=0 (to isolate).
    Expected: relay_kwh = 4.0
    vol = 4.0 * 3.6e6 / (4186 * 35) = 98.3 L
    """
    est = make_estimator(standby_w=0.0)
    est.T_set_calibrated = 45.0

    # Simulate 2 hours at 60 s ticks
    ticks = int(2 * 3600 / 60)
    for _ in range(ticks):
        est.tick(relay_on=True, T_case=None, T_amb=18.0, dt_s=60.0)

    vol = est.estimate_today(T_amb=18.0)
    expected = 4.0 * 3.6e6 / (C_WATER * RHO * (45.0 - 10.0))
    assert abs(vol - expected) < 1.0, f"Expected ~{expected:.1f} L, got {vol:.1f} L"


def test_standby_reduces_estimate():
    """Standby heat loss reduces net energy and hence volume estimate."""
    est_no_standby = make_estimator(standby_w=0.0)
    est_standby    = make_estimator(standby_w=50.0)
    for est in (est_no_standby, est_standby):
        est.T_set_calibrated = 45.0
        for _ in range(120):       # 2 hours
            est.tick(relay_on=True, T_case=None, T_amb=18.0, dt_s=60.0)

    v0 = est_no_standby.estimate_today(T_amb=18.0)
    vs = est_standby.estimate_today(T_amb=18.0)
    assert vs < v0, "Standby losses should reduce volume estimate"


def test_k_water_standby_vs_standby_w():
    """estimate_today with k_water should give similar result to standby_w for typical params."""
    # For 120 L, T_set=45, T_amb=18, standby_w=60:
    #   k_water = 60 * 3600 / (120 * 4186 * 27) = 0.01592 /h
    #   They should be identical by construction (see analysis notebook)
    T_set = 45.0
    T_amb = 18.0
    vol_L = 120.0
    standby_w = 60.0
    k_water = standby_w * 3600.0 / (vol_L * RHO * C_WATER * (T_set - T_amb))

    est = make_estimator(vol_L=vol_L, standby_w=standby_w)
    est.T_set_calibrated = T_set
    # Run for exactly 24 hours
    for _ in range(24 * 60):
        est.tick(relay_on=False, T_case=None, T_amb=T_amb, dt_s=60.0)

    v_kw = est.estimate_today(k_water=k_water, T_amb=T_amb)
    v_sw = est.estimate_today(k_water=None,    T_amb=T_amb)
    # Both approaches should agree within 0.1 L (purely standby, no relay)
    assert abs(v_kw - v_sw) < 0.1, f"k_water={v_kw:.3f}  standby_w={v_sw:.3f}"


def test_alpha_scales_volume():
    """Alpha directly scales the volume estimate."""
    est = make_estimator(standby_w=0.0)
    est.T_set_calibrated = 45.0
    for _ in range(60):
        est.tick(relay_on=True, T_case=None, T_amb=18.0, dt_s=60.0)

    v1 = est.estimate_today()
    est.alpha = 2.0
    v2 = est.estimate_today()
    assert abs(v2 - 2 * v1) < 1e-6


# ── T_set calibration ─────────────────────────────────────────────────────────

def test_T_set_inferred_at_relay_off():
    """
    At a relay-OFF transition, T_water should be inferred from T_case.
    With coupling=0.45, T_amb=18, T_case=30:
    T_water = 18 + (30 - 18) / 0.45 = 44.67°C
    """
    est = make_estimator(coupling=0.45)
    T_amb  = 18.0
    T_case = 30.0  # corresponds to T_water ≈ 44.67°C

    # First tick: relay ON → sets _was_relay_on = True
    est.tick(relay_on=True, T_case=T_case, T_amb=T_amb)
    # Second tick: relay OFF → transition triggers inference
    est.tick(relay_on=False, T_case=T_case, T_amb=T_amb)

    assert len(est._T_set_samples) == 1
    expected = T_amb + (T_case - T_amb) / 0.45
    assert abs(est._T_set_samples[0] - expected) < 1e-9


def test_T_set_not_inferred_without_transition():
    """No inference when relay stays OFF (no transition)."""
    est = make_estimator()
    for _ in range(5):
        est.tick(relay_on=False, T_case=30.0, T_amb=18.0)
    assert len(est._T_set_samples) == 0


def test_T_set_calibrated_after_finalize():
    """
    After finalize_day, T_set_calibrated should be set from the 90th percentile
    of samples collected today.
    """
    est = make_estimator(coupling=0.45)
    # Simulate a few relay-off transitions at different case temps
    # coupling=0.45, T_amb=18, T_case values → T_water values
    transitions = [30.0, 28.0, 31.0, 29.5]   # T_case at relay-off
    for T_c in transitions:
        est.tick(relay_on=True,  T_case=T_c, T_amb=18.0)
        est.tick(relay_on=False, T_case=T_c, T_amb=18.0)

    # Advance to next day for finalize to trigger
    with patch("consumption_estimator.datetime") as mock_dt:
        tomorrow = date.today() + timedelta(days=1)
        mock_dt.now.return_value = datetime.combine(tomorrow, datetime.min.time())
        # Manually call _finalize_day directly
        est._current_day = date.today()   # reset so finalize detects rollover
        est._finalize_day(k_water=None, T_amb=18.0, true_vol_L=None)

    assert est.T_set_calibrated is not None
    assert 30.0 < est.T_set_calibrated < 50.0  # plausible range


# ── Alpha learning ────────────────────────────────────────────────────────────

def test_alpha_update_with_true_vol():
    """Alpha should move toward true_vol / est_vol when ground truth is provided."""
    est = make_estimator(standby_w=0.0)
    est.T_set_calibrated = 45.0
    est.alpha = 1.0

    # Simulate 2 hours relay-on
    for _ in range(120):
        est.tick(relay_on=True, T_case=None, T_amb=18.0, dt_s=60.0)

    est_vol = est.estimate_today()
    true_vol = est_vol * 0.7   # system over-estimated by 30%

    # Mock tomorrow so _finalize_day runs
    est._current_day = date.today()
    est._finalize_day(k_water=None, T_amb=18.0, true_vol_L=true_vol)

    # Alpha should have moved down from 1.0
    assert est.alpha < 1.0


def test_alpha_clamp():
    """Alpha must stay within [0.5, 2.0] even with extreme true_vol."""
    est = make_estimator(standby_w=0.0)
    est.T_set_calibrated = 45.0
    est.alpha = 1.0
    for _ in range(60):
        est.tick(relay_on=True, T_case=None, T_amb=18.0, dt_s=60.0)

    # 10× overshoot should clamp at 2.0
    est_vol = est.estimate_today()
    est._current_day = date.today()
    est._finalize_day(k_water=None, T_amb=18.0, true_vol_L=est_vol * 10)
    assert est.alpha <= 2.0

    # Reset and check lower clamp
    est2 = make_estimator(standby_w=0.0)
    est2.T_set_calibrated = 45.0
    est2.alpha = 1.0
    for _ in range(60):
        est2.tick(relay_on=True, T_case=None, T_amb=18.0, dt_s=60.0)
    est_vol2 = est2.estimate_today()
    est2._current_day = date.today()
    est2._finalize_day(k_water=None, T_amb=18.0, true_vol_L=est_vol2 * 0.01)
    assert est2.alpha >= 0.5


# ── Persistence ───────────────────────────────────────────────────────────────

def test_round_trip_serialisation():
    """to_dict / from_dict round-trip preserves all learned state."""
    est = make_estimator()
    est.alpha = 1.23
    est.T_set_calibrated = 44.5
    est._daily_history = [{"date": "2026-01-01", "est_vol_L": 35.0, "true_vol_L": None, "alpha": 1.23}]

    d = est.to_dict()
    est2 = FlowlessConsumptionEstimator.from_dict(d)

    assert est2.alpha == est.alpha
    assert est2.T_set_calibrated == est.T_set_calibrated
    assert est2._daily_history == est._daily_history
    assert est2.vol_L == est.vol_L
    assert est2.relay_w == est.relay_w


# ── Diagnostics ───────────────────────────────────────────────────────────────

def test_diagnostics_keys():
    """diagnostics() must return expected keys."""
    est = make_estimator()
    d = est.diagnostics()
    for key in ("alpha", "T_set_calibrated", "relay_on_h_today", "est_vol_today_L"):
        assert key in d, f"Missing key: {key}"


# ── ThermalModel.infer_T_set_from_case ────────────────────────────────────────

def test_thermal_model_infer_T_set():
    """ThermalModel.infer_T_set_from_case should match manual formula."""
    from thermal_model import ThermalModel

    T_case  = 30.0
    T_amb   = 18.0
    coupling = 0.45
    expected = T_amb + (T_case - T_amb) / coupling

    result = ThermalModel.infer_T_set_from_case(T_case, T_amb, coupling)
    assert result is not None
    assert abs(result - expected) < 1e-9


def test_thermal_model_infer_T_set_returns_none_below_ambient():
    """Returns None when inferred T_water is < T_amb + 1°C (too close to ambient)."""
    from thermal_model import ThermalModel

    # T_case = T_amb + coupling * (T_water - T_amb)
    # For T_water = T_amb + 0.5 → T_case = 18 + 0.45*0.5 = 18.225
    # 18.225 should infer T_water = 18.5 < T_amb + 1.0 = 19.0 → None
    result = ThermalModel.infer_T_set_from_case(18.2, 18.0, coupling=0.45)
    assert result is None


def test_thermal_model_infer_T_set_bad_inputs():
    """Returns None for non-numeric inputs."""
    from thermal_model import ThermalModel
    assert ThermalModel.infer_T_set_from_case(float("nan"), 18.0) is None
    assert ThermalModel.infer_T_set_from_case(30.0, float("inf")) is None


# ── Integration: full day simulation ─────────────────────────────────────────

def test_full_day_simulation():
    """
    Simulate a day with:
      - 3 × 1-hour relay-on blocks
      - relay-off transitions providing T_set samples
      - finalize_day called with no ground truth
    Checks that estimate is positive and T_set is updated.
    """
    est = make_estimator(standby_w=0.0, coupling=0.45)

    T_set_true = 45.0
    T_amb = 18.0
    # T_case when tank is full (coupling=0.45):
    T_case_full = T_amb + 0.45 * (T_set_true - T_amb)  # = 30.15°C

    blocks = [
        ("on", 3600),
        ("off_transition", 60),  # relay-off with T_case reading
        ("off", 3600),
        ("on", 3600),
        ("off_transition", 60),
        ("off", 3600),
        ("on", 3600),
        ("off_transition", 60),
    ]

    for block_type, duration_s in blocks:
        ticks = int(duration_s / 60)
        if block_type == "on":
            for _ in range(ticks):
                est.tick(relay_on=True, T_case=T_case_full, T_amb=T_amb, dt_s=60.0)
        elif block_type == "off_transition":
            # Previous tick was relay ON; this tick is relay OFF → triggers infer
            est.tick(relay_on=False, T_case=T_case_full, T_amb=T_amb, dt_s=60.0)
        else:
            for _ in range(ticks):
                est.tick(relay_on=False, T_case=T_case_full * 0.95, T_amb=T_amb, dt_s=60.0)

    vol = est.estimate_today(T_amb=T_amb)
    assert vol > 0.0, "Should estimate positive volume after relay was ON"
    assert len(est._T_set_samples) == 3, "Should have 3 T_set samples from relay-off transitions"

    # Run finalize
    est._current_day = date.today()
    final_vol = est._finalize_day(k_water=None, T_amb=T_amb, true_vol_L=None)
    assert final_vol > 0.0
    assert est.T_set_calibrated is not None
    assert 30.0 < est.T_set_calibrated < 85.0
    assert est._relay_on_s == 0.0, "Accumulators should reset after finalize"
