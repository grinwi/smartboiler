"""
Tests for TemperatureEstimator — 4-level water temperature fallback chain.

Covers every branch in get_boiler_tmp():
  L1  Direct NTC probe
  L2  Power feedback:
        relay ON + power < 50 W  → T_set   (thermostat tripped)
        relay ON + power > 50 W  → last_known  (actively heating; L3 must be skipped)
        relay ON + power is None → fall through to L3
        relay OFF                → fall through to L3
  L3  Thermal model (cooling equation; only valid when relay is OFF)
  L4  Last known value

Also covers:
  - get_ambient_tmp() with and without entity
  - no power_entity_id configured (L2 skipped entirely)
  - no case_tmp_entity_id configured (L3 skipped entirely)
  - combinations of missing entities
"""

from unittest.mock import MagicMock

import pytest

from smartboiler.temperature_estimator import TemperatureEstimator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SET_TMP = 60.0
AREA_TMP_DEFAULT = 20.0


def _make_ha(
    *,
    direct_tmp=None,
    power=None,
    relay_on=False,
    case_tmp=None,
    area_tmp=None,
):
    """Return a mock HAClient configured for the given sensor readings."""
    ha = MagicMock()
    ha.get_state_value.side_effect = lambda eid: {
        "sensor.direct": direct_tmp,
        "sensor.power": power,
        "sensor.case": case_tmp,
        "sensor.area": area_tmp,
    }.get(eid)
    ha.is_entity_on.return_value = relay_on
    return ha


def _make_thermal_model(estimate=None):
    m = MagicMock()
    m.estimate_water_tmp.return_value = estimate
    m.debug_snapshot.return_value = {
        "available": estimate is not None,
        "estimate": estimate,
        "mode": "fitted_case_decay" if estimate is not None else "unavailable",
        "mode_label": "Thermal model preview",
        "reason": "preview",
        "equations": [],
        "inputs": {"case_tmp": None, "ambient_tmp": None},
        "intermediates": {},
        "calibration": None,
        "model": {},
        "recent_calibrations": [],
        "calibration_point": None,
        "current_cycle_samples": [],
        "current_point": None,
        "current_cycle_max_case_tmp": None,
    }
    return m


def _make_estimator(
    ha,
    *,
    direct_tmp_entity="sensor.direct",
    power_entity="sensor.power",
    case_entity="sensor.case",
    area_entity="sensor.area",
    thermal_model=None,
):
    return TemperatureEstimator(
        ha=ha,
        switch_entity_id="switch.boiler",
        power_entity_id=power_entity,
        case_tmp_entity_id=case_entity,
        area_tmp_entity_id=area_entity,
        direct_tmp_entity_id=direct_tmp_entity,
        thermal_model=thermal_model,
        boiler_set_tmp=SET_TMP,
        area_tmp_default=AREA_TMP_DEFAULT,
    )


# ---------------------------------------------------------------------------
# L1 — Direct NTC probe
# ---------------------------------------------------------------------------

class TestL1DirectProbe:
    def test_returns_ntc_value_when_available(self):
        ha = _make_ha(direct_tmp=52.3)
        est = _make_estimator(ha)
        assert est.get_boiler_tmp(last_known=30.0) == 52.3

    def test_ntc_takes_priority_over_power_feedback(self):
        # Even if relay is ON + power < 50 (would trigger L2), L1 wins
        ha = _make_ha(direct_tmp=45.0, power=5.0, relay_on=True)
        est = _make_estimator(ha)
        assert est.get_boiler_tmp(last_known=10.0) == 45.0

    def test_falls_through_when_ntc_unavailable(self):
        # direct_tmp=None → L1 misses → should try L2
        ha = _make_ha(direct_tmp=None, power=5.0, relay_on=True)
        est = _make_estimator(ha)
        # L2 fires: relay ON + power < 50 → T_set
        assert est.get_boiler_tmp(last_known=10.0) == SET_TMP

    def test_falls_through_when_no_direct_entity_configured(self):
        ha = _make_ha(power=5.0, relay_on=True)
        est = _make_estimator(ha, direct_tmp_entity=None)
        # L1 skipped; L2 fires
        assert est.get_boiler_tmp(last_known=10.0) == SET_TMP


# ---------------------------------------------------------------------------
# L2 — Power feedback
# ---------------------------------------------------------------------------

class TestL2PowerFeedback:
    def test_thermostat_tripped_returns_set_tmp(self):
        """relay ON + power < 50 W → internal thermostat cut element → T_set."""
        ha = _make_ha(direct_tmp=None, power=3.0, relay_on=True)
        est = _make_estimator(ha, thermal_model=_make_thermal_model(estimate=55.0))
        result = est.get_boiler_tmp(last_known=40.0)
        assert result == SET_TMP

    def test_thermostat_tripped_at_threshold_boundary(self):
        """Power exactly at 49.9 W still counts as tripped."""
        ha = _make_ha(direct_tmp=None, power=49.9, relay_on=True)
        est = _make_estimator(ha)
        assert est.get_boiler_tmp(last_known=30.0) == SET_TMP

    def test_thermostat_tripped_power_zero(self):
        """Power = 0 W (idle relay draw) → thermostat tripped."""
        ha = _make_ha(direct_tmp=None, power=0.0, relay_on=True)
        est = _make_estimator(ha)
        assert est.get_boiler_tmp(last_known=30.0) == SET_TMP

    def test_actively_heating_skips_l3_uses_last_known(self):
        """
        relay ON + power > 50 W → boiler is actively heating.
        The thermal model is a COOLING equation and gives nonsense during
        active heating — it MUST be skipped.  Result must be last_known,
        not the thermal model estimate.
        """
        thermal = _make_thermal_model(estimate=99.0)  # wrong answer if L3 were called
        ha = _make_ha(direct_tmp=None, power=2000.0, relay_on=True, case_tmp=35.0)
        est = _make_estimator(ha, thermal_model=thermal)
        result = est.get_boiler_tmp(last_known=45.0)
        assert result == 45.0
        # Confirm thermal model was NOT consulted
        thermal.estimate_water_tmp.assert_not_called()

    def test_actively_heating_last_known_is_none(self):
        """When actively heating and last_known is None, None is returned (not a model value)."""
        thermal = _make_thermal_model(estimate=55.0)
        ha = _make_ha(direct_tmp=None, power=1800.0, relay_on=True, case_tmp=35.0)
        est = _make_estimator(ha, thermal_model=thermal)
        assert est.get_boiler_tmp(last_known=None) is None
        thermal.estimate_water_tmp.assert_not_called()

    def test_relay_off_falls_through_to_l3(self):
        """relay OFF → power reading is irrelevant; skip L2, try L3."""
        thermal = _make_thermal_model(estimate=48.5)
        ha = _make_ha(direct_tmp=None, power=0.0, relay_on=False, case_tmp=38.0)
        est = _make_estimator(ha, thermal_model=thermal)
        assert est.get_boiler_tmp(last_known=40.0) == 48.5
        thermal.estimate_water_tmp.assert_called_once()

    def test_relay_on_power_none_falls_to_l4(self):
        """relay ON but power entity returns None → ambiguous (heating or tripped?);
        cooling model is invalid → skip L3, fall to L4 (last_known)."""
        thermal = _make_thermal_model(estimate=52.0)
        ha = _make_ha(direct_tmp=None, power=None, relay_on=True, case_tmp=40.0)
        est = _make_estimator(ha, thermal_model=thermal)
        assert est.get_boiler_tmp(last_known=30.0) == 30.0
        thermal.estimate_water_tmp.assert_not_called()

    def test_no_power_entity_skips_l2_entirely(self):
        """If power_entity_id not configured, L2 is skipped and L3 is tried."""
        thermal = _make_thermal_model(estimate=50.0)
        ha = _make_ha(direct_tmp=None, case_tmp=40.0)
        est = _make_estimator(ha, power_entity=None, thermal_model=thermal)
        assert est.get_boiler_tmp(last_known=30.0) == 50.0
        thermal.estimate_water_tmp.assert_called_once()


# ---------------------------------------------------------------------------
# L3 — Thermal model
# ---------------------------------------------------------------------------

class TestL3ThermalModel:
    def test_uses_thermal_model_when_relay_off(self):
        thermal = _make_thermal_model(estimate=47.2)
        ha = _make_ha(direct_tmp=None, power=0.0, relay_on=False, case_tmp=36.0, area_tmp=19.0)
        est = _make_estimator(ha, thermal_model=thermal)
        result = est.get_boiler_tmp(last_known=40.0)
        assert result == 47.2
        thermal.estimate_water_tmp.assert_called_once_with(36.0, 19.0)

    def test_thermal_model_uses_default_ambient_when_no_area_entity(self):
        """If area_entity not configured, AREA_TMP_DEFAULT is passed to the model."""
        thermal = _make_thermal_model(estimate=50.0)
        ha = _make_ha(direct_tmp=None, power=0.0, relay_on=False, case_tmp=38.0)
        est = _make_estimator(ha, area_entity=None, thermal_model=thermal)
        est.get_boiler_tmp(last_known=30.0)
        thermal.estimate_water_tmp.assert_called_once_with(38.0, AREA_TMP_DEFAULT)

    def test_falls_through_when_thermal_model_returns_none(self):
        """Thermal model not yet calibrated → falls to L4."""
        thermal = _make_thermal_model(estimate=None)
        ha = _make_ha(direct_tmp=None, power=0.0, relay_on=False, case_tmp=38.0)
        est = _make_estimator(ha, thermal_model=thermal)
        assert est.get_boiler_tmp(last_known=33.3) == 33.3

    def test_falls_through_when_no_case_entity(self):
        """If case_tmp_entity_id not configured, L3 is skipped."""
        thermal = _make_thermal_model(estimate=50.0)
        ha = _make_ha(direct_tmp=None, power=0.0, relay_on=False)
        est = _make_estimator(ha, case_entity=None, thermal_model=thermal)
        assert est.get_boiler_tmp(last_known=28.0) == 28.0
        thermal.estimate_water_tmp.assert_not_called()

    def test_falls_through_when_no_thermal_model_object(self):
        """If thermal_model is None (not yet instantiated), L3 is skipped."""
        ha = _make_ha(direct_tmp=None, power=0.0, relay_on=False, case_tmp=38.0)
        est = _make_estimator(ha, thermal_model=None)
        assert est.get_boiler_tmp(last_known=28.0) == 28.0

    def test_falls_through_when_case_tmp_entity_returns_none(self):
        """Case sensor entity unavailable → L3 skipped."""
        thermal = _make_thermal_model(estimate=50.0)
        ha = _make_ha(direct_tmp=None, power=0.0, relay_on=False, case_tmp=None)
        est = _make_estimator(ha, thermal_model=thermal)
        assert est.get_boiler_tmp(last_known=28.0) == 28.0
        thermal.estimate_water_tmp.assert_not_called()

    def test_l3_skipped_when_relay_on_and_no_power_entity(self):
        """
        Regression for Chunk 8: without a power sensor L2 is bypassed entirely,
        so L3 was reached even while the relay was ON (actively heating).
        The cooling equation produces nonsense during active heating.
        Expected: fall to L4 (last_known), NOT the thermal-model estimate.
        """
        thermal = _make_thermal_model(estimate=52.0)
        ha = _make_ha(direct_tmp=None, power=None, relay_on=True, case_tmp=38.0, area_tmp=19.0)
        est = _make_estimator(ha, power_entity=None, thermal_model=thermal)
        result = est.get_boiler_tmp(last_known=45.0)
        assert result == 45.0, (
            f"Expected last_known=45.0 (cooling model must be skipped during active heating), "
            f"got {result}"
        )
        thermal.estimate_water_tmp.assert_not_called()

    def test_thermal_model_preview_usable_now_false_when_relay_on_no_power_sensor(self):
        """
        _build_thermal_model_preview must mark usable_now=False when relay is ON
        even if there is no power sensor — the cooling equation is still invalid.
        """
        thermal = _make_thermal_model(estimate=52.0)
        ha = _make_ha(direct_tmp=None, power=None, relay_on=True, case_tmp=38.0, area_tmp=19.0)
        est = _make_estimator(ha, power_entity=None, thermal_model=thermal)
        preview = est._build_thermal_model_preview(
            case_tmp=38.0,
            ambient_tmp=19.0,
            relay_on=True,
            power_w=None,
        )
        assert preview["usable_now"] is False, (
            f"Expected usable_now=False when relay is ON (no power sensor), got {preview['usable_now']}"
        )


# ---------------------------------------------------------------------------
# L4 — Last known value
# ---------------------------------------------------------------------------

class TestL4LastKnown:
    def test_returns_last_known_when_all_sensors_fail(self):
        ha = _make_ha(direct_tmp=None, power=None, relay_on=False, case_tmp=None)
        est = _make_estimator(ha, thermal_model=None)
        assert est.get_boiler_tmp(last_known=41.5) == 41.5

    def test_returns_none_when_last_known_also_none(self):
        ha = _make_ha(direct_tmp=None, power=None, relay_on=False, case_tmp=None)
        est = _make_estimator(ha, thermal_model=None)
        assert est.get_boiler_tmp(last_known=None) is None


# ---------------------------------------------------------------------------
# Diagnostic report
# ---------------------------------------------------------------------------

class TestDiagnosticReport:
    def test_report_marks_direct_sensor_as_source(self):
        ha = _make_ha(direct_tmp=52.3, area_tmp=19.0)
        est = _make_estimator(ha, thermal_model=_make_thermal_model(estimate=49.0))

        report = est.get_boiler_tmp_report(last_known=40.0)

        assert report["estimate"] == 52.3
        assert report["source_key"] == "direct_sensor"
        assert report["source_level"] == "L1"
        assert report["direct_sensor"]["value"] == 52.3
        assert report["ambient"]["value"] == 19.0

    def test_report_marks_active_heating_as_last_known_guard(self):
        thermal = _make_thermal_model(estimate=55.0)
        ha = _make_ha(direct_tmp=None, power=1800.0, relay_on=True, case_tmp=35.0, area_tmp=19.0)
        est = _make_estimator(ha, thermal_model=thermal)

        report = est.get_boiler_tmp_report(last_known=44.0)

        assert report["estimate"] == 44.0
        assert report["source_key"] == "actively_heating_last_known"
        assert report["power_feedback"]["actively_heating"] is True
        assert report["thermal_model_preview"]["usable_now"] is False
        assert "disabled during active heating" in report["thermal_model_preview"]["skip_reason"]

    def test_report_marks_thermal_model_path_when_used(self):
        thermal = _make_thermal_model(estimate=47.2)
        ha = _make_ha(direct_tmp=None, power=0.0, relay_on=False, case_tmp=36.0, area_tmp=18.5)
        est = _make_estimator(ha, thermal_model=thermal)

        report = est.get_boiler_tmp_report(last_known=41.0)

        assert report["estimate"] == 47.2
        assert report["source_key"] == "thermal_model"
        assert report["source_level"] == "L3"
        assert report["thermal_model_preview"]["estimate"] == 47.2

    def test_report_warns_when_last_known_has_no_timestamp(self):
        thermal = _make_thermal_model(estimate=None)
        ha = _make_ha(direct_tmp=None, power=None, relay_on=False, case_tmp=None)
        est = _make_estimator(ha, thermal_model=thermal)

        report = est.get_boiler_tmp_report(last_known=41.0)

        assert report["source_key"] == "last_known"
        assert any("no timestamp metadata" in msg for msg in report["warnings"])


# ---------------------------------------------------------------------------
# get_ambient_tmp
# ---------------------------------------------------------------------------

class TestAmbientTemperature:
    def test_returns_entity_value_when_available(self):
        ha = _make_ha(area_tmp=18.5)
        est = _make_estimator(ha)
        assert est.get_ambient_tmp() == 18.5

    def test_returns_default_when_entity_unavailable(self):
        ha = _make_ha(area_tmp=None)
        est = _make_estimator(ha)
        assert est.get_ambient_tmp() == AREA_TMP_DEFAULT

    def test_returns_default_when_no_entity_configured(self):
        ha = _make_ha()
        est = _make_estimator(ha, area_entity=None)
        assert est.get_ambient_tmp() == AREA_TMP_DEFAULT


# ---------------------------------------------------------------------------
# Priority chain integration — verify exact fallback order
# ---------------------------------------------------------------------------

class TestFallbackChain:
    def test_l1_beats_l2_beats_l3(self):
        """All three sources available — L1 wins."""
        thermal = _make_thermal_model(estimate=40.0)
        ha = _make_ha(direct_tmp=55.0, power=3.0, relay_on=True, case_tmp=38.0)
        est = _make_estimator(ha, thermal_model=thermal)
        assert est.get_boiler_tmp(last_known=30.0) == 55.0
        thermal.estimate_water_tmp.assert_not_called()

    def test_l2_beats_l3_when_l1_missing(self):
        """L1 unavailable; relay ON + power < 50 → L2 wins, L3 not called."""
        thermal = _make_thermal_model(estimate=40.0)
        ha = _make_ha(direct_tmp=None, power=4.0, relay_on=True, case_tmp=38.0)
        est = _make_estimator(ha, thermal_model=thermal)
        assert est.get_boiler_tmp(last_known=30.0) == SET_TMP
        thermal.estimate_water_tmp.assert_not_called()

    def test_l3_beats_l4_when_l1_l2_missing(self):
        """L1 & L2 unavailable; thermal model available → L3 wins."""
        thermal = _make_thermal_model(estimate=46.0)
        ha = _make_ha(direct_tmp=None, power=None, relay_on=False, case_tmp=37.0)
        est = _make_estimator(ha, thermal_model=thermal)
        assert est.get_boiler_tmp(last_known=30.0) == 46.0

    def test_l4_when_all_fail(self):
        ha = _make_ha(direct_tmp=None, power=None, relay_on=False, case_tmp=None)
        est = _make_estimator(ha, thermal_model=None)
        assert est.get_boiler_tmp(last_known=35.0) == 35.0

    def test_actively_heating_chain(self):
        """
        L1 missing; relay ON + power = 2000 W (active heating).
        L2 detects active heating → must return last_known WITHOUT calling L3.
        This is the key regression test for the fix: the cooling model must
        not be consulted during active heating.
        """
        thermal = _make_thermal_model(estimate=99.0)
        ha = _make_ha(direct_tmp=None, power=2000.0, relay_on=True, case_tmp=42.0)
        est = _make_estimator(ha, thermal_model=thermal)
        result = est.get_boiler_tmp(last_known=48.0)
        assert result == 48.0
        thermal.estimate_water_tmp.assert_not_called()
