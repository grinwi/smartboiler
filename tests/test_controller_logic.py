"""Unit tests for SmartBoilerController decision logic (no network / HA required)."""
import pytest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch, PropertyMock

from smartboiler.controller import SmartBoilerController


# ---------------------------------------------------------------------------
# Fixture: SmartBoilerController with all heavy deps mocked out
# ---------------------------------------------------------------------------

@pytest.fixture
def ctrl():
    """Return a SmartBoilerController with __init__ bypassed and state injected."""
    with patch.object(SmartBoilerController, "__init__", return_value=None):
        c = SmartBoilerController.__new__(SmartBoilerController)

    # Core component mocks
    c.ha = MagicMock()
    c.store = MagicMock()
    c.collector = MagicMock()
    c.predictor = MagicMock()
    c.hdo_learner = MagicMock()
    c.scheduler = MagicMock()
    c.spot_fetcher = None
    c._run_dashboard = MagicMock()

    # Config
    c.boiler_switch_entity_id = "switch.boiler"
    c.boiler_power_entity_id = None
    c.boiler_direct_tmp_entity_id = None
    c.boiler_case_tmp_entity_id = None
    c.boiler_volume_l = 120.0
    c.boiler_set_tmp = 60.0
    c.boiler_min_tmp = 37.0
    c.boiler_watt = 2000.0
    c.area_tmp = 20.0
    c.boiler_case_max_tmp = 40.0
    c.has_spot_price = False
    c.has_hdo = False
    c.hdo_explicit_schedule = ""
    c.prediction_conservatism = "medium"
    c.min_training_days = 30
    c.pv_surplus_entity_id = None

    # Mutable state
    import threading
    c._heating_plan = [False] * 24
    c._plan_slots = []
    c._forecast_24h = [0.0] * 24
    c._spot_prices = {}
    c._last_boiler_tmp = 50.0
    c._plan_generated_at = datetime.now()
    c._lock = threading.Lock()

    # Default: relay off, no recent legionella, no power sensor
    c.ha.is_entity_on.return_value = False
    c.ha.get_state_value.return_value = None
    c.store.get_last_legionella_heating.return_value = datetime.now()

    return c


# ---------------------------------------------------------------------------
# _interpolate_from_case_tmp
# ---------------------------------------------------------------------------

class TestInterpolateFromCaseTmp:
    def test_at_case_max_returns_set_tmp(self, ctrl):
        # area=20, case_max=40, set_tmp=60
        # ratio = (40-20)/(40-20) = 1.0  →  1.0*(60-20)+20 = 60
        assert ctrl._interpolate_from_case_tmp(40.0) == pytest.approx(60.0)

    def test_at_area_tmp_returns_area_tmp(self, ctrl):
        # ratio = 0  →  0*(60-20)+20 = 20
        assert ctrl._interpolate_from_case_tmp(20.0) == pytest.approx(20.0)

    def test_midpoint(self, ctrl):
        # case_tmp=30, ratio=(30-20)/(40-20)=0.5  →  0.5*(60-20)+20 = 40
        assert ctrl._interpolate_from_case_tmp(30.0) == pytest.approx(40.0)

    def test_below_area_tmp_returns_input(self, ctrl):
        # Below area_tmp: ratio numerator is negative → guarded
        result = ctrl._interpolate_from_case_tmp(10.0)
        assert result == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# _get_boiler_tmp — level selection
# ---------------------------------------------------------------------------

class TestGetBoilerTmp:
    def test_uses_direct_sensor_when_configured(self, ctrl):
        ctrl.boiler_direct_tmp_entity_id = "sensor.direct_tmp"
        ctrl.ha.get_state_value.return_value = 55.0
        assert ctrl._get_boiler_tmp() == pytest.approx(55.0)

    def test_falls_back_to_case_tmp_interpolation(self, ctrl):
        ctrl.boiler_case_tmp_entity_id = "sensor.case_tmp"
        ctrl.ha.get_state_value.return_value = 30.0  # midpoint → 40°C
        result = ctrl._get_boiler_tmp()
        assert result == pytest.approx(40.0)

    def test_returns_last_known_when_no_sensors(self, ctrl):
        ctrl._last_boiler_tmp = 48.0
        assert ctrl._get_boiler_tmp() == pytest.approx(48.0)

    def test_direct_sensor_takes_priority_over_case(self, ctrl):
        ctrl.boiler_direct_tmp_entity_id = "sensor.direct_tmp"
        ctrl.boiler_case_tmp_entity_id = "sensor.case_tmp"
        ctrl.ha.get_state_value.return_value = 52.0
        result = ctrl._get_boiler_tmp()
        assert result == pytest.approx(52.0)


# ---------------------------------------------------------------------------
# _current_plan_hour_index
# ---------------------------------------------------------------------------

class TestCurrentPlanHourIndex:
    def test_returns_zero_when_no_plan(self, ctrl):
        ctrl._plan_generated_at = None
        assert ctrl._current_plan_hour_index() == 0

    def test_returns_zero_immediately_after_plan(self, ctrl):
        ctrl._plan_generated_at = datetime.now()
        assert ctrl._current_plan_hour_index() == 0

    def test_returns_correct_index_after_two_hours(self, ctrl):
        ctrl._plan_generated_at = datetime.now() - timedelta(hours=2, minutes=5)
        assert ctrl._current_plan_hour_index() == 2

    def test_capped_at_23(self, ctrl):
        ctrl._plan_generated_at = datetime.now() - timedelta(hours=30)
        assert ctrl._current_plan_hour_index() == 23


# ---------------------------------------------------------------------------
# run_control_workflow — freeze protection
# ---------------------------------------------------------------------------

class TestFreezeProtection:
    def test_turns_on_when_below_5_degrees(self, ctrl):
        ctrl._last_boiler_tmp = 3.0
        ctrl.ha.get_state_value.return_value = None
        ctrl.ha.is_entity_on.return_value = False
        ctrl.run_control_workflow()
        ctrl.ha.turn_on.assert_called_with("switch.boiler")

    def test_does_not_call_turn_off_when_freezing(self, ctrl):
        ctrl._last_boiler_tmp = 2.0
        ctrl.ha.is_entity_on.return_value = False
        ctrl.run_control_workflow()
        ctrl.ha.turn_off.assert_not_called()


# ---------------------------------------------------------------------------
# run_control_workflow — legionella protection
# ---------------------------------------------------------------------------

class TestLegionellaProtection:
    def test_turns_on_when_overdue(self, ctrl):
        ctrl.store.get_last_legionella_heating.return_value = (
            datetime.now() - timedelta(days=22)
        )
        ctrl._last_boiler_tmp = 50.0
        ctrl.run_control_workflow()
        ctrl.ha.turn_on.assert_called_with("switch.boiler")

    def test_marks_complete_and_turns_off_at_65(self, ctrl):
        ctrl.store.get_last_legionella_heating.return_value = (
            datetime.now() - timedelta(days=22)
        )
        ctrl._last_boiler_tmp = 66.0
        ctrl.run_control_workflow()
        ctrl.store.set_last_legionella_heating.assert_called_once()
        ctrl.ha.turn_off.assert_called_with("switch.boiler")

    def test_no_legionella_action_when_recent(self, ctrl):
        ctrl.store.get_last_legionella_heating.return_value = datetime.now()
        ctrl._last_boiler_tmp = 50.0
        ctrl._heating_plan = [False] * 24
        ctrl.run_control_workflow()
        # Should NOT have turned on for legionella (may have other calls for plan)
        # Verify set_last_legionella_heating was not called
        ctrl.store.set_last_legionella_heating.assert_not_called()


# ---------------------------------------------------------------------------
# run_control_workflow — heating plan execution
# ---------------------------------------------------------------------------

class TestHeatingPlanExecution:
    def test_turns_on_when_plan_says_heat(self, ctrl):
        ctrl._plan_generated_at = datetime.now()
        ctrl._heating_plan = [True] + [False] * 23  # heat in hour 0
        ctrl._last_boiler_tmp = 45.0
        ctrl.run_control_workflow()
        ctrl.ha.turn_on.assert_called_with("switch.boiler")

    def test_turns_off_when_plan_says_idle(self, ctrl):
        ctrl._plan_generated_at = datetime.now()
        ctrl._heating_plan = [False] * 24  # no heating
        ctrl._last_boiler_tmp = 50.0
        ctrl.run_control_workflow()
        ctrl.ha.turn_off.assert_called_with("switch.boiler")

    def test_forces_on_below_min_tmp_even_if_plan_idle(self, ctrl):
        ctrl._plan_generated_at = datetime.now()
        ctrl._heating_plan = [False] * 24
        ctrl._last_boiler_tmp = 30.0  # below boiler_min_tmp=37
        ctrl.run_control_workflow()
        ctrl.ha.turn_on.assert_called_with("switch.boiler")
