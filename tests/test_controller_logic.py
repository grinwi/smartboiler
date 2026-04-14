"""Unit tests for SmartBoilerController decision logic (no network / HA required)."""
import pytest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch, PropertyMock

from smartboiler.controller import SmartBoilerController
from smartboiler.legionella_protector import LegionellaProtector


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
    c.calendar = None
    c._run_dashboard = MagicMock()

    # Config
    c.boiler_switch_entity_id = "switch.boiler"
    c.boiler_power_entity_id = None
    c.boiler_direct_tmp_entity_id = None
    c.boiler_case_tmp_entity_id = None
    c.boiler_area_tmp_entity_id = None
    c.boiler_volume_l = 120.0
    c.boiler_set_tmp = 60.0
    c.boiler_min_tmp = 37.0
    c.boiler_watt = 2000.0
    c.area_tmp = 20.0
    c.has_spot_price = False
    c.has_hdo = False
    c.hdo_explicit_schedule = ""
    c.hdo_history_weeks = 3
    c.hdo_decay_weeks = 2
    c.prediction_conservatism = "medium"
    c.min_training_days = 30
    c.pv_surplus_entity_id = None
    c.boiler_inlet_tmp_entity_id = None
    c.boiler_outlet_tmp_entity_id = None
    c.operation_mode = "standard"
    c.cold_water_temp = 10.0
    c.thermal_coupling = 0.45
    c.boiler_standby_w = 50.0
    c.draw_detection_thr_c = 2.0
    c.person_entity_ids = []
    c.vacation_mode = "min_temp"
    c.vacation_min_temp = 30.0
    c.flow_estimator = None

    # Thermal model mock
    c.thermal_model = MagicMock()
    c.thermal_model.estimate_water_tmp.return_value = None  # unfitted by default

    # TemperatureEstimator mock — simulates three-level logic for _get_boiler_tmp tests
    c.temp_estimator = MagicMock()

    def _simulated_get_boiler_tmp(last_known=None):
        if c.boiler_direct_tmp_entity_id:
            val = c.ha.get_state_value(c.boiler_direct_tmp_entity_id)
            if val is not None:
                return float(val)
        if c.boiler_case_tmp_entity_id:
            case_tmp = c.ha.get_state_value(c.boiler_case_tmp_entity_id)
            if case_tmp is not None:
                amb = (
                    c.ha.get_state_value(c.boiler_area_tmp_entity_id)
                    if c.boiler_area_tmp_entity_id else None
                ) or c.area_tmp
                est = c.thermal_model.estimate_water_tmp(float(case_tmp), amb)
                if est is not None:
                    return est
        return last_known

    c.temp_estimator.get_boiler_tmp.side_effect = _simulated_get_boiler_tmp
    c.temp_estimator.get_ambient_tmp.return_value = c.area_tmp

    # Real LegionellaProtector backed by mock ha/store
    c.legionella = LegionellaProtector(c.store, c.ha, "switch.boiler")

    # Mutable state
    import threading
    c._heating_plan = [False] * 24
    c._plan_slots = []
    c._forecast_24h = [0.0] * 24
    c._spot_prices = {}
    c._last_boiler_tmp = 50.0
    c._last_boiler_tmp_updated_at = datetime.now().astimezone()
    c._plan_generated_at = datetime.now().astimezone().astimezone()
    c._last_calib_ts = 0.0
    c._lock = threading.Lock()

    # Default: relay state "off" (available, thermostat cut), no recent legionella
    c.ha.get_state.return_value = {"state": "off", "attributes": {}}
    c.ha.is_entity_on.return_value = False
    c.ha.get_state_value.return_value = None
    c.store.get_last_legionella_heating.return_value = datetime.now().astimezone().astimezone()

    return c


# ---------------------------------------------------------------------------
# __init__
# ---------------------------------------------------------------------------

class TestControllerInit:
    def test_passes_configured_standby_loss_to_scheduler(self):
        options = {
            "boiler_switch_entity_id": "switch.boiler",
            "boiler_standby_watts": 83,
        }

        with patch("smartboiler.ha_client.HAClient"), \
             patch("smartboiler.state_store.StateStore") as state_store_cls, \
             patch("smartboiler.ha_data_collector.HADataCollector"), \
             patch("smartboiler.predictor.RollingHistogramPredictor"), \
             patch("smartboiler.hdo_learner.HDOLearner"), \
             patch("smartboiler.scheduler.HeatingScheduler") as scheduler_cls, \
             patch("smartboiler.spot_price.SpotPriceFetcher"), \
             patch("smartboiler.thermal_model.ThermalModel"), \
             patch("smartboiler.temperature_estimator.TemperatureEstimator"), \
             patch("smartboiler.legionella_protector.LegionellaProtector"), \
             patch("smartboiler.web_server.set_state_provider"), \
             patch("smartboiler.web_server.set_extra_provider"), \
             patch("smartboiler.web_server.set_calendar_manager"), \
             patch("smartboiler.web_server.set_influx_bootstrap_handlers"), \
             patch("smartboiler.web_server.run_dashboard"):
            store = state_store_cls.return_value
            store.load_pickle.return_value = None
            store.get_last_boiler_tmp.return_value = None

            SmartBoilerController(options)

        boiler_params = scheduler_cls.call_args[0][0]
        assert boiler_params.standby_loss_w == pytest.approx(83.0)


# ---------------------------------------------------------------------------
# HDO helpers
# ---------------------------------------------------------------------------

class TestHDOHelpers:
    def test_new_hdo_learner_uses_configured_window_and_explicit_schedule(self, ctrl):
        ctrl.hdo_history_weeks = 3
        ctrl.hdo_decay_weeks = 2
        ctrl.hdo_explicit_schedule = "22:00-06:00"

        learner = ctrl._new_hdo_learner()

        assert learner.history_weeks == 3
        assert learner.decay_weeks == 2
        assert learner.is_blocked(0, 22) is True
        assert learner.is_blocked(0, 12) is False

    def test_run_control_workflow_treats_unknown_state_as_hdo_unavailable(self, ctrl):
        ctrl.ha.get_state.return_value = {"state": "unknown", "attributes": {}}

        ctrl.run_control_workflow()

        assert ctrl.hdo_learner.observe.call_args[0][1] is True

    def test_start_influx_bootstrap_rebuilds_hdo_from_fresh_learner(self, ctrl):
        import threading

        class _ImmediateThread:
            def __init__(self, target, daemon=None, name=None):
                self._target = target

            def start(self):
                self._target()

        fresh_hdo = MagicMock()
        bootstrapper = MagicMock()
        bootstrapper.is_configured.return_value = True
        bootstrapper.run.return_value = {"hdo_observations": 12}

        ctrl._options = {"boiler_switch_entity_id": "switch.boiler"}
        ctrl._influx_bootstrap_lock = threading.Lock()
        ctrl._influx_bootstrap_status = {
            "available": True,
            "configured": False,
            "running": False,
            "source": "",
            "last_started_at": "",
            "last_finished_at": "",
            "last_error": "",
            "last_summary": {},
        }
        ctrl._build_influx_bootstrapper = MagicMock(return_value=bootstrapper)
        ctrl._new_hdo_learner = MagicMock(return_value=fresh_hdo)
        ctrl._refresh_after_influx_bootstrap = MagicMock(return_value={"predictor_ready": True})

        with patch("smartboiler.controller.threading.Thread", _ImmediateThread):
            result = ctrl.start_influx_bootstrap(source="test")

        assert result["ok"] is True
        bootstrapper.run.assert_called_once_with(hdo_learner=fresh_hdo)
        assert ctrl.hdo_learner is fresh_hdo


# ---------------------------------------------------------------------------
# Level 3 temperature estimation via ThermalModel
# (_interpolate_from_case_tmp was replaced by ThermalModel.estimate_water_tmp)
# ---------------------------------------------------------------------------

class TestThermalModelEstimation:
    def test_returns_thermal_model_estimate(self, ctrl):
        ctrl.boiler_case_tmp_entity_id = "sensor.case_tmp"
        ctrl.ha.get_state_value.return_value = 35.0
        ctrl.thermal_model.estimate_water_tmp.return_value = 52.0
        assert ctrl._get_boiler_tmp() == pytest.approx(52.0)

    def test_falls_back_to_last_known_when_model_returns_none(self, ctrl):
        ctrl.boiler_case_tmp_entity_id = "sensor.case_tmp"
        ctrl.ha.get_state_value.return_value = 35.0
        ctrl.thermal_model.estimate_water_tmp.return_value = None
        ctrl._last_boiler_tmp = 48.0
        assert ctrl._get_boiler_tmp() == pytest.approx(48.0)

    def test_skips_model_when_case_sensor_unavailable(self, ctrl):
        ctrl.boiler_case_tmp_entity_id = "sensor.case_tmp"
        ctrl.ha.get_state_value.return_value = None  # sensor offline
        ctrl._last_boiler_tmp = 45.0
        assert ctrl._get_boiler_tmp() == pytest.approx(45.0)
        ctrl.thermal_model.estimate_water_tmp.assert_not_called()

    def test_uses_live_ambient_when_area_sensor_configured(self, ctrl):
        ctrl.boiler_case_tmp_entity_id = "sensor.case_tmp"
        ctrl.boiler_area_tmp_entity_id = "sensor.area_tmp"
        ctrl.ha.get_state_value.side_effect = lambda eid: (
            35.0 if eid == "sensor.case_tmp" else 15.0
        )
        ctrl.thermal_model.estimate_water_tmp.return_value = 50.0
        ctrl._get_boiler_tmp()
        ctrl.thermal_model.estimate_water_tmp.assert_called_once_with(35.0, 15.0)


# ---------------------------------------------------------------------------
# _get_boiler_tmp — level selection
# ---------------------------------------------------------------------------

class TestGetBoilerTmp:
    def test_uses_direct_sensor_when_configured(self, ctrl):
        ctrl.boiler_direct_tmp_entity_id = "sensor.direct_tmp"
        ctrl.ha.get_state_value.return_value = 55.0
        assert ctrl._get_boiler_tmp() == pytest.approx(55.0)

    def test_falls_back_to_thermal_model_estimate(self, ctrl):
        ctrl.boiler_case_tmp_entity_id = "sensor.case_tmp"
        ctrl.ha.get_state_value.return_value = 30.0
        ctrl.thermal_model.estimate_water_tmp.return_value = 40.0
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


class TestTemperatureEstimationDiagnostics:
    def test_returns_estimator_report_with_controller_metadata(self, ctrl):
        ctrl.temp_estimator.get_boiler_tmp_report.return_value = {
            "estimate": 48.5,
            "source_key": "thermal_model",
            "source_level": "L3",
            "source_label": "Case sensor thermal model",
            "thermal_model_preview": {"mode": "fitted_case_decay"},
        }

        data = ctrl._get_temperature_estimation_data()

        ctrl.temp_estimator.get_boiler_tmp_report.assert_called_once_with(
            ctrl._last_boiler_tmp,
            last_known_updated_at=ctrl._last_boiler_tmp_updated_at,
        )
        assert data["estimate"] == 48.5
        assert data["source_key"] == "thermal_model"
        assert data["operation_mode"] == "standard"
        assert data["boiler_set_tmp"] == ctrl.boiler_set_tmp


# ---------------------------------------------------------------------------
# _current_plan_hour_index
# ---------------------------------------------------------------------------

class TestCurrentPlanHourIndex:
    def test_returns_zero_when_no_plan(self, ctrl):
        ctrl._plan_generated_at = None
        assert ctrl._current_plan_hour_index() == 0

    def test_returns_zero_immediately_after_plan(self, ctrl):
        ctrl._plan_generated_at = datetime.now().astimezone()
        assert ctrl._current_plan_hour_index() == 0

    def test_returns_correct_index_after_two_hours(self, ctrl):
        ctrl._plan_generated_at = datetime.now().astimezone() - timedelta(hours=2, minutes=5)
        assert ctrl._current_plan_hour_index() == 2

    def test_capped_at_23(self, ctrl):
        ctrl._plan_generated_at = datetime.now().astimezone() - timedelta(hours=30)
        assert ctrl._current_plan_hour_index() == 23


# ---------------------------------------------------------------------------
# Influx bootstrap refresh
# ---------------------------------------------------------------------------

class TestInfluxBootstrapRefresh:
    def test_refresh_updates_predictor_and_forecast(self, ctrl):
        import pandas as pd

        idx = pd.date_range(end=datetime.now().replace(minute=0, second=0, microsecond=0), periods=48, freq="1h")
        df = pd.DataFrame({"consumed_kwh": [0.2] * 48}, index=idx)
        ctrl.store.load_consumption_history.return_value = df
        ctrl.predictor._total_samples = 48
        ctrl.predictor.predict_next_24h.return_value = [0.3] * 24
        ctrl.predictor.has_enough_data.return_value = True
        ctrl.hdo_learner.observation_count.return_value = 15
        ctrl.hdo_learner.get_weekly_schedule.return_value = {
            "Mon": [1, 2],
            "Tue": [],
            "Wed": [],
            "Thu": [],
            "Fri": [],
            "Sat": [],
            "Sun": [],
        }

        info = ctrl._refresh_after_influx_bootstrap()

        ctrl.predictor.update.assert_called_once_with(df)
        ctrl.store.save_pickle.assert_any_call("predictor", ctrl.predictor)
        assert ctrl._forecast_24h == [0.3] * 24
        assert info["predictor_total_samples"] == 48
        assert info["predictor_ready"] is True
        assert info["hdo_total_observations"] == 15
        assert info["hdo_weekly_blocked_hours"] == 2


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
            datetime.now().astimezone().astimezone() - timedelta(days=22)
        )
        ctrl._last_boiler_tmp = 50.0
        ctrl.run_control_workflow()
        ctrl.ha.turn_on.assert_called_with("switch.boiler")

    def test_marks_complete_and_turns_off_at_65(self, ctrl):
        ctrl.store.get_last_legionella_heating.return_value = (
            datetime.now().astimezone().astimezone() - timedelta(days=22)
        )
        ctrl._last_boiler_tmp = 66.0
        ctrl.run_control_workflow()
        ctrl.store.set_last_legionella_heating.assert_called_once()
        ctrl.ha.turn_off.assert_called_with("switch.boiler")

    def test_no_legionella_action_when_recent(self, ctrl):
        ctrl.store.get_last_legionella_heating.return_value = datetime.now().astimezone().astimezone()
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
        ctrl._plan_generated_at = datetime.now().astimezone()
        ctrl._heating_plan = [True] + [False] * 23  # heat in hour 0
        ctrl._last_boiler_tmp = 45.0
        ctrl.run_control_workflow()
        ctrl.ha.turn_on.assert_called_with("switch.boiler")

    def test_turns_off_when_plan_says_idle(self, ctrl):
        ctrl._plan_generated_at = datetime.now().astimezone()
        ctrl._heating_plan = [False] * 24  # no heating
        ctrl._last_boiler_tmp = 50.0
        ctrl.run_control_workflow()
        ctrl.ha.turn_off.assert_called_with("switch.boiler")

    def test_forces_on_below_min_tmp_even_if_plan_idle(self, ctrl):
        ctrl._plan_generated_at = datetime.now().astimezone()
        ctrl._heating_plan = [False] * 24
        ctrl._last_boiler_tmp = 30.0  # below boiler_min_tmp=37
        ctrl.run_control_workflow()
        ctrl.ha.turn_on.assert_called_with("switch.boiler")
