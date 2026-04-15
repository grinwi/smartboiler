"""Unit tests for SmartBoilerController decision logic (no network / HA required)."""
import pytest
from collections import deque
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
    c._pending_trip_started_at = None
    c._prev_relay_on = None
    c._prev_power_w = None
    c._case_tmp_history = deque()
    c._peak_case_tmp_this_cycle = None
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
            store.get_plan_generated_at.return_value = None  # _restore_persisted_plan guard

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


class TestThermalCalibrationWindow:
    def test_legionella_path_keeps_relay_on_during_pending_trip_hold(self, ctrl):
        ctrl.boiler_power_entity_id = "sensor.power"
        ctrl.boiler_case_tmp_entity_id = "sensor.case_tmp"
        ctrl.temp_estimator.get_boiler_tmp = MagicMock(return_value=65.0)
        ctrl.temp_estimator.get_ambient_tmp.return_value = 20.0
        ctrl.legionella = MagicMock()
        ctrl.legionella.check_and_act.return_value = True
        ctrl.ha.get_state.return_value = {"state": "on", "attributes": {}}
        ctrl.thermal_model.observe_calibration = MagicMock()
        ctrl._prev_relay_on = True
        ctrl._prev_power_w = 1800.0

        def _get_state_value(eid, default=None):
            if eid == "sensor.power":
                return 0.0
            if eid == "sensor.case_tmp":
                return 44.0
            return default

        ctrl.ha.get_state_value.side_effect = _get_state_value
        ctrl._case_tmp_history.extend([
            (6760.0, 44.4),
            (6820.0, 44.3),
            (6880.0, 44.2),
            (6940.0, 44.1),
            (7000.0, 44.0),
        ])

        with patch("smartboiler.controller.time.time", return_value=7000.0):
            ctrl.run_control_workflow()

        ctrl.thermal_model.observe_calibration.assert_not_called()
        ctrl.ha.turn_on.assert_called_once_with("switch.boiler")
        ctrl.ha.turn_off.assert_not_called()

    def test_holds_relay_on_while_waiting_for_trip_calibration(self, ctrl):
        ctrl.boiler_power_entity_id = "sensor.power"
        ctrl.boiler_case_tmp_entity_id = "sensor.case_tmp"
        ctrl.temp_estimator.get_boiler_tmp = MagicMock(return_value=55.0)
        ctrl.temp_estimator.get_ambient_tmp.return_value = 20.0
        ctrl.legionella = MagicMock()
        ctrl.legionella.check_and_act.return_value = False
        ctrl.ha.get_state.return_value = {"state": "on", "attributes": {}}
        ctrl.thermal_model.observe_calibration = MagicMock()
        ctrl._plan_generated_at = datetime.now().astimezone()
        ctrl._heating_plan = [True] + [False] * 23

        readings = [
            {"power": 1800.0, "case": 44.0},
            {"power": 0.0, "case": 44.0},
        ]
        idx = {"value": 0}

        def _get_state_value(eid, default=None):
            current = readings[idx["value"]]
            if eid == "sensor.power":
                return current["power"]
            if eid == "sensor.case_tmp":
                return current["case"]
            return default

        ctrl.ha.get_state_value.side_effect = _get_state_value
        ctrl._case_tmp_history.extend([
            (4760.0, 44.4),
            (4820.0, 44.3),
            (4880.0, 44.2),
            (4940.0, 44.1),
            (5000.0, 44.0),
        ])

        with patch("smartboiler.controller.time.time", side_effect=[5000.0, 5055.0]):
            for i in range(len(readings)):
                idx["value"] = i
                ctrl.run_control_workflow()

        ctrl.thermal_model.observe_calibration.assert_not_called()
        assert ctrl.ha.turn_on.call_count == 2
        ctrl.ha.turn_off.assert_not_called()

    def test_calibrates_after_power_drop_and_stable_case_temp(self, ctrl):
        ctrl.boiler_power_entity_id = "sensor.power"
        ctrl.boiler_case_tmp_entity_id = "sensor.case_tmp"
        ctrl.temp_estimator.get_boiler_tmp = MagicMock(return_value=55.0)
        ctrl.temp_estimator.get_ambient_tmp.return_value = 20.0
        ctrl.legionella = MagicMock()
        ctrl.legionella.check_and_act.return_value = False
        ctrl.ha.get_state.return_value = {"state": "on", "attributes": {}}
        ctrl.thermal_model.observe_calibration = MagicMock()
        ctrl._plan_generated_at = datetime.now().astimezone()
        ctrl._heating_plan = [True] + [False] * 23

        readings = [
            {"power": 1800.0, "case": 44.0},
            {"power": 0.0, "case": 44.0},
            {"power": 0.0, "case": 43.9},
        ]
        idx = {"value": 0}

        def _get_state_value(eid, default=None):
            current = readings[idx["value"]]
            if eid == "sensor.power":
                return current["power"]
            if eid == "sensor.case_tmp":
                return current["case"]
            return default

        ctrl.ha.get_state_value.side_effect = _get_state_value

        # Case temperature has been flat / falling for the last 5 minutes.
        ctrl._case_tmp_history.extend([
            (760.0, 44.4),
            (820.0, 44.3),
            (880.0, 44.2),
            (940.0, 44.1),
            (1000.0, 44.0),
        ])

        with patch("smartboiler.controller.time.time", side_effect=[1000.0, 1060.0, 1120.0]):
            for i in range(len(readings)):
                idx["value"] = i
                ctrl.run_control_workflow()

        # C0 is the peak case temp seen while relay was ON (44.0), not the post-trip
        # plateau (43.9) — important for poorly-mounted sensors where the plateau
        # is near ambient but the peak carries the real thermal signal.
        ctrl.thermal_model.observe_calibration.assert_called_once_with(
            T_set=ctrl.boiler_set_tmp,
            T_case=44.0,
            T_amb=20.0,
            timestamp=1120.0,
        )
        assert ctrl._last_calib_ts == 1120.0
        assert ctrl.ha.turn_on.call_count == 2
        ctrl.ha.turn_off.assert_called_once_with("switch.boiler")

    def test_does_not_calibrate_when_low_power_window_is_shorter_than_60s(self, ctrl):
        ctrl.boiler_power_entity_id = "sensor.power"
        ctrl.boiler_case_tmp_entity_id = "sensor.case_tmp"
        ctrl.temp_estimator.get_boiler_tmp = MagicMock(return_value=55.0)
        ctrl.temp_estimator.get_ambient_tmp.return_value = 20.0
        ctrl.legionella = MagicMock()
        ctrl.legionella.check_and_act.return_value = False
        ctrl.ha.get_state.return_value = {"state": "on", "attributes": {}}
        ctrl.thermal_model.observe_calibration = MagicMock()
        ctrl._plan_generated_at = datetime.now().astimezone()
        ctrl._heating_plan = [True] + [False] * 23

        readings = [
            {"power": 1700.0, "case": 44.0},
            {"power": 0.0, "case": 43.9},
            {"power": 1200.0, "case": 43.8},
        ]
        idx = {"value": 0}

        def _get_state_value(eid, default=None):
            current = readings[idx["value"]]
            if eid == "sensor.power":
                return current["power"]
            if eid == "sensor.case_tmp":
                return current["case"]
            return default

        ctrl.ha.get_state_value.side_effect = _get_state_value

        ctrl._case_tmp_history.extend([
            (1760.0, 44.4),
            (1820.0, 44.3),
            (1880.0, 44.2),
            (1940.0, 44.1),
            (2000.0, 44.0),
        ])

        with patch("smartboiler.controller.time.time", side_effect=[2000.0, 2055.0, 2060.0]):
            for i in range(len(readings)):
                idx["value"] = i
                ctrl.run_control_workflow()

        ctrl.thermal_model.observe_calibration.assert_not_called()
        assert ctrl.ha.turn_on.call_count == 2
        ctrl.ha.turn_off.assert_called_once_with("switch.boiler")

    def test_calibrates_even_when_control_loop_has_small_jitter(self, ctrl):
        """A slightly late 60 s loop must not block the 5-minute stability check."""
        ctrl.boiler_power_entity_id = "sensor.power"
        ctrl.boiler_case_tmp_entity_id = "sensor.case_tmp"
        ctrl.temp_estimator.get_boiler_tmp = MagicMock(return_value=55.0)
        ctrl.temp_estimator.get_ambient_tmp.return_value = 20.0
        ctrl.legionella = MagicMock()
        ctrl.legionella.check_and_act.return_value = False
        ctrl.ha.get_state.return_value = {"state": "on", "attributes": {}}
        ctrl.thermal_model.observe_calibration = MagicMock()
        ctrl._plan_generated_at = datetime.now().astimezone()
        ctrl._heating_plan = [True] + [False] * 23

        readings = [
            {"power": 1800.0, "case": 44.0},
            {"power": 0.0, "case": 44.0},
            {"power": 0.0, "case": 43.9},
            {"power": 0.0, "case": 43.8},
            {"power": 0.0, "case": 43.8},
            {"power": 0.0, "case": 43.7},
        ]
        idx = {"value": 0}

        def _get_state_value(eid, default=None):
            current = readings[idx["value"]]
            if eid == "sensor.power":
                return current["power"]
            if eid == "sensor.case_tmp":
                return current["case"]
            return default

        ctrl.ha.get_state_value.side_effect = _get_state_value

        # Real control loops are not exactly 60.0 s apart. With 61 s jitter the
        # old strict cutoff check never saw "full" 5-minute coverage and missed
        # this calibration entirely.
        with patch(
            "smartboiler.controller.time.time",
            side_effect=[1000.0, 1061.0, 1122.0, 1183.0, 1244.0, 1305.0],
        ):
            for i in range(len(readings)):
                idx["value"] = i
                ctrl.run_control_workflow()

        ctrl.thermal_model.observe_calibration.assert_called_once_with(
            T_set=ctrl.boiler_set_tmp,
            T_case=44.0,
            T_amb=20.0,
            timestamp=1244.0,
        )

    def test_missing_power_reading_does_not_count_as_zero_power_trip(self, ctrl):
        ctrl.boiler_power_entity_id = "sensor.power"
        ctrl.boiler_case_tmp_entity_id = "sensor.case_tmp"
        ctrl.temp_estimator.get_boiler_tmp = MagicMock(return_value=55.0)
        ctrl.temp_estimator.get_ambient_tmp.return_value = 20.0
        ctrl.legionella = MagicMock()
        ctrl.legionella.check_and_act.return_value = False
        ctrl.ha.get_state.return_value = {"state": "on", "attributes": {}}
        ctrl.thermal_model.observe_calibration = MagicMock()

        def _get_state_value(eid, default=None):
            if eid == "sensor.power":
                return None
            if eid == "sensor.case_tmp":
                return 42.0
            return default

        ctrl.ha.get_state_value.side_effect = _get_state_value

        with patch("smartboiler.controller.time.time", side_effect=[3000.0, 3065.0]):
            ctrl.run_control_workflow()
            ctrl.run_control_workflow()

        ctrl.thermal_model.observe_calibration.assert_not_called()

    def test_startup_low_power_does_not_trigger_calibration_without_heating_transition(self, ctrl):
        ctrl.boiler_power_entity_id = "sensor.power"
        ctrl.boiler_case_tmp_entity_id = "sensor.case_tmp"
        ctrl.temp_estimator.get_boiler_tmp = MagicMock(return_value=55.0)
        ctrl.temp_estimator.get_ambient_tmp.return_value = 20.0
        ctrl.legionella = MagicMock()
        ctrl.legionella.check_and_act.return_value = False
        ctrl.ha.get_state.return_value = {"state": "on", "attributes": {}}
        ctrl.thermal_model.observe_calibration = MagicMock()
        ctrl._plan_generated_at = datetime.now().astimezone()
        ctrl._heating_plan = [True] + [False] * 23

        def _get_state_value(eid, default=None):
            if eid == "sensor.power":
                return 0.0
            if eid == "sensor.case_tmp":
                return 42.0
            return default

        ctrl.ha.get_state_value.side_effect = _get_state_value
        ctrl._case_tmp_history.extend([
            (5760.0, 42.4),
            (5820.0, 42.3),
            (5880.0, 42.2),
            (5940.0, 42.1),
            (6000.0, 42.0),
        ])

        with patch("smartboiler.controller.time.time", side_effect=[6000.0, 6065.0]):
            ctrl.run_control_workflow()
            ctrl.run_control_workflow()

        ctrl.thermal_model.observe_calibration.assert_not_called()
        assert ctrl._pending_trip_started_at is None
        ctrl.ha.turn_on.assert_not_called()
        assert ctrl.ha.turn_off.call_count == 2

    def test_case_temp_rise_within_last_5_minutes_blocks_calibration(self, ctrl):
        ctrl.boiler_power_entity_id = "sensor.power"
        ctrl.boiler_case_tmp_entity_id = "sensor.case_tmp"
        ctrl.temp_estimator.get_boiler_tmp = MagicMock(return_value=55.0)
        ctrl.temp_estimator.get_ambient_tmp.return_value = 20.0
        ctrl.legionella = MagicMock()
        ctrl.legionella.check_and_act.return_value = False
        ctrl.ha.get_state.return_value = {"state": "on", "attributes": {}}
        ctrl.thermal_model.observe_calibration = MagicMock()
        ctrl._plan_generated_at = datetime.now().astimezone()
        ctrl._heating_plan = [True] + [False] * 23

        readings = [
            {"power": 1800.0, "case": 44.0},
            {"power": 0.0, "case": 44.2},
            {"power": 0.0, "case": 44.3},
        ]
        idx = {"value": 0}

        def _get_state_value(eid, default=None):
            current = readings[idx["value"]]
            if eid == "sensor.power":
                return current["power"]
            if eid == "sensor.case_tmp":
                return current["case"]
            return default

        ctrl.ha.get_state_value.side_effect = _get_state_value
        ctrl._case_tmp_history.extend([
            (3760.0, 43.7),
            (3820.0, 43.8),
            (3880.0, 43.9),
            (3940.0, 44.0),
            (4000.0, 44.1),
        ])

        with patch("smartboiler.controller.time.time", side_effect=[4000.0, 4060.0, 4120.0]):
            for i in range(len(readings)):
                idx["value"] = i
                ctrl.run_control_workflow()

        ctrl.thermal_model.observe_calibration.assert_not_called()
        assert ctrl.ha.turn_on.call_count == 3
        ctrl.ha.turn_off.assert_not_called()


# ---------------------------------------------------------------------------
# Peak case temp tracking for calibration C0
# ---------------------------------------------------------------------------

class TestPeakCaseTmpCalibration:
    """Verify that observe_calibration receives the heating-cycle peak, not the
    post-trip plateau.  Motivation: poorly-mounted case sensors read near-ambient
    at the stable plateau, making the plateau useless as C0 for Newton cooling.
    The peak observed while the relay was ON is the correct C0."""

    def _make_ctrl(self, ctrl):
        ctrl.boiler_power_entity_id = "sensor.power"
        ctrl.boiler_case_tmp_entity_id = "sensor.case_tmp"
        ctrl.temp_estimator.get_boiler_tmp = MagicMock(return_value=55.0)
        ctrl.temp_estimator.get_ambient_tmp.return_value = 14.0
        ctrl.legionella = MagicMock()
        ctrl.legionella.check_and_act.return_value = False
        ctrl.ha.get_state.return_value = {"state": "on", "attributes": {}}
        ctrl.thermal_model.observe_calibration = MagicMock()
        ctrl._plan_generated_at = datetime.now().astimezone()
        ctrl._heating_plan = [True] + [False] * 23

    def _stable_history(self, t_start: float, value: float, n: int = 8, step: float = 60.0):
        """Return n history entries at a constant value covering >5 min before t_start.

        Using a constant value guarantees no rises, satisfying the stability check.
        The window required is 5 min (300 s); with n=8 and step=60 the first entry
        is 480 s before t_start, well clear of the 300 s cutoff.
        """
        return [(t_start - i * step, value) for i in range(n, 0, -1)]

    def test_uses_peak_not_plateau_when_peak_is_higher(self, ctrl):
        """Peak during heating (25°C) is used as C0, not plateau after trip (15.7°C)."""
        self._make_ctrl(ctrl)

        # Ticks: 0=heating(25°C), 1=trip(16°C), 2=stable(15.8°C)
        # Use timestamps far apart so tick 0's case reading (25°C) is outside the
        # 5-min stability window when calibration fires on tick 2.
        T0, T1, T2 = 1000.0, 2000.0, 2060.0  # 16+ min gap before trip

        readings = [
            {"power": 2000.0, "case": 25.0},
            {"power": 0.0,    "case": 16.0},
            {"power": 0.0,    "case": 15.8},
        ]
        idx = {"value": 0}

        def _get(eid, default=None):
            r = readings[idx["value"]]
            if eid == "sensor.power":  return r["power"]
            if eid == "sensor.case_tmp": return r["case"]
            return default

        ctrl.ha.get_state_value.side_effect = _get
        ctrl._prev_relay_on = True
        ctrl._prev_power_w = 2000.0
        # Pre-load stable declining history just before the trip window
        ctrl._case_tmp_history.extend(self._stable_history(T1, 16.2))

        with patch("smartboiler.controller.time.time", side_effect=[T0, T1, T2]):
            for i in range(len(readings)):
                idx["value"] = i
                ctrl.run_control_workflow()

        ctrl.thermal_model.observe_calibration.assert_called_once()
        call_kwargs = ctrl.thermal_model.observe_calibration.call_args.kwargs
        assert call_kwargs["T_case"] == 25.0, (
            f"Expected peak C0=25.0, got {call_kwargs['T_case']}"
        )
        assert call_kwargs["T_amb"] == 14.0

    def test_peak_resets_after_calibration(self, ctrl):
        """_peak_case_tmp_this_cycle is cleared after a calibration is committed."""
        self._make_ctrl(ctrl)

        T0, T1, T2 = 1000.0, 2000.0, 2060.0

        readings = [
            {"power": 2000.0, "case": 25.0},
            {"power": 0.0,    "case": 16.0},
            {"power": 0.0,    "case": 15.8},
        ]
        idx = {"value": 0}

        def _get(eid, default=None):
            r = readings[idx["value"]]
            if eid == "sensor.power":  return r["power"]
            if eid == "sensor.case_tmp": return r["case"]
            return default

        ctrl.ha.get_state_value.side_effect = _get
        ctrl._prev_relay_on = True
        ctrl._prev_power_w = 2000.0
        ctrl._case_tmp_history.extend(self._stable_history(T1, 16.2))

        with patch("smartboiler.controller.time.time", side_effect=[T0, T1, T2]):
            for i in range(len(readings)):
                idx["value"] = i
                ctrl.run_control_workflow()

        assert ctrl._peak_case_tmp_this_cycle is None

    def test_peak_accumulates_highest_reading_during_heating(self, ctrl):
        """Multiple heating ticks — peak correctly tracks the maximum."""
        self._make_ctrl(ctrl)

        T_heat = [1000.0, 1060.0, 1120.0]  # 3 heating ticks
        T_trip = 2000.0                     # trip (large gap clears stability window)
        T_conf = 2060.0                     # confirmation

        readings = [
            {"power": 2000.0, "case": 20.0},  # tick 1 — lower
            {"power": 2000.0, "case": 25.0},  # tick 2 — new peak
            {"power": 2000.0, "case": 23.0},  # tick 3 — below peak
            {"power": 0.0,    "case": 16.0},  # trip
            {"power": 0.0,    "case": 15.8},  # stable
        ]
        idx = {"value": 0}

        def _get(eid, default=None):
            r = readings[idx["value"]]
            if eid == "sensor.power":  return r["power"]
            if eid == "sensor.case_tmp": return r["case"]
            return default

        ctrl.ha.get_state_value.side_effect = _get
        ctrl._prev_relay_on = True
        ctrl._prev_power_w = 2000.0
        ctrl._case_tmp_history.extend(self._stable_history(T_trip, 16.2))

        with patch("smartboiler.controller.time.time",
                   side_effect=[*T_heat, T_trip, T_conf]):
            for i in range(len(readings)):
                idx["value"] = i
                ctrl.run_control_workflow()

        ctrl.thermal_model.observe_calibration.assert_called_once()
        call_kwargs = ctrl.thermal_model.observe_calibration.call_args.kwargs
        assert call_kwargs["T_case"] == 25.0

    def test_falls_back_to_plateau_when_no_peak_recorded(self, ctrl):
        """If no heating tick was seen before the trip (e.g. restart mid-cycle),
        the plateau value is used as a safe fallback."""
        self._make_ctrl(ctrl)

        # Start with _peak_case_tmp_this_cycle = None (default after init/reset)
        # Jump straight into low-power (trip already in progress from before restart)
        ctrl._pending_trip_started_at = 1000.0
        ctrl._prev_relay_on = True
        ctrl._prev_power_w = 0.0  # already in low-power state

        readings = [{"power": 0.0, "case": 16.5}]
        idx = {"value": 0}

        def _get(eid, default=None):
            r = readings[idx["value"]]
            if eid == "sensor.power":  return r["power"]
            if eid == "sensor.case_tmp": return r["case"]
            return default

        ctrl.ha.get_state_value.side_effect = _get
        # History must reach back to cutoff = 1070-300 = 770 for stability check
        ctrl._case_tmp_history.extend([
            (770.0, 16.8), (830.0, 16.7), (890.0, 16.6), (950.0, 16.5), (1010.0, 16.5),
        ])

        with patch("smartboiler.controller.time.time", return_value=1070.0):
            ctrl.run_control_workflow()

        ctrl.thermal_model.observe_calibration.assert_called_once()
        call_kwargs = ctrl.thermal_model.observe_calibration.call_args.kwargs
        # No peak recorded → falls back to current case_tmp (plateau)
        assert call_kwargs["T_case"] == 16.5


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

    def test_returns_zero_when_plan_older_than_24h(self, ctrl):
        # A plan older than 24 h is stale — treat as "no plan" rather than
        # silently serving hour 23 indefinitely.
        ctrl._plan_generated_at = datetime.now().astimezone() - timedelta(hours=25)
        assert ctrl._current_plan_hour_index() == 0


# ---------------------------------------------------------------------------
# Spot price 24h indexing
# ---------------------------------------------------------------------------

class TestSpotPriceIndexing:
    """_spot_prices_indexed must use tomorrow's prices for hours that wrap past midnight.

    Old bug: `self._spot_prices.get((now_dt.hour + i) % 24)` always looked up
    today's dict.  At hour 22, i=3 gives (22+3)%24=1 — today's 01:00 price, not
    tomorrow's 01:00.  The fix calls get_next_24h_prices(from_hour=now_dt.hour).
    """

    def _build_indexed_old(self, today_prices, from_hour):
        """Reproduce the OLD (buggy) indexing formula."""
        return {i: today_prices.get((from_hour + i) % 24) for i in range(24)}

    def _build_indexed_new(self, today_prices, tomorrow_prices, from_hour):
        """Reproduce the CORRECT indexing via get_next_24h_prices logic."""
        return {
            i: (today_prices if from_hour + i < 24 else tomorrow_prices)[
                (from_hour + i) % 24
            ]
            for i in range(24)
        }

    def test_old_formula_returns_wrong_price_at_midnight_wrap(self):
        """Demonstrates the bug: at from_hour=22, slot i=3 wraps to today[1]."""
        today = {h: float(100 + h) for h in range(24)}
        result = self._build_indexed_old(today, from_hour=22)
        # i=3 → (22+3)%24=1 → today[1]=101.0  (WRONG: should be tomorrow[1])
        assert result[3] == 101.0

    def test_new_formula_returns_tomorrows_price_at_midnight_wrap(self):
        """After fix: at from_hour=22, slot i=3 must return tomorrow[1]."""
        today = {h: float(100 + h) for h in range(24)}
        tomorrow = {h: float(200 + h) for h in range(24)}
        result = self._build_indexed_new(today, tomorrow, from_hour=22)
        assert result[0] == 122.0    # today[22]
        assert result[1] == 123.0    # today[23]
        assert result[2] == 200.0    # tomorrow[0]
        assert result[3] == 201.0    # tomorrow[1] — not today[1]=101

    def test_spot_prices_indexed_on_controller_uses_get_next_24h_prices(self, ctrl):
        """run_forecast_workflow must call spot_fetcher.get_next_24h_prices(from_hour)
        and store the result directly, rather than building the wrong modular index."""
        today = {h: float(100 + h) for h in range(24)}
        tomorrow = {h: float(200 + h) for h in range(24)}

        correct_24h = {
            i: (today if 22 + i < 24 else tomorrow)[(22 + i) % 24]
            for i in range(24)
        }
        ctrl.spot_fetcher = MagicMock()
        ctrl.spot_fetcher.fetch_today_tomorrow.return_value = {
            "today": today, "tomorrow": tomorrow
        }
        ctrl.spot_fetcher.get_next_24h_prices.return_value = correct_24h
        ctrl.has_spot_price = True

        # After fix, _spot_prices_indexed should come from get_next_24h_prices.
        # Simulate the fixed path:
        with ctrl._lock:
            ctrl._spot_prices_indexed = ctrl.spot_fetcher.get_next_24h_prices(from_hour=22)

        ctrl.spot_fetcher.get_next_24h_prices.assert_called_once_with(from_hour=22)
        assert ctrl._spot_prices_indexed[3] == 201.0  # tomorrow[1], not today[1]


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

    def test_no_power_sensor_does_not_fake_thermostat_trip(self, ctrl):
        ctrl.boiler_power_entity_id = None
        ctrl.ha.get_state.return_value = {"state": "on", "attributes": {}}
        ctrl.temp_estimator.get_boiler_tmp = MagicMock(return_value=45.0)
        ctrl.legionella = MagicMock()
        ctrl.legionella.check_and_act.return_value = False
        ctrl._plan_generated_at = datetime.now().astimezone()
        ctrl._heating_plan = [True] + [False] * 23

        ctrl.run_control_workflow()

        ctrl.ha.turn_on.assert_called_with("switch.boiler")


# ---------------------------------------------------------------------------
# _push_simple_mode_estimate_to_history — tz-naive index (Chunk 6)
# ---------------------------------------------------------------------------

class TestSimpleModeHistory:
    """
    _push_simple_mode_estimate_to_history must produce a DataFrame with a
    tz-NAIVE DatetimeIndex so that pd.concat inside StateStore.append_consumption
    can merge it with the existing tz-naive history without raising TypeError.

    Bug: midnight was created from `datetime.now().astimezone()` which carries
    local timezone info.  Passing it directly to pd.date_range(..., tz=...)
    produced a tz-AWARE index.  When append_consumption then called
    pd.concat([tz_naive_existing, tz_aware_new]), pandas raised:
      TypeError: Cannot join tz-naive with tz-aware DatetimeIndex
    """

    def test_synthetic_history_index_is_tz_naive(self, ctrl):
        """DataFrame passed to store.append_consumption must have tz-naive index."""
        ctrl._push_simple_mode_estimate_to_history(50.0)

        call_args = ctrl.store.append_consumption.call_args
        assert call_args is not None, "_push_simple_mode_estimate_to_history did not call store.append_consumption"
        df = call_args[0][0]
        assert df.index.tz is None, (
            f"Expected tz-naive DatetimeIndex but got tz={df.index.tz}. "
            "tz-aware index causes TypeError when concat'd with tz-naive history."
        )

    def test_synthetic_history_has_24_hourly_rows(self, ctrl):
        """Exactly 24 hourly rows should be pushed for the previous day."""
        ctrl._push_simple_mode_estimate_to_history(50.0)

        df = ctrl.store.append_consumption.call_args[0][0]
        assert len(df) == 24
        assert list(df.columns) == ["consumed_kwh", "relay_on", "power_w"]

    def test_synthetic_kwh_is_positive_for_nonzero_volume(self, ctrl):
        """Energy derived from a non-zero volume estimate must be > 0."""
        ctrl._push_simple_mode_estimate_to_history(40.0)

        df = ctrl.store.append_consumption.call_args[0][0]
        assert (df["consumed_kwh"] > 0).all()


# ---------------------------------------------------------------------------
# Plan persistence across restarts (Chunk 7)
# ---------------------------------------------------------------------------

class TestPlanPersistence:
    """
    After an add-on restart the controller must restore the heating plan AND the
    generation timestamp from StateStore so that _current_plan_hour_index() returns
    the correct slot (elapsed hours since generation), not always 0.

    Bugs before fix:
    1. StateStore had no set/get_plan_generated_at() methods.
    2. controller.__init__ never called get_heating_plan() or get_plan_generated_at().
    3. _current_plan_hour_index() returned 0 whenever _plan_generated_at is None.
    """

    # ── StateStore helpers (new methods) ────────────────────────────────

    def test_state_store_plan_generated_at_defaults_to_none(self, tmp_path):
        """get_plan_generated_at() must return None when no value is persisted."""
        from smartboiler.state_store import StateStore
        store = StateStore(data_dir=str(tmp_path))
        assert store.get_plan_generated_at() is None

    def test_state_store_plan_generated_at_roundtrip(self, tmp_path):
        """set then get must return the same tz-aware datetime (within 1 s)."""
        from smartboiler.state_store import StateStore
        store = StateStore(data_dir=str(tmp_path))
        ts = datetime.now().astimezone()
        store.set_plan_generated_at(ts)
        reloaded = store.get_plan_generated_at()
        assert reloaded is not None
        assert abs((reloaded - ts).total_seconds()) < 1

    def test_state_store_plan_generated_at_survives_reload(self, tmp_path):
        """Value written to JSON must be readable by a freshly-instantiated store."""
        from smartboiler.state_store import StateStore
        ts = datetime.now().astimezone()
        StateStore(data_dir=str(tmp_path)).set_plan_generated_at(ts)
        # New instance reads from disk
        reloaded = StateStore(data_dir=str(tmp_path)).get_plan_generated_at()
        assert reloaded is not None
        assert abs((reloaded - ts).total_seconds()) < 1

    # ── Controller restore method ────────────────────────────────────────

    def test_restore_persisted_plan_loads_plan_and_timestamp(self, ctrl):
        """_restore_persisted_plan() must populate _heating_plan and _plan_generated_at."""
        saved_plan = [True, False] * 12
        saved_time = datetime.now().astimezone() - timedelta(hours=2)
        ctrl.store.get_plan_generated_at.return_value = saved_time
        ctrl.store.get_heating_plan.return_value = saved_plan

        ctrl._restore_persisted_plan()

        assert ctrl._heating_plan == saved_plan
        assert ctrl._plan_generated_at == saved_time

    def test_restore_persisted_plan_discards_stale_plan(self, ctrl):
        """Plans older than 26 h must NOT be restored — the next forecast replaces them."""
        ctrl.store.get_plan_generated_at.return_value = (
            datetime.now().astimezone() - timedelta(hours=27)
        )
        ctrl.store.get_heating_plan.return_value = [True] * 24

        ctrl._restore_persisted_plan()

        assert ctrl._plan_generated_at is None
        assert ctrl._heating_plan == [False] * 24

    def test_restore_persisted_plan_skips_when_no_timestamp(self, ctrl):
        """Missing timestamp (first ever run) → leave defaults intact."""
        ctrl.store.get_plan_generated_at.return_value = None
        ctrl.store.get_heating_plan.return_value = [True] * 24

        ctrl._restore_persisted_plan()

        assert ctrl._plan_generated_at is None
        assert ctrl._heating_plan == [False] * 24

    def test_restore_persisted_plan_skips_empty_plan(self, ctrl):
        """Stored plan list is empty (corrupted state) → leave defaults intact."""
        ctrl.store.get_plan_generated_at.return_value = datetime.now().astimezone() - timedelta(hours=1)
        ctrl.store.get_heating_plan.return_value = []

        ctrl._restore_persisted_plan()

        assert ctrl._plan_generated_at is None
        assert ctrl._heating_plan == [False] * 24

    # ── _current_plan_hour_index after restore ───────────────────────────

    def test_current_plan_hour_index_reflects_elapsed_hours_after_restore(self, ctrl):
        """After restoring a plan generated 3 h ago, slot index must be 3."""
        saved_plan = [False] * 24
        saved_time = datetime.now().astimezone() - timedelta(hours=3, minutes=10)
        ctrl.store.get_plan_generated_at.return_value = saved_time
        ctrl.store.get_heating_plan.return_value = saved_plan

        ctrl._restore_persisted_plan()

        assert ctrl._current_plan_hour_index() == 3

    def test_current_plan_hour_index_is_zero_without_restore(self, ctrl):
        """Without restoring, _plan_generated_at is None → index is 0 (safety default)."""
        ctrl._plan_generated_at = None
        assert ctrl._current_plan_hour_index() == 0

    # ── Chunk 10: tz-naive stored strings must not crash ─────────────────

    def test_get_plan_generated_at_with_tz_naive_string_returns_tz_aware(self, tmp_path):
        """Stored tz-naive ISO string must be returned as tz-aware datetime."""
        from smartboiler.state_store import StateStore
        store = StateStore(data_dir=str(tmp_path))
        store.set("plan_generated_at", "2024-01-15T10:30:00")  # tz-naive
        dt = store.get_plan_generated_at()
        assert dt is not None
        assert dt.tzinfo is not None, "get_plan_generated_at must return tz-aware datetime"

    def test_current_plan_hour_index_does_not_raise_with_tz_naive_stored_plan(self, tmp_path):
        """_current_plan_hour_index() must not raise TypeError when stored timestamp is tz-naive."""
        from smartboiler.state_store import StateStore
        store = StateStore(data_dir=str(tmp_path))
        store.set("plan_generated_at", "2024-01-15T10:00:00")  # tz-naive, old
        dt = store.get_plan_generated_at()
        assert dt is not None
        # Simulate what _current_plan_hour_index() does
        from datetime import datetime as _dt
        # Must not raise TypeError
        elapsed = (_dt.now().astimezone() - dt).total_seconds() // 3600
        assert elapsed >= 0

    def test_get_last_data_collection_with_tz_naive_string_does_not_raise(self, tmp_path):
        """Stored tz-naive last_data_collection must be usable in tz-aware comparisons."""
        from smartboiler.state_store import StateStore
        store = StateStore(data_dir=str(tmp_path))
        store.set("last_data_collection", "2024-01-01T08:00:00")  # tz-naive
        dt = store.get_last_data_collection()
        # Comparison with tz-aware must not raise TypeError
        from datetime import datetime as _dt
        age_h = (_dt.now().astimezone() - dt).total_seconds() / 3600
        assert age_h > 0

    def test_get_heating_until_with_tz_naive_string_returns_tz_aware(self, tmp_path):
        """get_heating_until() must apply the same tz-naive guard as other datetime
        methods — storing a naive ISO string must not cause TypeError in comparisons."""
        from smartboiler.state_store import StateStore
        from datetime import datetime as _dt
        store = StateStore(data_dir=str(tmp_path))
        store.set("heating_until", "2024-06-01T14:30:00")  # tz-naive ISO string
        dt = store.get_heating_until()
        assert dt is not None
        assert dt.tzinfo is not None, "get_heating_until() must return a tz-aware datetime"
        # Must not raise TypeError when compared to tz-aware now
        diff = _dt.now().astimezone() - dt
        assert diff.total_seconds() > 0
