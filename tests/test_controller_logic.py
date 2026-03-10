"""Unit tests for Controller decision logic (no network / DB required)."""
import pytest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch, call

import pandas as pd

from smartboiler.controller import Controller


# ---------------------------------------------------------------------------
# Fixture: a Controller with all heavy dependencies mocked out
# ---------------------------------------------------------------------------

@pytest.fixture
def ctrl():
    """Return a Controller instance with __init__ bypassed and state set manually."""
    with patch.object(Controller, "__init__", return_value=None):
        c = Controller()

    c.dataHandler = MagicMock()
    c.boiler = MagicMock()
    c.forecast = MagicMock()
    c.eventChecker = MagicMock()

    c.tmp_min = 5
    c.learning = False
    c.start_date = datetime.now()
    c.heating_until = None
    c.HEATING_TIMEOUT_MINUTES = 180
    c.last_model_training = datetime.now()
    c.last_legionella_heating = datetime.now()
    c.actual_forecast = pd.DataFrame({"prediction": [0.1] * 6})

    # Default sensor reading: 50 °C, relay off
    c.dataHandler.get_actual_boiler_stats.return_value = {
        "boiler_case_tmp": 50,
        "is_boiler_on": False,
    }
    c.boiler.real_tmp.return_value = 50

    return c


# ---------------------------------------------------------------------------
# _learning
# ---------------------------------------------------------------------------

class TestLearning:
    def test_returns_false_when_learning_disabled(self, ctrl):
        ctrl.learning = False
        assert ctrl._learning() is False

    def test_returns_true_within_28_days(self, ctrl):
        ctrl.learning = True
        ctrl.start_date = datetime.now() - timedelta(days=10)
        assert ctrl._learning() is True

    def test_returns_false_after_28_days(self, ctrl):
        ctrl.learning = True
        ctrl.start_date = datetime.now() - timedelta(days=29)
        assert ctrl._learning() is False

    def test_boundary_at_exactly_28_days(self, ctrl):
        ctrl.learning = True
        ctrl.start_date = datetime.now() - timedelta(days=28, seconds=1)
        assert ctrl._learning() is False


# ---------------------------------------------------------------------------
# _check_data  (model retraining)
# ---------------------------------------------------------------------------

class TestCheckData:
    def test_no_retrain_when_recent(self, ctrl):
        ctrl.last_model_training = datetime.now() - timedelta(days=3)
        ctrl._check_data()
        ctrl.forecast.train_model.assert_not_called()

    def test_retrain_when_stale(self, ctrl):
        ctrl.last_model_training = datetime.now() - timedelta(days=8)
        ctrl._check_data()
        ctrl.forecast.train_model.assert_called_once()

    def test_last_model_training_updated_after_retrain(self, ctrl):
        ctrl.last_model_training = datetime.now() - timedelta(days=8)
        before = datetime.now()
        ctrl._check_data()
        assert ctrl.last_model_training >= before

    def test_no_update_to_last_training_when_not_stale(self, ctrl):
        ts = datetime.now() - timedelta(days=3)
        ctrl.last_model_training = ts
        ctrl._check_data()
        assert ctrl.last_model_training == ts


# ---------------------------------------------------------------------------
# control() – off-event branch
# ---------------------------------------------------------------------------

class TestControlOffEvent:
    def test_turns_off_when_off_event_active(self, ctrl):
        ctrl.eventChecker.check_off_event.return_value = True
        ctrl.control()
        ctrl.boiler.turn_off.assert_called_once()

    def test_does_not_turn_on_when_off_event_active(self, ctrl):
        ctrl.eventChecker.check_off_event.return_value = True
        ctrl.control()
        ctrl.boiler.turn_on.assert_not_called()

    def test_returns_early_on_off_event(self, ctrl):
        """is_needed_to_heat must NOT be consulted when off-event is active."""
        ctrl.eventChecker.check_off_event.return_value = True
        ctrl.control()
        ctrl.boiler.is_needed_to_heat.assert_not_called()


# ---------------------------------------------------------------------------
# control() – freeze-protection branch
# ---------------------------------------------------------------------------

class TestControlFreezeProtection:
    def test_turns_on_when_below_5_degrees(self, ctrl):
        ctrl.eventChecker.check_off_event.return_value = False
        ctrl.dataHandler.get_actual_boiler_stats.return_value = {
            "boiler_case_tmp": 3,
            "is_boiler_on": False,
        }
        ctrl.boiler.real_tmp.return_value = 3
        ctrl.boiler.is_needed_to_heat.return_value = (False, 0)
        ctrl.control()
        ctrl.boiler.turn_on.assert_called()


# ---------------------------------------------------------------------------
# control() – active heating-window branch
# ---------------------------------------------------------------------------

class TestControlHeatingWindow:
    def test_keeps_on_within_active_window(self, ctrl):
        ctrl.eventChecker.check_off_event.return_value = False
        ctrl.heating_until = datetime.now() + timedelta(minutes=30)
        ctrl.control()
        ctrl.boiler.turn_on.assert_called()

    def test_does_not_consult_is_needed_within_window(self, ctrl):
        ctrl.eventChecker.check_off_event.return_value = False
        ctrl.heating_until = datetime.now() + timedelta(minutes=30)
        ctrl.control()
        ctrl.boiler.is_needed_to_heat.assert_not_called()

    def test_expired_window_falls_through_to_is_needed(self, ctrl):
        ctrl.eventChecker.check_off_event.return_value = False
        ctrl.heating_until = datetime.now() - timedelta(minutes=1)
        ctrl.boiler.is_needed_to_heat.return_value = (False, 0)
        ctrl.control()
        ctrl.boiler.is_needed_to_heat.assert_called_once()


# ---------------------------------------------------------------------------
# control() – is_needed_to_heat outcomes
# ---------------------------------------------------------------------------

class TestControlIsNeededToHeat:
    def test_turns_on_and_sets_window_when_heating_needed(self, ctrl):
        ctrl.eventChecker.check_off_event.return_value = False
        ctrl.boiler.is_needed_to_heat.return_value = (True, 90)
        ctrl.control()
        ctrl.boiler.turn_on.assert_called()
        assert ctrl.heating_until is not None
        assert ctrl.heating_until > datetime.now()

    def test_turns_off_and_clears_window_when_no_heat_needed(self, ctrl):
        ctrl.eventChecker.check_off_event.return_value = False
        ctrl.boiler.is_needed_to_heat.return_value = (False, 0)
        ctrl.control()
        ctrl.boiler.turn_off.assert_called()
        assert ctrl.heating_until is None

    def test_heating_window_capped_at_timeout(self, ctrl):
        ctrl.eventChecker.check_off_event.return_value = False
        ctrl.boiler.is_needed_to_heat.return_value = (True, 999)
        ctrl.control()
        expected_max = datetime.now() + timedelta(minutes=ctrl.HEATING_TIMEOUT_MINUTES)
        # Allow 5-second tolerance for execution time
        assert ctrl.heating_until <= expected_max + timedelta(seconds=5)
