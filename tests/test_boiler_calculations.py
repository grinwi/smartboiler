"""Unit tests for Boiler pure-calculation methods (no network / DB required)."""
import pytest
from datetime import time
from unittest.mock import MagicMock

import pandas as pd

from smartboiler.boiler import Boiler


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_schedule_df() -> pd.DataFrame:
    """Minimal high_tarif_schedule DataFrame (all unavailable_minutes = 0)."""
    rows = [
        {"time": time(h, 0), "weekday": d, "unavailable_minutes": 0}
        for d in range(7)
        for h in range(24)
    ]
    return pd.DataFrame(rows)


@pytest.fixture
def boiler() -> Boiler:
    data_handler = MagicMock()
    data_handler.get_high_tarif_schedule.return_value = _make_schedule_df()
    event_checker = MagicMock()

    return Boiler(
        shelly_ip="192.168.1.1",
        dataHandler=data_handler,
        eventChecker=event_checker,
        capacity=100,
        wattage=2000,
        set_tmp=60,
        min_tmp=40,
        heater_efficiency=0.88,
        average_boiler_surroundings_temp=15,
        boiler_case_max_tmp=40,
        cooldown_coef_B=1.12,
        hdo=False,
    )


# ---------------------------------------------------------------------------
# get_kWh_loss_in_time
# ---------------------------------------------------------------------------

class TestGetKWhLossInTime:
    def test_zero_time_gives_zero_loss(self, boiler):
        assert boiler.get_kWh_loss_in_time(0, tmp_act=60) == 0.0

    def test_positive_loss_when_above_ambient(self, boiler):
        assert boiler.get_kWh_loss_in_time(60, tmp_act=60) > 0

    def test_higher_temperature_more_loss(self, boiler):
        low = boiler.get_kWh_loss_in_time(60, tmp_act=50)
        high = boiler.get_kWh_loss_in_time(60, tmp_act=70)
        assert high > low

    def test_known_value(self, boiler):
        # (minutes * coef_B * delta_T / 1000) / 60
        # = (60 * 1.12 * (60 - 15) / 1000) / 60 = 1.12 * 45 / 1000
        expected = (60 * 1.12 * (60 - 15) / 1000) / 60
        assert abs(boiler.get_kWh_loss_in_time(60, tmp_act=60) - expected) < 1e-9

    def test_doubles_with_double_time(self, boiler):
        t1 = boiler.get_kWh_loss_in_time(30, tmp_act=60)
        t2 = boiler.get_kWh_loss_in_time(60, tmp_act=60)
        assert abs(t2 - 2 * t1) < 1e-9


# ---------------------------------------------------------------------------
# time_needed_to_heat_up_minutes
# ---------------------------------------------------------------------------

class TestTimeNeededToHeatUpMinutes:
    def test_zero_consumption_zero_time(self, boiler):
        assert boiler.time_needed_to_heat_up_minutes(0) == 0.0

    def test_known_value(self, boiler):
        # time = (kWh / real_wattage_kW) * 60
        # real_wattage = 2000 * 0.88 = 1760 W = 1.76 kW
        expected = (1.0 / 1.76) * 60
        assert abs(boiler.time_needed_to_heat_up_minutes(1.0) - expected) < 0.01

    def test_linear_in_consumption(self, boiler):
        t1 = boiler.time_needed_to_heat_up_minutes(1.0)
        t2 = boiler.time_needed_to_heat_up_minutes(2.0)
        assert abs(t2 - 2 * t1) < 1e-9

    def test_positive_for_positive_consumption(self, boiler):
        assert boiler.time_needed_to_heat_up_minutes(0.5) > 0


# ---------------------------------------------------------------------------
# get_kWh_delta_from_temperatures
# ---------------------------------------------------------------------------

class TestGetKWhDeltaFromTemperatures:
    def test_same_temperature_zero_delta(self, boiler):
        assert boiler.get_kWh_delta_from_temperatures(50, 50) == 0.0

    def test_positive_when_heating(self, boiler):
        assert boiler.get_kWh_delta_from_temperatures(20, 60) > 0

    def test_negative_when_cooling(self, boiler):
        assert boiler.get_kWh_delta_from_temperatures(60, 20) < 0

    def test_known_value(self, boiler):
        # 4.186 * capacity * delta_T / 3600
        expected = 4.186 * 100 * 40 / 3600
        assert abs(boiler.get_kWh_delta_from_temperatures(20, 60) - expected) < 1e-9

    def test_symmetric_magnitude(self, boiler):
        up = boiler.get_kWh_delta_from_temperatures(20, 60)
        down = boiler.get_kWh_delta_from_temperatures(60, 20)
        assert abs(up + down) < 1e-9


# ---------------------------------------------------------------------------
# showers_degrees
# ---------------------------------------------------------------------------

class TestShowersDegrees:
    def test_zero_showers_equals_min_tmp(self, boiler):
        # (min_tmp*capacity + 0) / capacity = min_tmp
        assert boiler.showers_degrees(0) == pytest.approx(40.0)

    def test_one_shower_known_value(self, boiler):
        # (40*100 + 40*40 - 40*10) / 100 = (4000 + 1600 - 400) / 100 = 52.0
        assert boiler.showers_degrees(1) == pytest.approx(52.0)

    def test_capped_at_set_tmp(self, boiler):
        assert boiler.showers_degrees(100) == boiler.set_tmp

    def test_monotonically_increasing(self, boiler):
        results = [boiler.showers_degrees(n) for n in range(5)]
        for i in range(len(results) - 1):
            assert results[i] <= results[i + 1]


# ---------------------------------------------------------------------------
# real_tmp
# ---------------------------------------------------------------------------

class TestRealTmp:
    def test_none_input_returns_50(self, boiler):
        assert boiler.real_tmp(None) == 50

    def test_below_area_tmp_returns_raw(self, boiler):
        # area_tmp = 15; anything below is returned as-is
        assert boiler.real_tmp(10) == 10

    def test_above_set_tmp_returns_raw(self, boiler):
        # set_tmp = 60; anything above is returned as-is
        assert boiler.real_tmp(65) == 65

    def test_at_boiler_case_max_maps_to_set_tmp(self, boiler):
        # boiler_case_max_tmp=40, area_tmp=15, set_tmp=60
        # p1 = (40-15)/(40-15) = 1.0  =>  tmp = 1.0*(60-15)+15 = 60
        assert boiler.real_tmp(40) == pytest.approx(60.0)

    def test_at_area_tmp_maps_to_area_tmp(self, boiler):
        # p1 = (15-15)/(40-15) = 0  =>  tmp = 0*(60-15)+15 = 15
        assert boiler.real_tmp(15) == pytest.approx(15.0)

    def test_midpoint(self, boiler):
        # tmp_act = 27.5  =>  p1 = (27.5-15)/(40-15) = 0.5
        # tmp = 0.5*(60-15)+15 = 37.5
        assert boiler.real_tmp(27.5) == pytest.approx(37.5)


# ---------------------------------------------------------------------------
# is_needed_to_heat – below min_tmp branch
# ---------------------------------------------------------------------------

class TestIsNeededToHeatBelowMinTmp:
    def test_heats_when_below_min_tmp(self, boiler):
        boiler.eventChecker.next_calendar_heat_up_event.return_value = {
            "minutes_to_event": None
        }
        prediction = pd.DataFrame({"consumption": [0.0] * 6})
        needed, minutes = boiler.is_needed_to_heat(tmp_act=35, prediction_of_consumption=prediction)
        assert needed is True
        assert minutes > 0

    def test_minutes_positive_below_min_tmp(self, boiler):
        boiler.eventChecker.next_calendar_heat_up_event.return_value = {
            "minutes_to_event": None
        }
        prediction = pd.DataFrame({"consumption": [0.0] * 6})
        _, minutes = boiler.is_needed_to_heat(tmp_act=30, prediction_of_consumption=prediction)
        assert minutes > 0
