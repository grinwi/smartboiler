"""Tests for the Open-Meteo PV forecast and how the controller uses it.

No network: requests.get is patched with a recorded-shape response.
"""
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from smartboiler.controller import SmartBoilerController
from smartboiler.pv_forecast import (
    OpenMeteoPVForecast,
    build_from_config,
    compass_to_open_meteo_azimuth,
)
from smartboiler.setup_config import DEFAULTS, validate_config

T0 = int(datetime(2026, 6, 1, 0, 0, tzinfo=timezone.utc).timestamp())


def _response(gti_by_hour):
    """Open-Meteo JSON: value at time T is the mean over the hour before T."""
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json.return_value = {
        "hourly": {
            "time": [T0 + 3600 * (h + 1) for h in range(len(gti_by_hour))],
            "global_tilted_irradiance": gti_by_hour,
        }
    }
    return resp


def _forecaster(**kw):
    args = dict(latitude=49.2, longitude=16.6, kwp=5.0, performance_ratio=0.8)
    args.update(kw)
    return OpenMeteoPVForecast(**args)


# ── Pure helpers ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize("compass,expected", [(180, 0), (90, -90), (270, 90), (0, 180), (135, -45)])
def test_compass_to_open_meteo_azimuth(compass, expected):
    assert compass_to_open_meteo_azimuth(compass) == pytest.approx(expected)


# ── OpenMeteoPVForecast ──────────────────────────────────────────────────────

class TestOpenMeteoPVForecast:
    def test_irradiance_to_kwh_and_hour_alignment(self):
        gti = [0.0] * 24
        gti[12] = 1000.0   # mean over 12:00–13:00
        with patch("smartboiler.pv_forecast.requests.get", return_value=_response(gti)):
            kwh = _forecaster().hourly_kwh()
        # 1000 W/m² × 5 kWp × 0.8 = 4 kWh in the hour starting 12:00
        assert kwh[T0 + 12 * 3600] == pytest.approx(4.0)
        assert kwh[T0 + 11 * 3600] == 0.0

    def test_request_parameters(self):
        with patch("smartboiler.pv_forecast.requests.get", return_value=_response([0.0] * 24)) as get:
            _forecaster(tilt_deg=30, azimuth_compass_deg=135).hourly_kwh()
        params = get.call_args.kwargs["params"]
        assert params["hourly"] == "global_tilted_irradiance"
        assert params["tilt"] == 30
        assert params["azimuth"] == pytest.approx(-45)
        assert params["timeformat"] == "unixtime"

    def test_surplus_subtracts_base_load_and_clamps(self):
        gti = [0.0] * 48
        gti[10] = 500.0    # 2.0 kWh
        gti[11] = 50.0     # 0.2 kWh
        with patch("smartboiler.pv_forecast.requests.get", return_value=_response(gti)):
            out = _forecaster().surplus_next_24h(
                datetime.fromtimestamp(T0, tz=timezone.utc), base_load_kw=0.3
            )
        assert len(out) == 24
        assert out[10] == pytest.approx(1.7)
        assert out[11] == 0.0   # 0.2 kWh production < 0.3 kW load
        assert out[3] == 0.0

    def test_surplus_starts_at_current_hour(self):
        gti = [0.0] * 48
        gti[14] = 1000.0
        start = datetime.fromtimestamp(T0 + 12 * 3600 + 25 * 60, tz=timezone.utc)  # 12:25
        with patch("smartboiler.pv_forecast.requests.get", return_value=_response(gti)):
            out = _forecaster().surplus_next_24h(start)
        assert out[2] == pytest.approx(4.0)

    def test_result_is_cached(self):
        f = _forecaster()
        with patch("smartboiler.pv_forecast.requests.get", return_value=_response([0.0] * 24)) as get:
            f.hourly_kwh()
            f.hourly_kwh()
        assert get.call_count == 1

    def test_null_values_are_skipped(self):
        with patch("smartboiler.pv_forecast.requests.get", return_value=_response([None, 800.0])):
            kwh = _forecaster().hourly_kwh()
        assert list(kwh) == [T0 + 3600]

    def test_empty_response_raises(self):
        with patch("smartboiler.pv_forecast.requests.get", return_value=_response([None])):
            with pytest.raises(ValueError):
                _forecaster().hourly_kwh()


# ── build_from_config ────────────────────────────────────────────────────────

class TestBuildFromConfig:
    LOC = {"latitude": 49.2, "longitude": 16.6}

    def _cfg(self, **kw):
        cfg = dict(DEFAULTS, has_pv=True, pv_installed_power_kw=5.0)
        cfg.update(kw)
        return cfg

    def test_builds_with_kwp_and_location(self):
        f = build_from_config(self._cfg(pv_tilt_deg=20, pv_azimuth_deg=90), self.LOC)
        assert f.kwp == 5.0 and f.tilt_deg == 20 and f.azimuth == -90

    @pytest.mark.parametrize("cfg_kw,loc", [
        ({"has_pv": False}, LOC),
        ({"pv_installed_power_kw": 0.0}, LOC),
        ({"pv_forecast_source": "entity"}, LOC),
        ({}, None),
        ({}, {"latitude": 49.2}),
    ])
    def test_none_when_not_usable(self, cfg_kw, loc):
        assert build_from_config(self._cfg(**cfg_kw), loc) is None


def test_config_validation():
    base = dict(DEFAULTS, boiler_switch_entity_id="switch.b", has_pv=True, pv_installed_power_kw=5)
    assert validate_config(base) == []
    assert any("pv_forecast_source" in e for e in validate_config(dict(base, pv_forecast_source="x")))
    assert any("pv_tilt_deg" in e for e in validate_config(dict(base, pv_tilt_deg=120)))


# ── Controller source priority ───────────────────────────────────────────────

@pytest.fixture
def ctrl():
    with patch.object(SmartBoilerController, "__init__", return_value=None):
        c = SmartBoilerController.__new__(SmartBoilerController)
    c.ha = MagicMock()
    c.ha.get_config.return_value = {"latitude": 49.2, "longitude": 16.6}
    c.ha.get_state_value.return_value = 1500.0   # current PV power, W
    c.has_pv = True
    c.pv_installed_power_kw = 5.0
    c.pv_forecast_entity_id = None
    c.pv_power_entity_id = "sensor.pv_power"
    c.pv_surplus_entity_id = None
    c.house_base_load_kw = 0.3
    c._pv_forecaster = None
    c._options = dict(DEFAULTS, has_pv=True, pv_installed_power_kw=5.0)
    return c


class TestControllerPVSource:
    def test_open_meteo_preferred(self, ctrl):
        curve = [0.0] * 24
        curve[5] = 2.0
        with patch("smartboiler.pv_forecast.OpenMeteoPVForecast.surplus_next_24h", return_value=curve) as s:
            assert ctrl._get_pv_forecast_24h() == curve
        assert s.call_args.kwargs["base_load_kw"] == pytest.approx(0.3)

    def test_falls_back_to_power_entity_on_error(self, ctrl):
        with patch("smartboiler.pv_forecast.OpenMeteoPVForecast.surplus_next_24h",
                   side_effect=ConnectionError("offline")):
            assert ctrl._get_pv_forecast_24h() == [pytest.approx(1.5)] * 24

    def test_entity_source_skips_open_meteo(self, ctrl):
        ctrl._options["pv_forecast_source"] = "entity"
        with patch("smartboiler.pv_forecast.requests.get") as get:
            assert ctrl._get_pv_forecast_24h() == [pytest.approx(1.5)] * 24
        get.assert_not_called()

    def test_retries_location_until_ha_answers(self, ctrl):
        ctrl.ha.get_config.return_value = None
        ctrl._get_pv_forecast_24h()
        assert ctrl._pv_forecaster is None
        ctrl.ha.get_config.return_value = {"latitude": 49.2, "longitude": 16.6}
        with patch("smartboiler.pv_forecast.OpenMeteoPVForecast.surplus_next_24h", return_value=[0.0] * 24):
            ctrl._get_pv_forecast_24h()
        assert ctrl._pv_forecaster is not None

    def test_pv_disabled_returns_zeros(self, ctrl):
        ctrl.has_pv = False
        assert ctrl._get_pv_forecast_24h() == [0.0] * 24
