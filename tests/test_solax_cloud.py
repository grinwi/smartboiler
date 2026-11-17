"""Tests for the SolaX Cloud client, its config, test route and controller use."""
from unittest.mock import MagicMock, patch

import pytest

from smartboiler import solax_cloud as sx
from smartboiler.setup_config import DEFAULTS, validate_config

TOKEN_OK = {"code": 0, "result": {"access_token": "tok", "expires_in": 2591999}}
DATA_OK = {"code": 10000, "result": [{"batterySOC": 64, "MPPTTotalInputPower": 3120.0,
                                      "gridPower": -850.0, "dailyYield": 12.3, "inverterSN": None}]}


def _resp(body, status=200):
    r = MagicMock(status_code=status)
    r.json.return_value = body
    r.raise_for_status = MagicMock()
    return r


def _client():
    return sx.SolaxCloud("id", "secret", "SN123")


@pytest.fixture
def http():
    with patch("smartboiler.solax_cloud.requests.post", return_value=_resp(TOKEN_OK)) as post, \
         patch("smartboiler.solax_cloud.requests.get", return_value=_resp(DATA_OK)) as get:
        yield post, get


def test_reads_values_with_bearer_token(http):
    post, get = http
    c = _client()
    assert c.battery_soc() == 64 and c.pv_power_w() == 3120.0 and c.grid_power_w() == -850.0
    assert post.call_args.kwargs["data"]["grant_type"] == "client_credentials"
    kw = get.call_args.kwargs
    assert kw["headers"]["Authorization"] == "bearer tok"
    assert kw["params"] == {"snList": "SN123", "deviceType": 1, "businessType": 1}
    assert get.call_args.args[0] == "https://openapi-eu.solaxcloud.com/openapi/v2/device/realtime_data"


def test_token_and_data_are_cached(http):
    post, get = http
    c = _client()
    c.battery_soc(); c.pv_power_w(); c.grid_power_w()
    assert post.call_count == 1 and get.call_count == 1


def test_bad_credentials(http):
    post, _ = http
    post.return_value = _resp({"code": 10400, "message": "bad client"})
    with pytest.raises(sx.SolaxError, match="login failed"):
        _client().realtime()


def test_data_error_drops_token(http):
    post, get = http
    get.return_value = _resp({"code": 10402, "message": "token invalid"})
    c = _client()
    with pytest.raises(sx.SolaxError):
        c.realtime()
    assert c._token is None


def test_values_are_none_on_errors(http):
    _, get = http
    get.side_effect = sx.requests.ConnectionError("down")
    assert _client().battery_soc() is None


def test_empty_result(http):
    _, get = http
    get.return_value = _resp({"code": 10000, "result": []})
    with pytest.raises(sx.SolaxError, match="no data"):
        _client().realtime()


def test_from_config_requires_all_three():
    assert sx.from_config(dict(DEFAULTS)) is None
    assert sx.from_config(dict(DEFAULTS, solax_client_id="a", solax_client_secret="b")) is None
    c = sx.from_config(dict(DEFAULTS, solax_client_id="a", solax_client_secret="b", solax_inverter_sn="c"))
    assert c.inverter_sn == "c" and c.base_url.endswith("eu.solaxcloud.com")


def test_unknown_region_rejected():
    with pytest.raises(ValueError):
        sx.SolaxCloud("a", "b", "c", region="mars")


def test_validation_needs_all_or_nothing():
    base = dict(DEFAULTS, boiler_switch_entity_id="switch.b")
    assert validate_config(base) == []
    assert any("SolaX" in e for e in validate_config(dict(base, solax_client_id="x")))
    assert any("solax_region" in e for e in validate_config(dict(base, solax_region="us")))


# ── Test route ───────────────────────────────────────────────────────────────

@pytest.fixture
def client():
    from smartboiler.web_server import app
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


def test_route_reports_values_and_fields(client, http):
    r = client.post("/api/test/solax", json={"solax_client_id": "a", "solax_client_secret": "b",
                                             "solax_inverter_sn": "c", "solax_region": "eu"}).get_json()
    assert r["ok"] and r["values"] == {"battery_soc": 64, "pv_power_w": 3120.0, "grid_power_w": -850.0}
    assert "inverterSN" not in r["fields"] and "dailyYield" in r["fields"]


def test_route_requires_credentials(client):
    assert client.post("/api/test/solax", json={}).status_code == 400


def test_route_reports_login_error(client, http):
    post, _ = http
    post.return_value = _resp({"code": 10400, "message": "bad client"})
    r = client.post("/api/test/solax", json={"solax_client_id": "a", "solax_client_secret": "b",
                                             "solax_inverter_sn": "c"}).get_json()
    assert r["ok"] is False and "bad client" in r["error"]


# ── Controller uses SolaX when no HA sensor is set ───────────────────────────

@pytest.fixture
def ctrl():
    from smartboiler.controller import SmartBoilerController
    with patch.object(SmartBoilerController, "__init__", return_value=None):
        c = SmartBoilerController.__new__(SmartBoilerController)
    c.ha = MagicMock()
    c.solax = MagicMock()
    c.solax.battery_soc.return_value = 50.0
    c.solax.grid_power_w.return_value = 1200.0
    c.solax.pv_power_w.return_value = 2500.0
    c.battery_soc_entity_id = None
    c.battery_soc_unit = "kwh"
    c.battery_capacity_kwh = 10.0
    c.pv_surplus_entity_id = None
    c.pv_surplus_inverted = False
    return c


def test_battery_soc_from_solax_is_percent(ctrl):
    assert ctrl._get_battery_soc_kwh() == pytest.approx(5.0)


def test_export_from_solax_with_sign_flip(ctrl):
    assert ctrl._read_export_w() == 1200.0
    ctrl.pv_surplus_inverted = True
    assert ctrl._read_export_w() == -1200.0


def test_ha_sensor_wins_over_solax(ctrl):
    ctrl.pv_surplus_entity_id = "sensor.grid"
    ctrl.ha.get_state_value.return_value = 300.0
    assert ctrl._read_export_w() == 300.0
    ctrl.solax.grid_power_w.assert_not_called()
