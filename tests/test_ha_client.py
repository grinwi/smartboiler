"""
Tests for HAClient — Home Assistant REST API wrapper.

Uses unittest.mock to intercept requests.Session calls so no real HA instance
is needed. Covers:
- get_state() success and error paths
- get_state_value() numeric, string, unavailable, unknown states
- get_all_states() filtering by domain
- get_history() list-of-lists unwrapping
- call_service(), turn_on(), turn_off()
- is_entity_on() for all state string variants
- get_attribute()
- HTTP error propagation (raise_for_status)
"""
from datetime import datetime
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from smartboiler.ha_client import HAClient


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_client(base_url="http://ha-test/api", token="testtoken"):
    return HAClient(base_url=base_url, token=token, timeout=5)


def _mock_response(json_data, status_code=200):
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_data
    resp.raise_for_status = MagicMock()
    if status_code >= 400:
        from requests import HTTPError
        resp.raise_for_status.side_effect = HTTPError(f"HTTP {status_code}")
    return resp


def _state_obj(entity_id="sensor.test", state="42.5", attributes=None):
    return {
        "entity_id": entity_id,
        "state": state,
        "attributes": attributes or {},
        "last_changed": "2024-03-11T10:00:00+00:00",
        "last_updated": "2024-03-11T10:00:00+00:00",
    }


# ---------------------------------------------------------------------------
# get_state()
# ---------------------------------------------------------------------------

class TestGetState:
    def test_returns_state_dict_on_success(self):
        client = _make_client()
        obj = _state_obj()
        with patch.object(client._session, "get", return_value=_mock_response(obj)):
            result = client.get_state("sensor.test")
        assert result["entity_id"] == "sensor.test"
        assert result["state"] == "42.5"

    def test_returns_none_on_http_error(self):
        client = _make_client()
        with patch.object(client._session, "get", return_value=_mock_response({}, 404)):
            result = client.get_state("sensor.missing")
        assert result is None

    def test_returns_none_on_connection_error(self):
        client = _make_client()
        with patch.object(client._session, "get", side_effect=ConnectionError("refused")):
            result = client.get_state("sensor.test")
        assert result is None

    def test_correct_url_constructed(self):
        client = _make_client(base_url="http://supervisor/core/api")
        with patch.object(client._session, "get", return_value=_mock_response(_state_obj())) as mock_get:
            client.get_state("switch.boiler")
        mock_get.assert_called_once()
        url = mock_get.call_args[0][0]
        assert url == "http://supervisor/core/api/states/switch.boiler"

    def test_auth_header_present(self):
        client = _make_client(token="mysecrettoken")
        assert client._session.headers.get("Authorization") == "Bearer mysecrettoken"


# ---------------------------------------------------------------------------
# get_state_value()
# ---------------------------------------------------------------------------

class TestGetStateValue:
    def test_returns_float_for_numeric_state(self):
        client = _make_client()
        with patch.object(client, "get_state", return_value=_state_obj(state="55.3")):
            assert client.get_state_value("sensor.temp") == pytest.approx(55.3)

    def test_returns_raw_string_for_non_numeric_state(self):
        client = _make_client()
        with patch.object(client, "get_state", return_value=_state_obj(state="home")):
            result = client.get_state_value("person.adam")
        assert result == "home"

    def test_returns_default_for_unavailable(self):
        client = _make_client()
        with patch.object(client, "get_state", return_value=_state_obj(state="unavailable")):
            assert client.get_state_value("sensor.x", default=99.0) == pytest.approx(99.0)

    def test_returns_default_for_unknown(self):
        client = _make_client()
        with patch.object(client, "get_state", return_value=_state_obj(state="unknown")):
            assert client.get_state_value("sensor.x", default=0.0) == pytest.approx(0.0)

    def test_returns_default_for_empty_string(self):
        client = _make_client()
        with patch.object(client, "get_state", return_value=_state_obj(state="")):
            assert client.get_state_value("sensor.x", default=-1.0) == pytest.approx(-1.0)

    def test_returns_default_when_entity_missing(self):
        client = _make_client()
        with patch.object(client, "get_state", return_value=None):
            assert client.get_state_value("sensor.missing", default=42.0) == pytest.approx(42.0)

    def test_default_is_none_when_not_specified(self):
        client = _make_client()
        with patch.object(client, "get_state", return_value=None):
            assert client.get_state_value("sensor.missing") is None

    def test_integer_string_parsed_as_float(self):
        client = _make_client()
        with patch.object(client, "get_state", return_value=_state_obj(state="2000")):
            assert client.get_state_value("sensor.power") == pytest.approx(2000.0)


# ---------------------------------------------------------------------------
# get_all_states()
# ---------------------------------------------------------------------------

class TestGetAllStates:
    def _states(self):
        return [
            _state_obj("sensor.temp"),
            _state_obj("switch.boiler"),
            _state_obj("sensor.power"),
        ]

    def test_returns_all_states_when_no_domain_filter(self):
        client = _make_client()
        with patch.object(client._session, "get", return_value=_mock_response(self._states())):
            result = client.get_all_states()
        assert len(result) == 3

    def test_filters_by_domain(self):
        client = _make_client()
        with patch.object(client._session, "get", return_value=_mock_response(self._states())):
            result = client.get_all_states(domain="sensor")
        assert len(result) == 2
        assert all(s["entity_id"].startswith("sensor.") for s in result)

    def test_returns_empty_list_on_error(self):
        client = _make_client()
        with patch.object(client._session, "get", side_effect=Exception("network error")):
            result = client.get_all_states()
        assert result == []

    def test_domain_filter_is_case_sensitive(self):
        client = _make_client()
        with patch.object(client._session, "get", return_value=_mock_response(self._states())):
            result = client.get_all_states(domain="SENSOR")
        assert result == []  # no match for uppercase domain


# ---------------------------------------------------------------------------
# get_history()
# ---------------------------------------------------------------------------

class TestGetHistory:
    def _history_response(self, states):
        """HA wraps history in list-of-lists."""
        return [states]

    def test_unwraps_list_of_lists(self):
        client = _make_client()
        states = [_state_obj("sensor.power", "1800"), _state_obj("sensor.power", "2000")]
        with patch.object(client._session, "get",
                          return_value=_mock_response(self._history_response(states))):
            result = client.get_history("sensor.power", datetime(2024, 3, 11, 0, 0))
        assert len(result) == 2
        assert result[0]["state"] == "1800"

    def test_returns_empty_list_for_no_history(self):
        client = _make_client()
        with patch.object(client._session, "get", return_value=_mock_response([[]])):
            result = client.get_history("sensor.new", datetime(2024, 3, 11))
        assert result == []

    def test_returns_empty_list_on_error(self):
        client = _make_client()
        with patch.object(client._session, "get", side_effect=Exception("timeout")):
            result = client.get_history("sensor.x", datetime(2024, 3, 11))
        assert result == []

    def test_history_url_includes_start_datetime(self):
        client = _make_client(base_url="http://ha/api")
        start = datetime(2024, 3, 11, 6, 0, 0)
        states = [_state_obj("sensor.x", "1")]
        with patch.object(client._session, "get",
                          return_value=_mock_response([states])) as mock_get:
            client.get_history("sensor.x", start)
        url = mock_get.call_args[0][0]
        assert "history/period" in url
        assert "2024-03-11" in url

    def test_history_includes_end_time_param(self):
        client = _make_client()
        start = datetime(2024, 3, 11, 6, 0)
        end = datetime(2024, 3, 11, 12, 0)
        with patch.object(client._session, "get",
                          return_value=_mock_response([[_state_obj()]])) as mock_get:
            client.get_history("sensor.x", start, end)
        params = mock_get.call_args[1]["params"]
        assert "end_time" in params

    def test_empty_outer_list_returns_empty(self):
        client = _make_client()
        with patch.object(client._session, "get", return_value=_mock_response([])):
            result = client.get_history("sensor.x", datetime(2024, 3, 11))
        assert result == []


# ---------------------------------------------------------------------------
# call_service() / turn_on() / turn_off()
# ---------------------------------------------------------------------------

class TestCallService:
    def test_returns_true_on_success(self):
        client = _make_client()
        with patch.object(client._session, "post", return_value=_mock_response({})):
            assert client.call_service("switch", "turn_on", "switch.boiler") is True

    def test_returns_false_on_error(self):
        client = _make_client()
        with patch.object(client._session, "post", side_effect=Exception("refused")):
            assert client.call_service("switch", "turn_on", "switch.boiler") is False

    def test_post_url_correct(self):
        client = _make_client(base_url="http://ha/api")
        with patch.object(client._session, "post",
                          return_value=_mock_response({})) as mock_post:
            client.call_service("switch", "turn_on", "switch.boiler")
        url = mock_post.call_args[0][0]
        assert url == "http://ha/api/services/switch/turn_on"

    def test_post_payload_includes_entity_id(self):
        client = _make_client()
        with patch.object(client._session, "post",
                          return_value=_mock_response({})) as mock_post:
            client.call_service("switch", "turn_on", "switch.boiler", extra_param="foo")
        payload = mock_post.call_args[1]["json"]
        assert payload["entity_id"] == "switch.boiler"
        assert payload["extra_param"] == "foo"


class TestTurnOnOff:
    def test_turn_on_uses_correct_domain(self):
        client = _make_client()
        with patch.object(client, "call_service", return_value=True) as mock_cs:
            client.turn_on("switch.boiler")
        mock_cs.assert_called_once_with("switch", "turn_on", "switch.boiler")

    def test_turn_off_uses_correct_domain(self):
        client = _make_client()
        with patch.object(client, "call_service", return_value=True) as mock_cs:
            client.turn_off("switch.boiler")
        mock_cs.assert_called_once_with("switch", "turn_off", "switch.boiler")

    def test_turn_on_light_entity(self):
        client = _make_client()
        with patch.object(client, "call_service", return_value=True) as mock_cs:
            client.turn_on("light.kitchen")
        mock_cs.assert_called_once_with("light", "turn_on", "light.kitchen")


# ---------------------------------------------------------------------------
# is_entity_on()
# ---------------------------------------------------------------------------

class TestIsEntityOn:
    @pytest.mark.parametrize("state,expected", [
        ("on", True),
        ("home", True),
        ("true", True),
        ("1", True),
        ("off", False),
        ("away", False),
        ("false", False),
        ("0", False),
        ("unavailable", False),
    ])
    def test_state_values(self, state, expected):
        client = _make_client()
        with patch.object(client, "get_state", return_value=_state_obj(state=state)):
            assert client.is_entity_on("sensor.x") == expected

    def test_returns_none_when_entity_unavailable(self):
        client = _make_client()
        with patch.object(client, "get_state", return_value=None):
            assert client.is_entity_on("sensor.missing") is None


# ---------------------------------------------------------------------------
# get_attribute()
# ---------------------------------------------------------------------------

class TestGetAttribute:
    def test_returns_attribute_value(self):
        client = _make_client()
        state = _state_obj(attributes={"unit_of_measurement": "°C", "friendly_name": "Temp"})
        with patch.object(client, "get_state", return_value=state):
            assert client.get_attribute("sensor.temp", "unit_of_measurement") == "°C"

    def test_returns_default_for_missing_attribute(self):
        client = _make_client()
        with patch.object(client, "get_state", return_value=_state_obj()):
            assert client.get_attribute("sensor.x", "nonexistent", default="N/A") == "N/A"

    def test_returns_default_when_entity_missing(self):
        client = _make_client()
        with patch.object(client, "get_state", return_value=None):
            assert client.get_attribute("sensor.missing", "anything", default=-1) == -1

    def test_nested_numeric_attribute(self):
        client = _make_client()
        state = _state_obj(attributes={"voltage": 230.0})
        with patch.object(client, "get_state", return_value=state):
            assert client.get_attribute("sensor.power", "voltage") == pytest.approx(230.0)
