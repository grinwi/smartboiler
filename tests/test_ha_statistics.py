"""Tests for the HA statistics WebSocket client (fake socket, no network)."""
import json
import sys
import types
from datetime import datetime, timezone

import pandas as pd
import pytest

from smartboiler import ha_statistics as hs


class FakeSocket:
    """Plays the HA side of the WebSocket protocol."""

    def __init__(self, token_ok=True, units=None, rows=None, statistic_ids=None):
        self.token_ok, self.units, self.rows = token_ok, units or {}, rows or {}
        self.statistic_ids = statistic_ids or []
        self.outbox = [{"type": "auth_required"}]
        self.sent = []
        self.closed = False

    def send(self, raw):
        msg = json.loads(raw)
        self.sent.append(msg)
        if msg["type"] == "auth":
            self.outbox.append({"type": "auth_ok" if self.token_ok else "auth_invalid",
                                "message": "Invalid access token"})
            return
        result = None
        if msg["type"] == "recorder/get_statistics_metadata":
            sid = msg["statistic_ids"][0]
            result = ([{"statistic_id": sid, "statistics_unit_of_measurement": self.units[sid]}]
                      if sid in self.units else [])
        elif msg["type"] == "recorder/statistics_during_period":
            sid = msg["statistic_ids"][0]
            result = {sid: self.rows.get(sid, [])}
        elif msg["type"] == "recorder/list_statistic_ids":
            result = self.statistic_ids
        # An unrelated event first: the client must skip it.
        self.outbox.append({"type": "event", "id": 999})
        self.outbox.append({"id": msg["id"], "type": "result", "success": True, "result": result})

    def recv(self):
        return json.dumps(self.outbox.pop(0))

    def close(self):
        self.closed = True


@pytest.fixture
def fake_ws(monkeypatch):
    holder = {}

    def connect(url, timeout):
        holder["url"] = url
        return holder["socket"]

    monkeypatch.setitem(sys.modules, "websocket", types.SimpleNamespace(create_connection=connect))
    return holder


T0 = datetime(2025, 6, 1, tzinfo=timezone.utc)
MS = int(T0.timestamp() * 1000)


def test_default_ws_url(monkeypatch):
    monkeypatch.delenv("HA_BASE_URL", raising=False)
    assert hs.default_ws_url() == "ws://supervisor/core/websocket"
    monkeypatch.setenv("HA_BASE_URL", "https://ha.example.com/api")
    assert hs.default_ws_url() == "wss://ha.example.com/websocket"


def test_rows_to_series_accepts_ms_and_iso():
    rows = [{"start": MS, "change": 5.0}, {"start": "2025-06-01T01:00:00+00:00", "change": 2.0},
            {"start": MS + 7_200_000, "change": None}]
    s = hs.rows_to_series(rows, "change", 1000.0)
    assert list(s) == [5000.0, 2000.0]
    assert s.index[0] == pd.Timestamp(T0)


def test_auth_failure_raises(fake_ws):
    fake_ws["socket"] = FakeSocket(token_ok=False)
    with pytest.raises(PermissionError):
        with hs.HAStatistics(url="ws://x", token="bad"):
            pass
    assert fake_ws["socket"].closed


def test_counter_uses_change_and_converts_m3(fake_ws):
    fake_ws["socket"] = sock = FakeSocket(
        units={"sensor.water": "m³"},
        rows={"sensor.water": [{"start": MS, "change": 0.05}, {"start": MS + 3_600_000, "change": 0.12}]})
    with hs.HAStatistics(url="ws://x", token="t") as ha:
        s, unit = ha.hourly("sensor.water", T0, T0)
    assert unit == "m³" and list(s.round(6)) == [50.0, 120.0]
    query = next(m for m in sock.sent if m["type"] == "recorder/statistics_during_period")
    assert query["types"] == ["change"] and query["period"] == "hour"
    assert sock.sent[0] == {"type": "auth", "access_token": "t"}


def test_power_uses_mean_to_kwh(fake_ws):
    fake_ws["socket"] = FakeSocket(units={"sensor.pv": "W"},
                                   rows={"sensor.pv": [{"start": MS, "mean": 2500.0}]})
    with hs.HAStatistics(url="ws://x", token="t") as ha:
        s, _ = ha.hourly("sensor.pv", T0, T0)
    assert list(s) == [2.5]


def test_flow_mean_to_litres_per_hour(fake_ws):
    fake_ws["socket"] = FakeSocket(units={"sensor.flow": "L/min"},
                                   rows={"sensor.flow": [{"start": MS, "mean": 0.5}]})
    with hs.HAStatistics(url="ws://x", token="t") as ha:
        s, _ = ha.hourly("sensor.flow", T0, T0)
    assert list(s) == [30.0]


def test_missing_statistics_explained(fake_ws):
    fake_ws["socket"] = FakeSocket()
    with hs.HAStatistics(url="ws://x", token="t") as ha:
        with pytest.raises(ValueError, match="state_class"):
            ha.hourly("sensor.nostats", T0, T0)


def test_usable_sensors_filters_units_and_domain(fake_ws):
    fake_ws["socket"] = FakeSocket(statistic_ids=[
        {"statistic_id": "sensor.water", "statistics_unit_of_measurement": "L", "name": "Water"},
        {"statistic_id": "sensor.temp", "statistics_unit_of_measurement": "°C", "name": "Temp"},
        {"statistic_id": "sensor.pv", "statistics_unit_of_measurement": "kWh", "name": None},
        {"statistic_id": "external:prices", "statistics_unit_of_measurement": "kWh", "name": "X"},
    ])
    with hs.HAStatistics(url="ws://x", token="t") as ha:
        sensors = ha.usable_sensors()
    assert [s["entity_id"] for s in sensors] == ["sensor.pv", "sensor.water"]
    assert sensors[0]["name"] == "sensor.pv"
