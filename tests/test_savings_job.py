"""Tests for the background savings job and its web routes (fake HA statistics)."""
import json
import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from smartboiler import savings_job


class FakeStats:
    """Context manager standing in for HAStatistics."""

    def __init__(self, series, tz="Europe/Prague"):
        self.series, self.tz = series, tz

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def call(self, msg_type, **params):
        assert msg_type == "get_config"
        return {"time_zone": self.tz}

    def hourly(self, sid, start, end):
        if sid not in self.series:
            raise ValueError(f"{sid}: no long-term statistics")
        return self.series[sid]


def _series(days=4):
    idx = pd.date_range("2025-06-01", periods=24 * days, freq="h", tz="UTC")
    litres = pd.Series(np.where(idx.hour.isin([5, 18]), 40.0, 0.0), index=idx)
    pv = pd.Series(np.where(idx.hour.isin(range(8, 15)), 2.5, 0.0), index=idx)
    return {"sensor.water": (litres, "L"), "sensor.pv": (pv, "kWh")}


@pytest.fixture(autouse=True)
def data_dir(monkeypatch):
    monkeypatch.setenv("DATA_PATH", tempfile.mkdtemp())
    savings_job._set(status="idle")


def test_run_produces_result():
    st = savings_job.start({"water_entity": "sensor.water", "pv_entity": "sensor.pv", "days": 30},
                           stats_factory=lambda: FakeStats(_series()), background=False)
    assert st["status"] == "done", st
    r = st["result"]
    assert [s["name"] for s in r["strategies"]] == ["thermostat", "smart_ideal", "smart_real"]
    assert r["hot_water_kwh"] > 0 and r["pv_surplus_kwh"] > 0
    assert r["strategies"][0]["saving"] == 0
    assert "# SmartBoiler savings analysis" in r["report_md"]
    json.dumps(r)   # must be JSON-serialisable for the API


def test_water_entity_required():
    assert savings_job.start({"water_entity": ""})["status"] == "error"


def test_errors_are_reported_not_raised():
    st = savings_job.start({"water_entity": "sensor.missing"},
                           stats_factory=lambda: FakeStats(_series()), background=False)
    assert st["status"] == "error" and "no long-term statistics" in st["error"]


def test_only_one_job_at_a_time():
    savings_job._set(status="running")
    st = savings_job.start({"water_entity": "sensor.water"}, stats_factory=lambda: FakeStats(_series()),
                           background=False)
    assert st["status"] == "running"


def test_heat_sensor_is_used_directly():
    s = _series()
    kwh = (s["sensor.water"][0] * 0.04, "kWh")
    st = savings_job.start({"water_entity": "sensor.heat", "days": 30},
                           stats_factory=lambda: FakeStats({"sensor.heat": kwh}), background=False)
    assert st["result"]["hot_water_kwh"] == pytest.approx(float(kwh[0].sum()), abs=0.1)


def test_default_params_from_config():
    d = savings_job.default_params({"boiler_water_flow_entity_id": "sensor.flow", "pv_power_entity_id": ""})
    assert d["water_entity"] == "sensor.flow" and d["pv_entity"] == "" and d["days"] == 365


# ── Web routes ───────────────────────────────────────────────────────────────

@pytest.fixture
def client():
    from smartboiler.web_server import app
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


def test_savings_page_renders(client):
    resp = client.get("/savings")
    assert resp.status_code == 200 and b"Run analysis" in resp.data


def test_status_endpoint(client):
    assert client.get("/api/savings").get_json() == {"status": "idle"}


def test_start_endpoint_validates(client):
    resp = client.post("/api/savings", json={"water_entity": ""})
    assert resp.status_code == 400


def test_sensors_endpoint_reports_ha_errors(client, monkeypatch):
    import smartboiler.ha_statistics as hs

    def boom(*a, **k):
        raise ConnectionError("no supervisor")
    monkeypatch.setattr(hs.HAStatistics, "__enter__", boom)
    resp = client.get("/api/savings/sensors")
    assert resp.status_code == 502 and "no supervisor" in resp.get_json()["error"]
