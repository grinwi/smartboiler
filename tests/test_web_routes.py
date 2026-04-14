"""
Tests for web routes:
  1. Ingress-compatible redirects (relative, not absolute)
  2. /api/test/influxdb returns {measurement, entity_id, display} objects
  3. /api/bootstrap/influxdb exposes status and start endpoints
  4. controller.start() has no inner `import threading` that shadows the module-level one
"""

import os
import tempfile
import types
from unittest.mock import MagicMock, patch

import pytest

# ── Flask test client setup ───────────────────────────────────────────────────

@pytest.fixture(scope="module")
def client():
    """Create a Flask test client with a temp DATA_PATH."""
    tmp = tempfile.mkdtemp()
    os.environ["DATA_PATH"] = tmp

    # Import after env var is set
    from smartboiler.web_server import app
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


# ── 1. Relative redirect for HA ingress ──────────────────────────────────────

class TestIngressRedirect:
    def test_root_redirects_to_relative_setup(self, client):
        """GET / must redirect to 'setup' (relative), not '/setup' (absolute).

        HA ingress strips the /api/hassio_ingress/TOKEN prefix, so an absolute
        redirect would land on HA's own /setup → 404.
        """
        resp = client.get("/")
        assert resp.status_code in (301, 302, 307, 308)
        location = resp.headers.get("Location", "")
        # Must not start with a slash (absolute path) or contain a scheme (absolute URL)
        assert not location.startswith("/"), (
            f"Redirect to '{location}' is absolute — breaks HA ingress. "
            "Use redirect('setup') not redirect('/setup')."
        )

    def test_root_redirect_destination_is_setup(self, client):
        """The redirect must point at 'setup' (the wizard route)."""
        resp = client.get("/")
        location = resp.headers.get("Location", "")
        assert "setup" in location


# ── 2. InfluxDB endpoint returns {measurement, entity_id, display} ────────────

class TestInfluxDBEndpoint:
    def _mock_influx_client(self, measurements, tag_values_by_measurement):
        """Build a mock InfluxDBClient that returns the given data."""
        mock_client = MagicMock()
        mock_client.ping.return_value = True
        mock_client.get_list_measurements.return_value = [
            {"name": m} for m in measurements
        ]

        def fake_query(query):
            # Match SHOW TAG VALUES FROM "X" WITH KEY = "entity_id"
            for mname, eids in tag_values_by_measurement.items():
                if f'FROM "{mname}"' in query:
                    result = MagicMock()
                    result.get_points.return_value = [{"value": eid} for eid in eids]
                    return result
            result = MagicMock()
            result.get_points.return_value = []
            return result

        mock_client.query.side_effect = fake_query
        return mock_client

    def test_returns_entity_objects_not_strings(self, client):
        """Each item in 'entities' must be a dict with measurement/entity_id/display."""
        mock_influx = self._mock_influx_client(
            measurements=["°C"],
            tag_values_by_measurement={"°C": ["sensor.boiler_temp"]},
        )
        with patch("influxdb.InfluxDBClient", return_value=mock_influx):
            resp = client.post(
                "/api/test/influxdb",
                json={"host": "localhost", "port": 8086, "db": "homeassistant"},
            )
        data = resp.get_json()
        assert data["ok"] is True
        entities = data["entities"]
        assert len(entities) == 1
        e = entities[0]
        assert e["measurement"] == "°C"
        assert e["entity_id"] == "sensor.boiler_temp"
        assert e["display"] == "°C :: sensor.boiler_temp"

    def test_multiple_measurements_and_entities(self, client):
        mock_influx = self._mock_influx_client(
            measurements=["°C", "state", "W"],
            tag_values_by_measurement={
                "°C": ["sensor.temp1", "sensor.temp2"],
                "state": ["switch.relay"],
                "W": ["sensor.power"],
            },
        )
        with patch("influxdb.InfluxDBClient", return_value=mock_influx):
            resp = client.post(
                "/api/test/influxdb",
                json={"host": "localhost", "port": 8086, "db": "homeassistant"},
            )
        data = resp.get_json()
        assert data["ok"] is True
        entities = data["entities"]
        assert len(entities) == 4
        displays = [e["display"] for e in entities]
        assert "°C :: sensor.temp1" in displays
        assert "state :: switch.relay" in displays

    def test_sorted_by_display(self, client):
        mock_influx = self._mock_influx_client(
            measurements=["state", "°C"],
            tag_values_by_measurement={
                "state": ["switch.z", "switch.a"],
                "°C": ["sensor.b"],
            },
        )
        with patch("influxdb.InfluxDBClient", return_value=mock_influx):
            resp = client.post(
                "/api/test/influxdb",
                json={"host": "localhost", "port": 8086, "db": "homeassistant"},
            )
        entities = resp.get_json()["entities"]
        displays = [e["display"] for e in entities]
        assert displays == sorted(displays)

    def test_skips_measurements_with_no_entity_id(self, client):
        mock_influx = self._mock_influx_client(
            measurements=["°C", "prediction"],
            tag_values_by_measurement={
                "°C": ["sensor.temp"],
                "prediction": [],  # no entity_id tags
            },
        )
        with patch("influxdb.InfluxDBClient", return_value=mock_influx):
            resp = client.post(
                "/api/test/influxdb",
                json={"host": "localhost", "port": 8086, "db": "homeassistant"},
            )
        entities = resp.get_json()["entities"]
        assert len(entities) == 1
        assert entities[0]["measurement"] == "°C"

    def test_missing_host_returns_400(self, client):
        resp = client.post("/api/test/influxdb", json={"port": 8086})
        assert resp.status_code == 400

    def test_connection_failure_returns_error(self, client):
        mock_influx = MagicMock()
        mock_influx.ping.side_effect = Exception("Connection refused")
        with patch("influxdb.InfluxDBClient", return_value=mock_influx):
            resp = client.post(
                "/api/test/influxdb",
                json={"host": "bad-host", "port": 8086, "db": "homeassistant"},
            )
        data = resp.get_json()
        assert data["ok"] is False
        assert "error" in data


class TestInfluxBootstrapAction:
    def test_status_unavailable_without_handler(self, client):
        from smartboiler.web_server import set_influx_bootstrap_handlers

        set_influx_bootstrap_handlers(None, None)
        resp = client.get("/api/bootstrap/influxdb")
        assert resp.status_code == 503
        data = resp.get_json()
        assert data["available"] is False

    def test_status_returns_provider_payload(self, client):
        from smartboiler.web_server import set_influx_bootstrap_handlers

        try:
            set_influx_bootstrap_handlers(
                lambda _cfg: {"ok": True, "started": True},
                lambda: {"available": True, "configured": True, "running": False},
            )
            resp = client.get("/api/bootstrap/influxdb")
            assert resp.status_code == 200
            data = resp.get_json()
            assert data["available"] is True
            assert data["configured"] is True
            assert data["running"] is False
        finally:
            set_influx_bootstrap_handlers(None, None)

    def test_post_saves_config_and_starts_bootstrap(self, client):
        from smartboiler.web_server import set_influx_bootstrap_handlers

        payload = {
            "boiler_switch_entity_id": "switch.boiler",
            "operation_mode": "standard",
            "influxdb_host": "influx.local",
            "influxdb_db": "homeassistant",
            "influxdb_standard_relay_entity_id": "switch.boiler",
            "influxdb_standard_power_entity_id": "sensor.boiler_power",
        }
        seen = {}

        def _start(cfg):
            seen.update(cfg)
            return {
                "ok": True,
                "started": True,
                "available": True,
                "status": {"available": True, "configured": True, "running": True},
            }

        try:
            set_influx_bootstrap_handlers(
                _start,
                lambda: {"available": True, "configured": True, "running": False},
            )
            with patch("smartboiler.web_routes.save_setup_config") as mock_save, patch(
                "smartboiler.web_routes.load_setup_config",
                return_value=payload,
            ):
                resp = client.post("/api/bootstrap/influxdb", json={"config": payload})
            assert resp.status_code == 202
            mock_save.assert_called_once_with(payload)
            assert seen["influxdb_standard_relay_entity_id"] == "switch.boiler"
            assert seen["operation_mode"] == "standard"
        finally:
            set_influx_bootstrap_handlers(None, None)

    def test_post_returns_conflict_when_bootstrap_already_running(self, client):
        from smartboiler.web_server import set_influx_bootstrap_handlers

        try:
            set_influx_bootstrap_handlers(
                lambda _cfg: {
                    "ok": False,
                    "started": False,
                    "available": True,
                    "error": "already running",
                    "status": {"available": True, "configured": True, "running": True},
                },
                lambda: {"available": True, "configured": True, "running": True},
            )
            with patch("smartboiler.web_routes.load_setup_config", return_value={"boiler_switch_entity_id": "switch.boiler"}):
                resp = client.post("/api/bootstrap/influxdb", json={})
            assert resp.status_code == 409
            assert resp.get_json()["error"] == "already running"
        finally:
            set_influx_bootstrap_handlers(None, None)


class TestTemperatureEstimationEndpoint:
    def test_returns_extra_provider_payload(self, client):
        from smartboiler.web_server import set_extra_provider

        def _provider(endpoint, _params):
            if endpoint == "temperature_estimation":
                return {
                    "estimate": 48.5,
                    "source_key": "thermal_model",
                    "source_level": "L3",
                }
            return {}

        try:
            set_extra_provider(_provider)
            resp = client.get("/api/temperature-estimation")
            assert resp.status_code == 200
            data = resp.get_json()
            assert data["estimate"] == 48.5
            assert data["source_key"] == "thermal_model"
            assert data["source_level"] == "L3"
        finally:
            set_extra_provider(lambda _endpoint, _params: {})


# ── 4. Controller has no shadowing `import threading` inside start() ──────────

class TestControllerThreadingImport:
    def test_no_inner_import_threading_in_start(self):
        """Verify the `import threading` inside start() was removed.

        Python treats a variable as local to a function if it appears in any
        assignment (including `import X`) anywhere in the function body. A local
        `import threading` before the first use of `threading.Thread` causes
        UnboundLocalError at runtime.
        """
        import ast, inspect, textwrap
        from smartboiler import controller

        source = textwrap.dedent(inspect.getsource(controller.SmartBoilerController.start))
        tree = ast.parse(source)

        inner_imports = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                for alias in getattr(node, "names", []):
                    if alias.name == "threading":
                        inner_imports.append(node)

        assert len(inner_imports) == 0, (
            "Found `import threading` inside SmartBoilerController.start(). "
            "This shadows the module-level import and causes UnboundLocalError. "
            "Remove the inner import — threading is already imported at module level."
        )
