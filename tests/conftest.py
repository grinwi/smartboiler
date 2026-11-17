"""Shared pytest configuration.

Tests import ``smartboiler`` from ``src/`` (see pytest.ini) and need only the
runtime dependencies from requirements.txt.

``influxdb`` is an optional dependency (used only by the one-off InfluxDB
history import in the setup wizard; the add-on image installs it). It is
stubbed so the suite runs without it; tests that exercise it patch
``influxdb.InfluxDBClient`` themselves.
"""
import sys
from unittest.mock import MagicMock

if "influxdb" not in sys.modules:
    sys.modules["influxdb"] = MagicMock()
