# Home Assistant long-term statistics over the WebSocket API.
#
# The recorder keeps raw states only for a few days (purge_keep_days, 10 by
# default) but keeps hourly statistics for every sensor with a state_class
# forever. Those hourly rows are exactly what the savings analysis needs:
#
#   counters (state_class total / total_increasing, e.g. L, m³, kWh)
#       → "change": how much the counter grew in each hour
#   measurements (state_class measurement, e.g. W, L/min)
#       → "mean": average value in each hour
#
# Statistics are only available over the WebSocket API. Inside the add-on we
# connect through the Supervisor proxy with SUPERVISOR_TOKEN; the token is
# sent in the auth message, as the protocol requires.
# Protocol: https://developers.home-assistant.io/docs/api/websocket

import json
import os
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd

# Statistics unit → (kind, factor). Counters become litres or kWh per hour;
# power becomes kWh per hour; flow rates become litres per hour.
STAT_UNITS: Dict[str, tuple] = {
    "L": ("counter", 1.0), "m³": ("counter", 1000.0), "gal": ("counter", 3.785),
    "kWh": ("counter", 1.0), "Wh": ("counter", 0.001), "MWh": ("counter", 1000.0),
    "W": ("power", 0.001), "kW": ("power", 1.0),
    "L/min": ("flow", 60.0), "L/h": ("flow", 1.0), "m³/h": ("flow", 1000.0),
}


def default_ws_url() -> str:
    """WebSocket URL derived from HA_BASE_URL (Supervisor proxy by default)."""
    base = os.getenv("HA_BASE_URL", "http://supervisor/core/api").rstrip("/")
    if base.endswith("/api"):
        base = base[: -len("/api")]
    return base.replace("https://", "wss://", 1).replace("http://", "ws://", 1) + "/websocket"


class HAStatistics:
    """Read-only access to recorder statistics. Use as a context manager:

        with HAStatistics() as ha:
            series, unit = ha.hourly("sensor.water_total", start, end)
    """

    def __init__(self, url: Optional[str] = None, token: Optional[str] = None, timeout: int = 60):
        self.url = url or default_ws_url()
        self.token = token if token is not None else os.getenv("SUPERVISOR_TOKEN", "")
        self.timeout = timeout
        self._ws = None
        self._next_id = 1

    def __enter__(self) -> "HAStatistics":
        import websocket   # websocket-client; imported lazily, only this feature needs it

        self._ws = websocket.create_connection(self.url, timeout=self.timeout)
        self._recv()                                   # auth_required
        self._send({"type": "auth", "access_token": self.token})
        reply = self._recv()
        if reply.get("type") != "auth_ok":
            self.close()
            raise PermissionError(f"Home Assistant refused the token: {reply.get('message', reply)}")
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def close(self) -> None:
        if self._ws is not None:
            self._ws.close()
            self._ws = None

    def _send(self, msg: dict) -> None:
        self._ws.send(json.dumps(msg))

    def _recv(self) -> dict:
        return json.loads(self._ws.recv())

    def call(self, msg_type: str, **params):
        """Send one command and return its result (raises on error)."""
        msg_id = self._next_id
        self._next_id += 1
        self._send({"id": msg_id, "type": msg_type, **params})
        while True:
            reply = self._recv()
            if reply.get("id") == msg_id and reply.get("type") == "result":
                if not reply.get("success"):
                    raise RuntimeError(f"{msg_type} failed: {reply.get('error')}")
                return reply["result"]

    # ── Queries ─────────────────────────────────────────────────────────

    def usable_sensors(self) -> List[dict]:
        """Sensors with statistics in a unit the analysis understands."""
        out = []
        for item in self.call("recorder/list_statistic_ids"):
            unit = item.get("statistics_unit_of_measurement")
            sid = item.get("statistic_id", "")
            if sid.startswith("sensor.") and unit in STAT_UNITS:
                out.append({"entity_id": sid, "unit": unit, "name": item.get("name") or sid})
        return sorted(out, key=lambda x: x["entity_id"])

    def unit_of(self, statistic_id: str) -> Optional[str]:
        meta = self.call("recorder/get_statistics_metadata", statistic_ids=[statistic_id])
        return meta[0].get("statistics_unit_of_measurement") if meta else None

    def hourly(self, statistic_id: str, start: datetime, end: datetime) -> tuple:
        """Hourly amounts (litres or kWh per hour) with a UTC index, and the unit.

        Raises ValueError when the sensor has no statistics or an unknown unit.
        """
        unit = self.unit_of(statistic_id)
        if unit not in STAT_UNITS:
            raise ValueError(
                f"{statistic_id}: no long-term statistics in a supported unit "
                f"(got {unit!r}; the sensor needs a state_class)"
            )
        kind, factor = STAT_UNITS[unit]
        field = "change" if kind == "counter" else "mean"
        rows = self.call(
            "recorder/statistics_during_period",
            start_time=start.isoformat(), end_time=end.isoformat(),
            statistic_ids=[statistic_id], period="hour", types=[field],
        ).get(statistic_id, [])
        return rows_to_series(rows, field, factor), unit


def rows_to_series(rows: List[dict], field: str, factor: float) -> pd.Series:
    """Convert statistics rows to a Series. ``start`` is epoch ms (HA ≥ 2023.3)
    or an ISO string (older releases)."""
    starts, values = [], []
    for r in rows:
        if r.get(field) is None:
            continue
        st = r["start"]
        starts.append(pd.Timestamp(st, unit="ms", tz="UTC") if isinstance(st, (int, float))
                      else pd.Timestamp(st).tz_convert("UTC"))
        values.append(max(0.0, float(r[field])) * factor)
    return pd.Series(values, index=pd.DatetimeIndex(starts), dtype=float).sort_index()
