# PV production forecast from Open-Meteo (free, no API key).
#
# Open-Meteo returns hourly global *tilted* irradiance (GTI, W/m²) for a
# panel plane given by tilt and azimuth. PV output scales with irradiance:
# at standard test conditions (1000 W/m²) a 1 kWp array produces 1 kW, so
#
#     pv_kwh(hour) = GTI / 1000 × kWp × performance_ratio
#
# performance_ratio (default 0.8) covers inverter, wiring, heat and soiling
# losses. Open-Meteo radiation values are the mean over the *preceding*
# hour, so the value stamped HH:00 belongs to the hour (HH-1):00–HH:00.
#
# The boiler can only use the *surplus*: production minus what the rest of
# the house consumes. ``surplus_next_24h`` subtracts a constant base load.
#
# API: https://open-meteo.com/en/docs (parameters hourly, tilt, azimuth,
# timeformat=unixtime, forecast_days).

import logging
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

import requests

logger = logging.getLogger(__name__)

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"
CACHE_SECONDS = 30 * 60   # forecast changes slowly; one call per half hour is plenty


def compass_to_open_meteo_azimuth(compass_deg: float) -> float:
    """Convert a compass bearing (0 = north, 90 = east, 180 = south) to
    Open-Meteo's convention (0 = south, -90 = east, 90 = west)."""
    az = (compass_deg - 180.0) % 360.0
    return az - 360.0 if az > 180.0 else az


class OpenMeteoPVForecast:
    """Hourly PV energy forecast for one panel orientation.

    Args:
        latitude, longitude: site location (from Home Assistant's config).
        kwp: installed peak power in kWp.
        tilt_deg: panel tilt, 0 = flat, 90 = vertical.
        azimuth_compass_deg: direction the panels face, 180 = south.
        performance_ratio: share of the ideal output actually delivered.
    """

    def __init__(
        self,
        latitude: float,
        longitude: float,
        kwp: float,
        tilt_deg: float = 35.0,
        azimuth_compass_deg: float = 180.0,
        performance_ratio: float = 0.8,
        timeout: int = 15,
    ):
        self.latitude = latitude
        self.longitude = longitude
        self.kwp = kwp
        self.tilt_deg = tilt_deg
        self.azimuth = compass_to_open_meteo_azimuth(azimuth_compass_deg)
        self.performance_ratio = performance_ratio
        self.timeout = timeout
        self._cache: Dict[int, float] = {}   # unix ts of hour start → kWh
        self._cache_time = 0.0

    def hourly_kwh(self) -> Dict[int, float]:
        """Return {unix timestamp of hour start: expected PV kWh}.

        Results are cached for CACHE_SECONDS. Raises on network/format
        errors so the caller can fall back to another source.
        """
        if self._cache and time.monotonic() - self._cache_time < CACHE_SECONDS:
            return self._cache
        params = {
            "latitude": self.latitude,
            "longitude": self.longitude,
            "hourly": "global_tilted_irradiance",
            "tilt": self.tilt_deg,
            "azimuth": self.azimuth,
            "timeformat": "unixtime",
            "forecast_days": 2,
        }
        resp = requests.get(OPEN_METEO_URL, params=params, timeout=self.timeout)
        resp.raise_for_status()
        hourly = resp.json()["hourly"]
        result: Dict[int, float] = {}
        for ts, gti in zip(hourly["time"], hourly["global_tilted_irradiance"]):
            if gti is None:
                continue
            hour_start = int(ts) - 3600   # value is the mean of the preceding hour
            result[hour_start] = max(0.0, float(gti)) / 1000.0 * self.kwp * self.performance_ratio
        if not result:
            raise ValueError("Open-Meteo returned no irradiance values")
        self._cache, self._cache_time = result, time.monotonic()
        return result

    def surplus_next_24h(self, from_dt: datetime, base_load_kw: float = 0.0) -> List[float]:
        """PV surplus (kWh) for the 24 hours starting at ``from_dt``'s hour.

        Surplus = forecast production − base load, never negative. Hours the
        forecast does not cover count as 0.
        """
        start = from_dt.astimezone(timezone.utc).replace(minute=0, second=0, microsecond=0)
        hourly = self.hourly_kwh()
        out = []
        for i in range(24):
            ts = int((start + timedelta(hours=i)).timestamp())
            out.append(round(max(0.0, hourly.get(ts, 0.0) - base_load_kw), 4))
        return out


def build_from_config(cfg: dict, location: Optional[Dict]) -> Optional[OpenMeteoPVForecast]:
    """Create a forecaster when PV is enabled and kWp and location are known.

    ``location`` is Home Assistant's /api/config response (latitude,
    longitude). Returns None when the forecast cannot be used.
    """
    if not cfg.get("has_pv") or cfg.get("pv_forecast_source", "auto") != "auto":
        return None
    kwp = float(cfg.get("pv_installed_power_kw", 0.0) or 0.0)
    lat = (location or {}).get("latitude")
    lon = (location or {}).get("longitude")
    if kwp <= 0 or lat is None or lon is None:
        return None
    return OpenMeteoPVForecast(
        latitude=float(lat),
        longitude=float(lon),
        kwp=kwp,
        tilt_deg=float(cfg.get("pv_tilt_deg", 35)),
        azimuth_compass_deg=float(cfg.get("pv_azimuth_deg", 180)),
    )
