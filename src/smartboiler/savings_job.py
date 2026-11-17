# Savings analysis as a background job for the add-on dashboard.
#
# The dashboard starts one analysis at a time (a year takes ~30 s) and polls
# status(). Data comes from Home Assistant's long-term statistics
# (ha_statistics.py), boiler and tariff settings from the saved setup config.

import logging
import threading
from datetime import datetime, timedelta, timezone
from typing import Callable, Dict, Optional

from smartboiler import savings as sv
from smartboiler.ha_statistics import HAStatistics
from smartboiler.setup_config import load_setup_config
from smartboiler.tariff import Tariff

logger = logging.getLogger(__name__)

HEAT_UNITS = ("kWh", "Wh", "MWh")   # a "water" sensor in these units already measures heat
MIN_DAYS, MAX_DAYS = 7, 730

_lock = threading.Lock()
_state: Dict = {"status": "idle"}


def status() -> Dict:
    with _lock:
        return dict(_state)


def _set(**kw) -> None:
    with _lock:
        _state.clear()
        _state.update(kw)


def default_params(cfg: Optional[dict] = None) -> Dict:
    """Pre-fill the form from the setup config."""
    cfg = cfg if cfg is not None else load_setup_config()
    return {
        "water_entity": cfg.get("boiler_water_flow_entity_id") or "",
        "pv_entity": cfg.get("pv_power_entity_id") or "",
        "days": 365,
        "hot_temp": 50,
    }


def start(params: Dict, stats_factory: Callable[[], HAStatistics] = HAStatistics,
          background: bool = True) -> Dict:
    """Start an analysis unless one is running. Returns the new status."""
    if not str(params.get("water_entity", "")).strip():
        return {"status": "error", "error": "Choose the hot-water sensor."}
    with _lock:
        if _state.get("status") == "running":
            return dict(_state)
        _state.clear()
        _state.update(status="running", started_at=datetime.now(timezone.utc).isoformat())
    if background:
        threading.Thread(target=run, args=(params, stats_factory), daemon=True).start()
    else:
        run(params, stats_factory)
    return status()


def run(params: Dict, stats_factory: Callable[[], HAStatistics] = HAStatistics) -> None:
    """Fetch data, simulate the strategies and store the result in the state."""
    try:
        cfg = load_setup_config()
        days = max(MIN_DAYS, min(MAX_DAYS, int(params.get("days", 365))))
        end = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
        start_dt = end - timedelta(days=days)
        water_id = params["water_entity"].strip()
        pv_id = str(params.get("pv_entity", "") or "").strip()

        with stats_factory() as ha:
            tz = ha.call("get_config").get("time_zone") or "UTC"
            water, unit = ha.hourly(water_id, start_dt, end)
            pv = ha.hourly(pv_id, start_dt, end)[0] if pv_id else None
        if water.empty:
            raise ValueError(f"No statistics for {water_id} in the last {days} days.")

        if unit in HEAT_UNITS:
            demand = water
        else:
            demand = sv.water_to_heat_kwh(water, float(params.get("hot_temp", 50)),
                                          float(cfg.get("cold_water_temp", 10)))

        tariff = Tariff.from_config(cfg)
        spot = None
        if cfg.get("has_spot_price") and (tariff.mode == "spot" or tariff.export_at_spot):
            from smartboiler.spot_price import SpotPriceFetcher
            spot = SpotPriceFetcher(cfg.get("spot_price_region", "CZ")).fetch_series(start_dt, end)

        inputs = sv.build_inputs(
            demand, pv, base_load_kw=float(cfg.get("house_base_load_w", 300)) / 1000.0,
            tz=tz, spot=spot, hdo_schedule=str(cfg.get("hdo_explicit_schedule", "") or ""),
        )
        results = sv.analyse(inputs, sv.boiler_from_config(cfg), tariff)
        _set(status="done", result=sv.results_to_dict(results, inputs),
             params={"water_entity": water_id, "pv_entity": pv_id, "days": days})
        logger.info("Savings analysis finished for %d days", days)
    except Exception as e:
        logger.warning("Savings analysis failed: %s", e)
        _set(status="error", error=str(e))
