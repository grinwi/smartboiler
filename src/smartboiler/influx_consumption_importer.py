# Created as a part of Master's Thesis "Using machine learning methods to save energy in a smart home"
# Faculty of Information Technology, Brno University of Technology, 2024
# Author: Adam Grünwald
#
# Consumption history importer from InfluxDB.
# Pulls water-flow + outlet-temp at 1-min resolution, computes kWh/min,
# resamples to hourly rows, and merges into the local consumption cache.
# Also seeds the HDO learner from relay "unavailable" state history.

import logging
from datetime import datetime

import pandas as pd

from smartboiler._influx_helpers import fmt_ts, query_series

logger = logging.getLogger(__name__)

# Maximum gap (in 1-minute buckets) to forward-fill after a sensor outage.
_RELAY_MAX_FFILL_MIN = 60   # relay offline > 1 h → assume unknown (→ False)
_TEMP_MAX_FFILL_MIN  = 120  # temperature offline > 2 h → NaN


def fetch_consumption_chunk(
    client,
    relay_entity: str,
    flow_entity: str,
    water_tmp_entity: str,
    power_entity: str,
    cold_water_tmp: float,
    boiler_watt: float,
    meas_state: str,
    meas_flow: str,
    meas_temp: str,
    meas_power: str,
    start: datetime,
    end: datetime,
) -> pd.DataFrame:
    """
    Pull one time chunk from InfluxDB and return an hourly consumption DataFrame.

    Columns returned: consumed_kwh, relay_on, power_w
    Index: tz-naive DatetimeIndex at 1-hour frequency.

    If flow + water-temp sensors are available, consumption is computed as
    Q = flow × 4.186 × ΔT / 3600 kWh/min.
    Otherwise falls back to relay on-time × wattage.
    """
    t0 = fmt_ts(start)
    t1 = fmt_ts(end)

    relay_series = query_series(
        client, meas=meas_state, entity=relay_entity,
        field="value", agg="last", t0=t0, t1=t1, fill="null", dtype="bool",
    )
    if relay_series.empty:
        return pd.DataFrame()

    relay_series = relay_series.ffill(limit=_RELAY_MAX_FFILL_MIN).fillna(False)

    if flow_entity and water_tmp_entity:
        flow_series = query_series(
            client, meas=meas_flow, entity=flow_entity,
            field="value", agg="mean", t0=t0, t1=t1, fill="0", dtype="float",
        )
        temp_series = query_series(
            client, meas=meas_temp, entity=water_tmp_entity,
            field="value", agg="mean", t0=t0, t1=t1, fill="null", dtype="float",
        )
        temp_series = temp_series.ffill(limit=_TEMP_MAX_FFILL_MIN)
        delta_t = (temp_series - cold_water_tmp).clip(lower=0)
        consumed_min = flow_series.fillna(0) * 4.186 * delta_t / 3600.0
    else:
        # Fallback: estimate from relay on-time
        consumed_min = relay_series.astype(float) * (boiler_watt / 60_000.0)
        logger.debug("InfluxDB bootstrap: no flow sensor; estimating from relay on-time")

    power_series = pd.Series(dtype=float)
    if power_entity:
        power_series = query_series(
            client, meas=meas_power, entity=power_entity,
            field="value", agg="mean", t0=t0, t1=t1, fill="0", dtype="float",
        )

    df = pd.DataFrame({
        "consumed_kwh": consumed_min,
        "relay_on": relay_series,
        "power_w": power_series if not power_series.empty else 0.0,
    })
    df.index = pd.to_datetime(df.index)

    df_hourly = df.resample("1h").agg({
        "consumed_kwh": lambda x: x.sum(min_count=1),
        "relay_on": "mean",
        "power_w": "mean",
    })
    df_hourly["relay_on"] = df_hourly["relay_on"] > 0.5
    return df_hourly.dropna(how="all")


def seed_hdo_learner(
    client,
    relay_entity: str,
    meas_state: str,
    hdo_learner,
    start: datetime,
    end: datetime,
) -> int:
    """
    Feed the HDO learner from InfluxDB relay state history.

    Scans minute-resolution relay states looking for "unavailable" (HDO blocking).
    Returns the number of observations recorded.
    """
    t0 = fmt_ts(start)
    t1 = fmt_ts(end)

    unavail_series = query_series(
        client, meas=meas_state, entity=relay_entity,
        field="value", agg="last", t0=t0, t1=t1, fill="null", dtype="unavailable",
    )
    if unavail_series.empty:
        return 0

    relay_bool = query_series(
        client, meas=meas_state, entity=relay_entity,
        field="value", agg="last", t0=t0, t1=t1, fill="null", dtype="bool",
    )

    idx = unavail_series.index.union(
        relay_bool.index if not relay_bool.empty else unavail_series.index
    )
    unavail = unavail_series.reindex(idx)
    known = relay_bool.reindex(idx).notna() | unavail.notna()

    n = 0
    for ts in idx[known]:
        try:
            dt = ts.to_pydatetime()
            is_unavailable = bool(unavail.get(ts, False) or False)
            hdo_learner.observe(dt, relay_unavailable=is_unavailable)
            n += 1
        except Exception:
            continue

    return n
