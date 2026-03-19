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


def fetch_consumption_chunk_simple_mode(
    client,
    relay_entity: str,
    case_tmp_entity: str,
    inlet_tmp_entity: str,
    outlet_tmp_entity: str,
    power_entity: str,
    coupling: float,
    T_set: float,
    cold_water_tmp: float,
    standby_w: float,
    boiler_volume: float,
    meas_state: str,
    meas_temp: str,
    meas_power: str,
    start: datetime,
    end: datetime,
) -> pd.DataFrame:
    """
    Simple-mode bootstrap: pull T_case + T_in + T_out + power from InfluxDB
    and compute energy-balance estimates (litres consumed per day → synthetic
    hourly kWh rows).

    Uses the same formula validated in analysis/raw_data_analysis.ipynb:
        T_water = T_amb + (T_case - T_amb) / coupling
        est_L   = kWh_net * 3600 / (Cp * (T_water - T_cold))

    Returns hourly DataFrame with columns: consumed_kwh, relay_on, power_w
    (same schema as fetch_consumption_chunk so the bootstrapper can use either).
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

    power_series = pd.Series(dtype=float)
    if power_entity:
        power_series = query_series(
            client, meas=meas_power, entity=power_entity,
            field="value", agg="mean", t0=t0, t1=t1, fill="null", dtype="float",
        )

    # Fall back to standby_w × time if no real power meter
    if power_series.empty or power_series.isna().all():
        logger.debug("Simple-mode bootstrap: no power entity; estimating from relay on-time")
        power_series = relay_series.astype(float) * standby_w  # will be overridden below
    else:
        power_series = power_series.ffill(limit=_TEMP_MAX_FFILL_MIN).fillna(0.0)

    case_series = pd.Series(dtype=float)
    if case_tmp_entity:
        case_series = query_series(
            client, meas=meas_temp, entity=case_tmp_entity,
            field="value", agg="mean", t0=t0, t1=t1, fill="null", dtype="float",
        ).ffill(limit=_TEMP_MAX_FFILL_MIN)

    inlet_series = pd.Series(dtype=float)
    if inlet_tmp_entity:
        inlet_series = query_series(
            client, meas=meas_temp, entity=inlet_tmp_entity,
            field="value", agg="mean", t0=t0, t1=t1, fill="null", dtype="float",
        ).ffill(limit=_TEMP_MAX_FFILL_MIN)

    outlet_series = pd.Series(dtype=float)
    if outlet_tmp_entity:
        outlet_series = query_series(
            client, meas=meas_temp, entity=outlet_tmp_entity,
            field="value", agg="mean", t0=t0, t1=t1, fill="null", dtype="float",
        ).ffill(limit=_TEMP_MAX_FFILL_MIN)

    # ── Build 1-min aligned frame ─────────────────────────────────────────
    idx = relay_series.index
    df1 = pd.DataFrame({
        "relay_on": relay_series,
        "power_w":  power_series.reindex(idx).fillna(0.0),
        "T_case":   case_series.reindex(idx) if not case_series.empty else float("nan"),
        "T_in":     inlet_series.reindex(idx) if not inlet_series.empty else float("nan"),
        "T_out":    outlet_series.reindex(idx) if not outlet_series.empty else float("nan"),
    })
    df1.index = pd.to_datetime(df1.index)

    # ── Daily energy balance → estimated litres ───────────────────────────
    # kWh_net = total kWh from power meter minus standby losses
    # T_water from case sensor; T_cold from inlet probe or config default
    Cp = 4.186 / 3600.0  # kWh per litre per °C

    daily = df1.resample("1D").agg(
        relay_on=("relay_on", "mean"),
        power_kwh=("power_w", lambda x: x.sum() / 60_000.0),  # W·min → kWh
        T_case_med=("T_case", "median"),
        T_in_med=("T_in", "median"),
        T_out_med=("T_out", "median"),
    ).dropna(subset=["power_kwh"])

    daily["kWh_net"] = (daily["power_kwh"] - standby_w * 24.0 / 1000.0).clip(lower=0.0)

    T_case_med = daily["T_case_med"].fillna(float("nan"))
    T_in_med   = daily["T_in_med"].fillna(cold_water_tmp)
    T_out_med  = daily["T_out_med"].fillna(float("nan"))

    # Prefer a valid outlet-pipe reading when available; otherwise infer tank
    # water temperature from the case sensor and inlet/coupling model.
    T_from_case = (T_in_med + (T_case_med - T_in_med) / coupling).clip(lower=30.0, upper=95.0)
    valid_outlet = T_out_med.notna() & ((T_out_med - T_in_med).fillna(0.0) >= 5.0)
    T_water = T_from_case.where(~valid_outlet, other=T_out_med.clip(lower=30.0, upper=95.0))
    T_water = T_water.where(T_case_med.notna() | valid_outlet, other=float(T_set))

    dT = (T_water - T_in_med).clip(lower=5.0)
    daily["est_L"] = (daily["kWh_net"] / (Cp * dT)).clip(lower=0.0)

    # ── Convert daily litres → synthetic hourly consumed_kwh ─────────────
    # Spread the day's heat uniformly over relay-ON hours (same as
    # _push_simple_mode_estimate_to_history in controller.py)
    frames = []
    for day, row in daily.iterrows():
        day_str = pd.Timestamp(day).floor("1D")
        day_df = df1.loc[day_str:day_str + pd.Timedelta(hours=23, minutes=59)]
        relay_hourly = day_df["relay_on"].resample("1h").mean()
        relay_on_h = float(relay_hourly.sum()) or 1.0
        kwh_total = row["kWh_net"]
        # Heat per relay-ON hour proportional to relay fraction
        kwh_per_h = relay_hourly * (kwh_total / relay_on_h)
        day_frame = pd.DataFrame({
            "consumed_kwh": kwh_per_h,
            "relay_on": relay_hourly > 0.5,
            "power_w": day_df["power_w"].resample("1h").mean(),
        })
        frames.append(day_frame)

    if not frames:
        return pd.DataFrame()

    result = pd.concat(frames)
    result = result[~result.index.duplicated(keep="last")].sort_index()
    return result.dropna(how="all")


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
