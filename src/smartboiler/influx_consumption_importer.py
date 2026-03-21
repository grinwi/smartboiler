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
_HDO_PRESENCE_FFILL_MIN = 15   # tolerate sparse auxiliary telemetry
_HDO_MIN_GAP_MIN = 20          # shorter outages are treated as noise


def _build_minute_index(start: datetime, end: datetime) -> pd.DatetimeIndex:
    # Always produce a tz-naive index (UTC wall-clock) to match the tz-naive
    # output of query_series (which strips tz via .tz_localize(None)).
    def _naive_utc(dt) -> pd.Timestamp:
        ts = pd.Timestamp(dt)
        if ts.tzinfo is not None:
            return ts.tz_convert("UTC").tz_localize(None)
        return ts  # already tz-naive — treated as UTC

    start_ts = _naive_utc(start)
    end_ts   = _naive_utc(end)
    start_min = start_ts.floor("1min") + pd.Timedelta(minutes=1)
    end_min   = end_ts.floor("1min")
    if end_min < start_min:
        return pd.DatetimeIndex([])
    return pd.date_range(start=start_min, end=end_min, freq="1min")


def _infer_hdo_gaps(known: pd.Series, min_gap_minutes: int = _HDO_MIN_GAP_MIN) -> pd.Series:
    """Infer HDO from bounded telemetry gaps in a complete 1-minute availability mask."""
    inferred = pd.Series(False, index=known.index, dtype=bool)
    if known.empty:
        return inferred

    known_mask = known.fillna(False).astype(bool)
    missing_mask = ~known_mask
    gap_start = None

    for pos, is_missing in enumerate(missing_mask.tolist()):
        if is_missing:
            if gap_start is None:
                gap_start = pos
            continue

        if gap_start is None:
            continue

        gap_len = pos - gap_start
        if (
            gap_start > 0
            and bool(known_mask.iloc[gap_start - 1])
            and bool(known_mask.iloc[pos])
            and gap_len >= min_gap_minutes
        ):
            inferred.iloc[gap_start:pos] = True
        gap_start = None

    return inferred


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
    presence_sources=None,
) -> int:
    """
    Feed the HDO learner from InfluxDB relay state history.

    Scans minute-resolution relay states looking for explicit "unavailable"
    samples and can also infer HDO from longer telemetry gaps across auxiliary
    Shelly entities (power/temperature/flow) when those entities disappear
    together with the relay.

    Returns the number of observations recorded.
    """
    t0 = fmt_ts(start)
    t1 = fmt_ts(end)
    full_idx = _build_minute_index(start, end)
    if full_idx.empty:
        return 0

    unavail_series = query_series(
        client, meas=meas_state, entity=relay_entity,
        field="value", agg="last", t0=t0, t1=t1, fill="null", dtype="unavailable",
    )

    relay_bool = query_series(
        client, meas=meas_state, entity=relay_entity,
        field="value", agg="last", t0=t0, t1=t1, fill="null", dtype="bool",
    )

    explicit_unavailable = unavail_series.reindex(full_idx).fillna(False).astype(bool)
    known = relay_bool.reindex(full_idx).notna() | explicit_unavailable

    for meas, entity in presence_sources or []:
        if not entity:
            continue
        aux_series = query_series(
            client, meas=meas, entity=entity,
            field="value", agg="mean", t0=t0, t1=t1, fill="null", dtype="float",
        )
        if aux_series.empty:
            continue
        aux_known = aux_series.reindex(full_idx).ffill(limit=_HDO_PRESENCE_FFILL_MIN).notna()
        known = known | aux_known

    # Per-ISO-week consistent gap analysis at 1-minute resolution.
    # HDO physically cuts Shelly mains power → relay + all aux sensors lose data
    # simultaneously, identical to a WiFi outage.  WiFi outages are random;
    # HDO repeats at the same weekday×minute-of-day every week.
    # Detection: find bounded absent intervals of _MIN_GAP.._MAX_GAP minutes;
    # for each gap minute record which ISO week it fell in.
    # Slots absent in ≥ MIN_CONFIDENCE_TO_BLOCK fraction of total weeks = HDO.
    from collections import defaultdict
    from smartboiler.hdo_learner import MIN_WEEKS_TO_TRUST, MIN_CONFIDENCE_TO_BLOCK

    _MIN_GAP = 5    # minutes — shorter = measurement noise
    _MAX_GAP = 600  # minutes — Czech HDO overnight blocks run up to ~8 h (480 min);
                    # gaps > 10 h are true power / ISP outages and filtered by the
                    # ISO-week consistency check anyway (random outages < 70% of weeks)

    known_arr = known.reindex(full_idx).fillna(False).astype(bool).values
    n_ts = len(full_idx)

    # Collect distinct ISO weeks present in the full dataset
    all_iso_weeks: set = set()
    for ts in full_idx:
        iso = ts.isocalendar()
        all_iso_weeks.add((int(iso[0]), int(iso[1])))
    n_total_weeks = len(all_iso_weeks)

    # Single pass: find bounded absent gaps of _MIN_GAP.._MAX_GAP minutes and
    # record which (weekday, minute_of_day) × ISO-week combinations fall inside.
    absent_weeks: dict = defaultdict(set)  # (wd, min_of_day) -> {(yr, wk), ...}

    gap_start = None
    for i in range(n_ts):
        if not bool(known_arr[i]):
            if gap_start is None:
                gap_start = i
        else:
            if gap_start is not None:
                gap_len = i - gap_start
                bounded_before = gap_start > 0 and bool(known_arr[gap_start - 1])
                if bounded_before and _MIN_GAP <= gap_len <= _MAX_GAP:
                    for j in range(gap_start, i):
                        ts_j = full_idx[j]
                        iso_j = ts_j.isocalendar()
                        yw = (int(iso_j[0]), int(iso_j[1]))
                        slot_key = (ts_j.weekday(), ts_j.hour * 60 + ts_j.minute)
                        absent_weeks[slot_key].add(yw)
                gap_start = None

    # HDO slots: consistently absent in >= MIN_CONFIDENCE_TO_BLOCK fraction of weeks
    hdo_slots = {
        slot_key
        for slot_key, wks in absent_weeks.items()
        if n_total_weeks > 0 and len(wks) / n_total_weeks >= MIN_CONFIDENCE_TO_BLOCK
    }

    n_hdo_hours = len({(wd, min_d // 60) for wd, min_d in hdo_slots})
    logger.info(
        "InfluxDB bootstrap: %d consistent HDO minute-slots (~%d hours/week) "
        "detected from %d weeks of history",
        len(hdo_slots), n_hdo_hours, n_total_weeks,
    )

    # Inject synthetic recent observations for HDO slots only (1-min resolution).
    # Non-HDO slots get no observations → correctly not blocked by default.
    now_naive = pd.Timestamp.now()  # local time, matches full_idx timestamps
    n_inject = max(MIN_WEEKS_TO_TRUST + 1, getattr(hdo_learner, "history_weeks", 3))

    n = 0
    for wd, min_of_day in hdo_slots:
        hour   = min_of_day // 60
        minute = min_of_day % 60
        for weeks_ago in range(1, n_inject + 1):
            ref = now_naive - pd.Timedelta(weeks=weeks_ago)
            days_to_target = (wd - ref.weekday()) % 7
            target_day = ref + pd.Timedelta(days=days_to_target)
            dt = target_day.to_pydatetime().replace(
                hour=hour, minute=minute, second=0, microsecond=0
            )
            try:
                hdo_learner.observe(dt, relay_unavailable=True)
                n += 1
            except Exception:
                continue

    return n
