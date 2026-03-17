# Created as a part of Master's Thesis "Using machine learning methods to save energy in a smart home"
# Faculty of Information Technology, Brno University of Technology, 2024
# Author: Adam Grünwald
#
# Thermal model seeder from InfluxDB history.
# Scans relay + power history for thermostat trips (relay ON, power < 50 W),
# feeds calibration events and subsequent case-temperature cooling curves into
# the ThermalModel, then triggers a fit.

import logging
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from smartboiler._influx_helpers import fmt_ts, query_series

logger = logging.getLogger(__name__)

# Minimum gap between two consecutive thermostat-trip calibrations [minutes]
_CALIB_MIN_GAP_MIN = 60

# How many minutes of case-temp cooling to collect after each thermostat trip
_COOLING_WINDOW_MIN = 8 * 60  # 8 hours

# Maximum forward-fill gaps
_RELAY_MAX_FFILL_MIN = 60
_TEMP_MAX_FFILL_MIN  = 120
_POWER_MAX_FFILL_MIN = 60


def seed_thermal_model(
    client,
    thermal_model,
    relay_entity: str,
    case_tmp_entity: str,
    power_entity: str,
    meas_state: str,
    meas_temp: str,
    meas_power: str,
    boiler_set_tmp: float,
    area_tmp: float,
    start: datetime,
    end: datetime,
) -> int:
    """
    Scan InfluxDB for thermostat trips and case-temp cooling curves.

    Feeds them into thermal_model.observe_calibration() / observe_case_tmp(),
    then calls thermal_model._try_fit() to compute the initial fit.

    Returns the number of calibration events recorded.
    """
    t0 = fmt_ts(start)
    t1 = fmt_ts(end)

    relay_series = query_series(
        client, meas=meas_state, entity=relay_entity,
        field="value", agg="last", t0=t0, t1=t1, fill="null", dtype="bool",
    )
    case_series = query_series(
        client, meas=meas_temp, entity=case_tmp_entity,
        field="value", agg="mean", t0=t0, t1=t1, fill="null", dtype="float",
    )

    power_series = pd.Series(dtype=float)
    if power_entity:
        power_series = query_series(
            client, meas=meas_power, entity=power_entity,
            field="value", agg="mean", t0=t0, t1=t1, fill="null", dtype="float",
        )

    if relay_series.empty or case_series.empty:
        return 0

    # Align to common 1-min index with gap-limited forward-fill
    idx = relay_series.index.union(case_series.index)
    relay = relay_series.reindex(idx).ffill(limit=_RELAY_MAX_FFILL_MIN).fillna(False)
    case  = case_series.reindex(idx).ffill(limit=_TEMP_MAX_FFILL_MIN)
    power = (
        power_series.reindex(idx).ffill(limit=_POWER_MAX_FFILL_MIN).fillna(9999.0)
        if not power_series.empty
        else pd.Series(9999.0, index=idx)
    )

    # Find thermostat trips: relay ON + power < 50 W
    # Fallback (no power sensor): relay ON → OFF transition
    if not power_series.empty:
        trip_mask = relay.astype(bool) & (power < 50.0)
    else:
        relay_b = relay.astype(bool)
        trip_mask = (~relay_b) & relay_b.shift(1, fill_value=False)

    trip_times = trip_mask[trip_mask].index
    n_calib = 0
    last_calib_ts = None

    for trip_ts in trip_times:
        ts_unix = trip_ts.timestamp()

        # Debounce: skip if within _CALIB_MIN_GAP_MIN of last recorded trip
        if last_calib_ts is not None and (ts_unix - last_calib_ts) < _CALIB_MIN_GAP_MIN * 60:
            continue

        case_at_trip = case.get(trip_ts)
        if case_at_trip is None or np.isnan(case_at_trip):
            continue

        thermal_model.observe_calibration(
            T_set=boiler_set_tmp,
            T_case=float(case_at_trip),
            T_amb=area_tmp,
            timestamp=ts_unix,
        )
        last_calib_ts = ts_unix
        n_calib += 1

        # Collect cooling samples for the next _COOLING_WINDOW_MIN (relay OFF only)
        cooling_end = trip_ts + timedelta(minutes=_COOLING_WINDOW_MIN)
        cooling_idx = case.index[
            (case.index > trip_ts) &
            (case.index <= cooling_end) &
            (~relay.astype(bool))
        ]
        last_sample_ts = 0.0
        for sample_ts in cooling_idx:
            sample_unix = sample_ts.timestamp()
            if sample_unix - last_sample_ts < 14 * 60:   # keep ~15-min spacing
                continue
            t_case_val = case.get(sample_ts)
            if t_case_val is None or np.isnan(t_case_val):
                continue
            thermal_model.observe_case_tmp(
                T_case=float(t_case_val),
                T_amb=area_tmp,
                timestamp=sample_unix,
            )
            last_sample_ts = sample_unix

    if n_calib > 0:
        thermal_model._try_fit()

    return n_calib
