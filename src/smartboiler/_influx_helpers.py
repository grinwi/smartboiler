# Created as a part of Master's Thesis "Using machine learning methods to save energy in a smart home"
# Faculty of Information Technology, Brno University of Technology, 2024
# Author: Adam Grünwald
#
# Shared InfluxDB query helpers used by influx_consumption_importer and influx_thermal_seeder.
# Not part of the public API — prefix _ signals internal use only.

import logging
from datetime import datetime

import pandas as pd

logger = logging.getLogger(__name__)


def fmt_ts(dt: datetime) -> str:
    """Format a datetime as an InfluxDB timestamp literal."""
    return f"'{dt.strftime('%Y-%m-%dT%H:%M:%SZ')}'"


def query_series(
    client,
    meas: str,
    entity: str,
    field: str,
    agg: str,
    t0: str,
    t1: str,
    fill: str = "null",
    dtype: str = "float",
) -> pd.Series:
    """
    Execute a 1-minute grouped InfluxDB query for a single entity and return
    a pandas Series with a tz-naive DatetimeIndex.

    Args:
        client:  influxdb.DataFrameClient (already connected)
        meas:    InfluxDB measurement name (e.g. "°C", "state")
        entity:  entity_id tag value
        field:   field name (typically "value")
        agg:     aggregation function ("last", "mean", …)
        t0, t1:  RFC3339 timestamp literals as returned by fmt_ts()
        fill:    FILL() strategy ("null", "0", "previous", …)
        dtype:   output type — "float", "bool", or "unavailable"
    """
    if not entity:
        return pd.Series(dtype=float)

    safe_entity = entity.replace("'", "\\'")
    q = (
        f'SELECT {agg}("{field}") AS "v" '
        f'FROM "{meas}" '
        f'WHERE time > {t0} AND time <= {t1} '
        f'AND "entity_id"=\'{safe_entity}\' '
        f'GROUP BY time(1m) FILL({fill})'
    )
    try:
        result = client.query(q)
    except Exception as e:
        logger.debug("InfluxDB query failed (%s / %s): %s", meas, entity, e)
        return pd.Series(dtype=float)

    if not result or meas not in result:
        return pd.Series(dtype=float)

    df = result[meas]
    if df.empty or "v" not in df.columns:
        return pd.Series(dtype=float)

    series = df["v"]
    series.index = pd.to_datetime(series.index, utc=True).tz_localize(None)

    if dtype == "bool":
        series = series.map(
            lambda x: True if str(x).lower() in ("on", "true", "1", "1.0") else
                      (False if str(x).lower() in ("off", "false", "0", "0.0") else None)
        )
    elif dtype == "unavailable":
        series = series.map(lambda x: str(x).lower() in ("unavailable", "unknown"))
    else:
        series = pd.to_numeric(series, errors="coerce")

    return series
