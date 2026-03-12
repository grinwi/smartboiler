# Created as a part of Master's Thesis "Using machine learning methods to save energy in a smart home"
# Faculty of Information Technology, Brno University of Technology, 2024
# Author: Adam Grünwald
#
# One-time historical bootstrap from InfluxDB.
#
# Problem it solves
# ─────────────────
# The HA REST API collector only fetches the last 6 hours per run, so the
# consumption predictor and thermal model need ~30 days to warm up organically.
# If the user already has months of data in InfluxDB (from HA's InfluxDB
# integration) this bootstrap imports it all in a single startup pass.
#
# What it does
# ────────────
# 1. Consumption history  — pulls water-flow + outlet-temp at 1-min resolution,
#    computes kWh per minute (Q = flow × 4.186 × ΔT / 3600), resamples to
#    hourly rows, and merges into consumption_history.pkl.
#    If no flow sensor is configured, falls back to relay-on-time with a
#    physics estimate: on-time × wattage × (set_tmp − min_tmp) / set_tmp range.
#
# 2. Thermal model seeding — scans relay + power history for thermostat trips
#    (relay='on', power < 50 W) and feeds calibration events + subsequent
#    case-temperature cooling curves into the ThermalModel.
#
# InfluxDB schema (standard HA InfluxDB integration defaults)
# ────────────────────────────────────────────────────────────
#   measurement = unit of measurement  (e.g. "°C", "W", "L/min")
#   tag entity_id                       (value stored by HA as entity_id)
#   field "value"                       (numeric readings)
#   measurement "state"                 (binary: on/off, field "value" = text)
#
# Run condition
# ─────────────
# Runs once on startup when consumption_history has fewer than
# `min_training_days × 24` rows, or if the "influx_bootstrap_done" flag is
# absent from state.json.  Subsequent startups skip the bootstrap.

import logging
import math
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Chunk size for InfluxDB queries — avoids pulling everything into RAM at once
_CHUNK_DAYS = 30

# Minimum gap between two consecutive thermostat-trip calibrations [minutes]
_CALIB_MIN_GAP_MIN = 60

# How many minutes of case-temp cooling to collect after each thermostat trip
_COOLING_WINDOW_MIN = 8 * 60  # 8 hours

# Maximum gap (in 1-minute buckets) to forward-fill after a sensor outage.
# Readings older than this many minutes are treated as unknown / NaN.
_RELAY_MAX_FFILL_MIN = 60     # relay offline > 1 h → assume unknown (→ False)
_TEMP_MAX_FFILL_MIN  = 120    # temperature offline > 2 h → NaN
_POWER_MAX_FFILL_MIN = 60     # power offline > 1 h → NaN → treated as high (9999)


class InfluxBootstrapper:
    """
    One-time import of historical InfluxDB data into the local cache.

    Typical call from the controller's __init__:
        bootstrapper = InfluxBootstrapper(options, store, thermal_model, ...)
        if bootstrapper.should_run():
            bootstrapper.run()
    """

    def __init__(
        self,
        options: dict,
        store,           # StateStore
        thermal_model,   # ThermalModel (may be None if case sensor not configured)
        boiler_set_tmp: float = 60.0,
        boiler_min_tmp: float = 37.0,
        boiler_watt: float = 2000.0,
        area_tmp: float = 20.0,
        cold_water_tmp: float = 10.0,
    ):
        self.store = store
        self.thermal_model = thermal_model
        self.boiler_set_tmp = boiler_set_tmp
        self.boiler_min_tmp = boiler_min_tmp
        self.boiler_watt = boiler_watt
        self.area_tmp = area_tmp
        self.cold_water_tmp = cold_water_tmp

        # ── InfluxDB connection ───────────────────────────────────────────
        self.host     = options.get("influxdb_host", "") or ""
        self.port     = int(options.get("influxdb_port", 8086))
        self.database = options.get("influxdb_db", "") or ""
        self.username = options.get("influxdb_username", "") or ""
        self.password = options.get("influxdb_password", "") or ""

        # ── Entity IDs as stored in InfluxDB (tag "entity_id") ───────────
        # These often match the HA entity IDs but can differ if the HA
        # InfluxDB integration is configured with a custom component_config.
        self.relay_entity    = options.get("influxdb_relay_entity_id", "") or ""
        self.flow_entity     = options.get("influxdb_flow_entity_id", "") or ""
        self.water_tmp_entity = options.get("influxdb_water_temp_entity_id", "") or ""
        self.case_tmp_entity = options.get("influxdb_case_tmp_entity_id", "") or ""
        self.power_entity    = options.get("influxdb_power_entity_id", "") or ""

        # Measurement names (HA defaults)
        self.meas_temp  = options.get("influxdb_measurement_temp",  "°C")
        self.meas_flow  = options.get("influxdb_measurement_flow",  "L/min")
        self.meas_power = options.get("influxdb_measurement_power", "W")
        self.meas_state = options.get("influxdb_measurement_state", "state")

        # How far back to import
        years_back = int(options.get("influxdb_history_years", 2))
        start_str  = options.get("influxdb_start_date", "") or ""
        if start_str:
            try:
                self.start_date = datetime.fromisoformat(start_str)
            except ValueError:
                logger.warning("influxdb_start_date '%s' is not ISO format; using %d years back",
                               start_str, years_back)
                self.start_date = datetime.now() - timedelta(days=365 * years_back)
        else:
            self.start_date = datetime.now() - timedelta(days=365 * years_back)

        self._client = None   # lazy-initialised DataFrameClient

    # ── Public API ────────────────────────────────────────────────────────

    def is_configured(self) -> bool:
        """Returns True only if the minimum required options are present."""
        return bool(self.host and self.database and self.relay_entity)

    def should_run(self) -> bool:
        """
        Returns True if a bootstrap pass is needed.
        Skips if the flag 'influx_bootstrap_done' is already set in state.json,
        or if the consumption history already has >= 30 days of data.
        """
        if not self.is_configured():
            return False
        if self.store.get("influx_bootstrap_done"):
            return False
        df = self.store.load_consumption_history()
        min_rows = 30 * 24   # 30 days of hourly rows
        if len(df) >= min_rows:
            logger.info("InfluxDB bootstrap skipped: already have %d hourly rows", len(df))
            self.store.set("influx_bootstrap_done", True)
            return False
        return True

    def run(self) -> dict:
        """
        Run the full bootstrap.  Returns a summary dict with counters.
        Marks 'influx_bootstrap_done' in state.json when complete.
        """
        if not self.is_configured():
            logger.warning("InfluxDB bootstrap: not configured, skipping.")
            return {}

        try:
            self._connect()
        except Exception as e:
            logger.error("InfluxDB bootstrap: cannot connect to %s: %s", self.host, e)
            return {}

        end = datetime.now()
        summary = {"consumption_hours": 0, "calib_events": 0}

        logger.info(
            "InfluxDB bootstrap: importing from %s to %s in %d-day chunks",
            self.start_date.date(), end.date(), _CHUNK_DAYS,
        )

        # ── 1. Consumption history ─────────────────────────────────────────
        chunk_start = self.start_date
        frames = []
        while chunk_start < end:
            chunk_end = min(chunk_start + timedelta(days=_CHUNK_DAYS), end)
            try:
                df_chunk = self._fetch_consumption_chunk(chunk_start, chunk_end)
                if not df_chunk.empty:
                    frames.append(df_chunk)
            except Exception as e:
                logger.warning("InfluxDB bootstrap: chunk %s failed: %s", chunk_start.date(), e)
            chunk_start = chunk_end

        if frames:
            df_all = pd.concat(frames)
            df_all = df_all[~df_all.index.duplicated(keep="last")].sort_index()
            df_merged = self.store.append_consumption(df_all)
            summary["consumption_hours"] = len(df_all)
            logger.info(
                "InfluxDB bootstrap: imported %d hourly rows; total now %d",
                summary["consumption_hours"], len(df_merged),
            )

        # ── 2. Thermal model seeding ──────────────────────────────────────
        if self.thermal_model is not None and self.case_tmp_entity:
            try:
                n = self._seed_thermal_model(self.start_date, end)
                summary["calib_events"] = n
                logger.info("InfluxDB bootstrap: seeded thermal model with %d calibration events", n)
            except Exception as e:
                logger.warning("InfluxDB bootstrap: thermal seeding failed: %s", e)

        self.store.set("influx_bootstrap_done", True)
        logger.info("InfluxDB bootstrap complete: %s", summary)
        return summary

    # ── Consumption import ─────────────────────────────────────────────────

    def _fetch_consumption_chunk(self, start: datetime, end: datetime) -> pd.DataFrame:
        """Pull one time chunk and return hourly consumption DataFrame."""
        t0 = _fmt(start)
        t1 = _fmt(end)

        # Use FILL(null) so InfluxDB returns NaN for missing minutes instead of
        # silently propagating a stale value across multi-day sensor outages.
        relay_series = self._query_series(
            meas=self.meas_state,
            entity=self.relay_entity,
            field="value",
            agg="last",
            t0=t0, t1=t1,
            fill="null",
            dtype="bool",
        )
        if relay_series.empty:
            return pd.DataFrame()

        # Gap-limited ffill: propagate only up to the configured window;
        # beyond that treat as unknown → safe default (False for relay, NaN for temps).
        relay_series = relay_series.ffill(limit=_RELAY_MAX_FFILL_MIN).fillna(False)

        if self.flow_entity and self.water_tmp_entity:
            flow_series = self._query_series(
                meas=self.meas_flow,
                entity=self.flow_entity,
                field="value",
                agg="mean",
                t0=t0, t1=t1,
                fill="0",       # offline flow sensor → assume 0 (no water drawn)
                dtype="float",
            )
            temp_series = self._query_series(
                meas=self.meas_temp,
                entity=self.water_tmp_entity,
                field="value",
                agg="mean",
                t0=t0, t1=t1,
                fill="null",
                dtype="float",
            )
            # Gap-limited ffill for temp; remaining NaN propagates to consumed_min
            # so those minutes are excluded from the hourly sum rather than
            # being estimated with a stale or worst-case temperature.
            temp_series = temp_series.ffill(limit=_TEMP_MAX_FFILL_MIN)
            delta_t = (temp_series - self.cold_water_tmp).clip(lower=0)
            # kWh per minute = flow_l_min * 4.186 * ΔT / 3600
            # NaN delta_t → NaN consumed_min → excluded by min_count=1 in resample
            consumed_min = (flow_series.fillna(0) * 4.186 * delta_t / 3600.0)
        else:
            # Fallback: estimate from relay on-time.
            # relay_series is already gap-filled (False for unknown) so unknown
            # periods contribute 0 kWh rather than a phantom estimate.
            consumed_min = relay_series.astype(float) * (self.boiler_watt / 60_000.0)
            logger.debug("InfluxDB bootstrap: no flow sensor; estimating from relay on-time")

        power_series = pd.Series(dtype=float)
        if self.power_entity:
            power_series = self._query_series(
                meas=self.meas_power,
                entity=self.power_entity,
                field="value",
                agg="mean",
                t0=t0, t1=t1,
                fill="0",
                dtype="float",
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
        # min_count=1: hours where all consumed_kwh minutes are NaN (sensor fully
        # offline that hour) produce NaN and are dropped rather than a phantom 0.
        df_hourly["relay_on"] = df_hourly["relay_on"] > 0.5
        return df_hourly.dropna(how="all")

    # ── Thermal model seeding ──────────────────────────────────────────────

    def _seed_thermal_model(self, start: datetime, end: datetime) -> int:
        """
        Scan InfluxDB for thermostat trips and case-temp cooling curves.
        Feeds them into thermal_model.observe_calibration() / observe_case_tmp().
        Returns the number of calibration events recorded.
        """
        t0 = _fmt(start)
        t1 = _fmt(end)

        relay_series = self._query_series(
            meas=self.meas_state,
            entity=self.relay_entity,
            field="value",
            agg="last",
            t0=t0, t1=t1,
            fill="null",
            dtype="bool",
        )
        case_series = self._query_series(
            meas=self.meas_temp,
            entity=self.case_tmp_entity,
            field="value",
            agg="mean",
            t0=t0, t1=t1,
            fill="null",
            dtype="float",
        )

        power_series = pd.Series(dtype=float)
        if self.power_entity:
            power_series = self._query_series(
                meas=self.meas_power,
                entity=self.power_entity,
                field="value",
                agg="mean",
                t0=t0, t1=t1,
                fill="null",
                dtype="float",
            )

        if relay_series.empty or case_series.empty:
            return 0

        # Align to common 1-min index; apply gap-limited ffill so a sensor
        # outage longer than the threshold produces NaN rather than stale data.
        idx = relay_series.index.union(case_series.index)
        relay = relay_series.reindex(idx).ffill(limit=_RELAY_MAX_FFILL_MIN).fillna(False)
        case  = case_series.reindex(idx).ffill(limit=_TEMP_MAX_FFILL_MIN)
        power = (
            power_series.reindex(idx).ffill(limit=_POWER_MAX_FFILL_MIN).fillna(9999.0)
            if not power_series.empty
            else pd.Series(9999.0, index=idx)
        )

        # Find thermostat trips: relay ON + power < 50 W
        # If no power series: fall back to relay ON → OFF transition
        if not power_series.empty:
            trip_mask = relay.astype(bool) & (power < 50.0)
        else:
            # relay OFF and was ON 1 minute ago  →  approximate trip
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

            self.thermal_model.observe_calibration(
                T_set=self.boiler_set_tmp,
                T_case=float(case_at_trip),
                T_amb=self.area_tmp,
                timestamp=ts_unix,
            )
            last_calib_ts = ts_unix
            n_calib += 1

            # Collect cooling samples for the next _COOLING_WINDOW_MIN
            # Only minutes where relay is OFF
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
                self.thermal_model.observe_case_tmp(
                    T_case=float(t_case_val),
                    T_amb=self.area_tmp,
                    timestamp=sample_unix,
                )
                last_sample_ts = sample_unix

        # Fit the model with all the seeded data
        if n_calib > 0:
            self.thermal_model._try_fit()

        return n_calib

    # ── InfluxDB helpers ───────────────────────────────────────────────────

    def _connect(self) -> None:
        try:
            from influxdb import DataFrameClient
        except ImportError:
            raise RuntimeError(
                "influxdb-python package not installed. "
                "Add 'influxdb' to the Dockerfile or run: pip install influxdb"
            )
        self._client = DataFrameClient(
            host=self.host,
            port=self.port,
            username=self.username,
            password=self.password,
            database=self.database,
        )
        # Smoke-test the connection
        self._client.ping()
        logger.info("InfluxDB bootstrap: connected to %s:%d/%s", self.host, self.port, self.database)

    def _query_series(
        self,
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
        Execute a 1-minute grouped InfluxDB query for a single entity,
        return a pandas Series with a DatetimeIndex.
        """
        if not entity:
            return pd.Series(dtype=float)

        # Escape single quotes in entity_id
        safe_entity = entity.replace("'", "\\'")
        q = (
            f'SELECT {agg}("{field}") AS "v" '
            f'FROM "{meas}" '
            f'WHERE time > {t0} AND time <= {t1} '
            f'AND "entity_id"=\'{safe_entity}\' '
            f'GROUP BY time(1m) FILL({fill})'
        )
        try:
            result = self._client.query(q)
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
        else:
            series = pd.to_numeric(series, errors="coerce")

        return series


def _fmt(dt: datetime) -> str:
    """Format datetime as InfluxDB timestamp literal."""
    return f"'{dt.strftime('%Y-%m-%dT%H:%M:%SZ')}'"
