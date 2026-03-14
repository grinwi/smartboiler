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
# 1. Consumption history  — delegates to influx_consumption_importer.py
# 2. Thermal model seeding — delegates to influx_thermal_seeder.py
# 3. HDO learner seeding  — delegates to influx_consumption_importer.py
#
# Run condition
# ─────────────
# Runs once on startup when consumption_history has fewer than
# `min_training_days × 24` rows, or if the "influx_bootstrap_done" flag is
# absent from state.json.  Subsequent startups skip the bootstrap.

import logging
from datetime import datetime, timedelta

from smartboiler._influx_helpers import fmt_ts  # noqa: F401 — re-exported for tests
from smartboiler.influx_consumption_importer import fetch_consumption_chunk, seed_hdo_learner
from smartboiler.influx_thermal_seeder import seed_thermal_model

import pandas as pd

logger = logging.getLogger(__name__)

_CHUNK_DAYS = 30  # query window size to avoid loading everything into RAM at once


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

        # ── InfluxDB connection params ─────────────────────────────────────
        self.host     = options.get("influxdb_host", "") or ""
        self.port     = int(options.get("influxdb_port", 8086))
        self.database = options.get("influxdb_db", "") or ""
        self.username = options.get("influxdb_username", "") or ""
        self.password = options.get("influxdb_password", "") or ""

        # ── Entity IDs as stored in InfluxDB ─────────────────────────────
        self.relay_entity     = options.get("influxdb_relay_entity_id", "") or ""
        self.flow_entity      = options.get("influxdb_flow_entity_id", "") or ""
        self.water_tmp_entity = options.get("influxdb_water_temp_entity_id", "") or ""
        self.case_tmp_entity  = options.get("influxdb_case_tmp_entity_id", "") or ""
        self.power_entity     = options.get("influxdb_power_entity_id", "") or ""

        # ── Measurement names (HA InfluxDB integration defaults) ──────────
        self.meas_temp  = options.get("influxdb_measurement_temp",  "°C")
        self.meas_flow  = options.get("influxdb_measurement_flow",  "L/min")
        self.meas_power = options.get("influxdb_measurement_power", "W")
        self.meas_state = options.get("influxdb_measurement_state", "state")

        # ── Import window ─────────────────────────────────────────────────
        years_back = int(options.get("influxdb_history_years", 2))
        start_str  = options.get("influxdb_start_date", "") or ""
        if start_str:
            try:
                self.start_date = datetime.fromisoformat(start_str)
            except ValueError:
                logger.warning(
                    "influxdb_start_date '%s' is not ISO format; using %d years back",
                    start_str, years_back,
                )
                self.start_date = datetime.now() - timedelta(days=365 * years_back)
        else:
            self.start_date = datetime.now() - timedelta(days=365 * years_back)

        self._client = None  # lazy-initialised DataFrameClient

    # ── Public API ────────────────────────────────────────────────────────

    def is_configured(self) -> bool:
        """Returns True only if the minimum required options are present."""
        return bool(self.host and self.database and self.relay_entity)

    def should_run(self) -> bool:
        """
        Returns True if a bootstrap pass is needed.
        Skips if the flag 'influx_bootstrap_done' is set in state.json,
        or if the consumption history already has >= 30 days of data.
        """
        if not self.is_configured():
            return False
        if self.store.get("influx_bootstrap_done"):
            return False
        df = self.store.load_consumption_history()
        min_rows = 30 * 24
        if len(df) >= min_rows:
            logger.info("InfluxDB bootstrap skipped: already have %d hourly rows", len(df))
            self.store.set("influx_bootstrap_done", True)
            return False
        return True

    def run(self, hdo_learner=None) -> dict:
        """
        Run the full bootstrap.  Returns a summary dict with counters.
        Marks 'influx_bootstrap_done' in state.json when complete.

        Args:
            hdo_learner: Optional HDOLearner instance to seed from relay history.
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
        summary = {"consumption_hours": 0, "calib_events": 0, "hdo_observations": 0}

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
                df_chunk = fetch_consumption_chunk(
                    client=self._client,
                    relay_entity=self.relay_entity,
                    flow_entity=self.flow_entity,
                    water_tmp_entity=self.water_tmp_entity,
                    power_entity=self.power_entity,
                    cold_water_tmp=self.cold_water_tmp,
                    boiler_watt=self.boiler_watt,
                    meas_state=self.meas_state,
                    meas_flow=self.meas_flow,
                    meas_temp=self.meas_temp,
                    meas_power=self.meas_power,
                    start=chunk_start,
                    end=chunk_end,
                )
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
                n = seed_thermal_model(
                    client=self._client,
                    thermal_model=self.thermal_model,
                    relay_entity=self.relay_entity,
                    case_tmp_entity=self.case_tmp_entity,
                    power_entity=self.power_entity,
                    meas_state=self.meas_state,
                    meas_temp=self.meas_temp,
                    meas_power=self.meas_power,
                    boiler_set_tmp=self.boiler_set_tmp,
                    area_tmp=self.area_tmp,
                    start=self.start_date,
                    end=end,
                )
                summary["calib_events"] = n
                logger.info("InfluxDB bootstrap: seeded thermal model with %d calibration events", n)
            except Exception as e:
                logger.warning("InfluxDB bootstrap: thermal seeding failed: %s", e)

        # ── 3. HDO learner seeding ─────────────────────────────────────────
        if hdo_learner is not None and self.relay_entity:
            try:
                n = seed_hdo_learner(
                    client=self._client,
                    relay_entity=self.relay_entity,
                    meas_state=self.meas_state,
                    hdo_learner=hdo_learner,
                    start=self.start_date,
                    end=end,
                )
                summary["hdo_observations"] = n
                logger.info("InfluxDB bootstrap: seeded HDO learner with %d observations", n)
            except Exception as e:
                logger.warning("InfluxDB bootstrap: HDO seeding failed: %s", e)

        self.store.set("influx_bootstrap_done", True)
        logger.info("InfluxDB bootstrap complete: %s", summary)
        return summary

    # ── InfluxDB connection ────────────────────────────────────────────────

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
        self._client.ping()
        logger.info("InfluxDB bootstrap: connected to %s:%d/%s", self.host, self.port, self.database)
