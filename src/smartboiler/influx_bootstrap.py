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
from smartboiler.influx_consumption_importer import (
    fetch_consumption_chunk,
    fetch_consumption_chunk_simple_mode,
    seed_hdo_learner,
)
from smartboiler.influx_thermal_seeder import seed_thermal_model

import pandas as pd

logger = logging.getLogger(__name__)

_CHUNK_DAYS = 30  # query window size to avoid loading everything into RAM at once


def _resolve_entity(
    options: dict,
    operation_mode: str,
    legacy_key: str,
    standard_key: str = "",
    simple_key: str = "",
) -> str:
    if operation_mode == "simple" and simple_key:
        value = options.get(simple_key, "") or ""
        if value:
            return value
    if operation_mode == "standard" and standard_key:
        value = options.get(standard_key, "") or ""
        if value:
            return value
    return options.get(legacy_key, "") or ""


class InfluxBootstrapper:
    """
    Periodic import of historical InfluxDB data into the local cache.

    Runs on first startup and then every shared retrain interval
    (the shorter of `predictor_retrain_weeks` and `hdo_retrain_weeks`),
    so the predictor stays up-to-date without manual intervention.

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
        boiler_volume: float = 120.0,
        area_tmp: float = 20.0,
        cold_water_tmp: float = 10.0,
        operation_mode: str = "standard",
        coupling: float = 0.45,
        standby_w: float = 50.0,
    ):
        self.store = store
        self.thermal_model = thermal_model
        self.boiler_set_tmp = boiler_set_tmp
        self.boiler_min_tmp = boiler_min_tmp
        self.boiler_watt = boiler_watt
        self.boiler_volume = boiler_volume
        self.area_tmp = area_tmp
        self.cold_water_tmp = cold_water_tmp
        cfg_operation_mode = options.get("operation_mode", "") or ""
        self.operation_mode = (
            cfg_operation_mode
            if cfg_operation_mode in ("simple", "standard")
            else operation_mode
        )
        self.coupling = coupling
        self.standby_w = standby_w

        self._options = options

        # ── InfluxDB connection params ─────────────────────────────────────
        self.host     = options.get("influxdb_host", "") or ""
        self.port     = int(options.get("influxdb_port", 8086))
        self.database = options.get("influxdb_db", "") or ""
        self.username = options.get("influxdb_username", "") or ""
        self.password = options.get("influxdb_password", "") or ""

        # ── Entity IDs as stored in InfluxDB ─────────────────────────────
        self.relay_entity      = _resolve_entity(
            options, self.operation_mode, "influxdb_relay_entity_id",
            standard_key="influxdb_standard_relay_entity_id",
            simple_key="influxdb_simple_relay_entity_id",
        )
        self.flow_entity       = _resolve_entity(
            options, self.operation_mode, "influxdb_flow_entity_id",
            standard_key="influxdb_standard_flow_entity_id",
        )
        self.water_tmp_entity  = _resolve_entity(
            options, self.operation_mode, "influxdb_water_temp_entity_id",
            standard_key="influxdb_standard_water_temp_entity_id",
        )
        self.case_tmp_entity   = _resolve_entity(
            options, self.operation_mode, "influxdb_case_tmp_entity_id",
            standard_key="influxdb_standard_case_tmp_entity_id",
            simple_key="influxdb_simple_case_tmp_entity_id",
        )
        self.inlet_tmp_entity  = _resolve_entity(
            options, self.operation_mode, "influxdb_inlet_tmp_entity_id",
            simple_key="influxdb_simple_inlet_tmp_entity_id",
        )
        self.outlet_tmp_entity = _resolve_entity(
            options, self.operation_mode, "influxdb_water_temp_entity_id",
            simple_key="influxdb_simple_outlet_tmp_entity_id",
        )
        self.power_entity      = _resolve_entity(
            options, self.operation_mode, "influxdb_power_entity_id",
            standard_key="influxdb_standard_power_entity_id",
            simple_key="influxdb_simple_power_entity_id",
        )

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

        # ── Periodic retrain interval ──────────────────────────────────────
        predictor_retrain_weeks = int(options.get("predictor_retrain_weeks", 4))
        self.hdo_retrain_weeks = int(options.get("hdo_retrain_weeks", 3))
        self._retrain_weeks = min(predictor_retrain_weeks, self.hdo_retrain_weeks)
        self.hdo_history_weeks = int(options.get("hdo_history_weeks", 3))

        self._client = None  # lazy-initialised DataFrameClient

    # ── Public API ────────────────────────────────────────────────────────

    def is_configured(self) -> bool:
        """Returns True only if the minimum required options are present."""
        return bool(self.host and self.database and self.relay_entity)

    def should_run(self) -> bool:
        """
        Returns True if a bootstrap/retrain pass is needed.

        Runs on first startup (no timestamp stored) and then every
        the configured shared retrain interval so imported models stay current.
        The old boolean `influx_bootstrap_done` flag is migrated automatically:
        if it is True and no timestamp exists the interval clock starts from now.
        """
        if not self.is_configured():
            return False

        last_str = self.store.get("influx_last_bootstrap")

        # Migrate legacy one-shot flag → timestamp-based
        if last_str is None and self.store.get("influx_bootstrap_done"):
            now_str = datetime.now().isoformat()
            self.store.set("influx_last_bootstrap", now_str)
            logger.info(
                "InfluxDB bootstrap: migrated legacy flag; next retrain in %d weeks",
                self._retrain_weeks,
            )
            return False

        if last_str is None:
            return True  # first run

        try:
            last_dt = datetime.fromisoformat(last_str)
        except (ValueError, TypeError):
            return True

        interval = timedelta(weeks=self._retrain_weeks)
        due = datetime.now() - last_dt > interval
        if due:
            logger.info(
                "InfluxDB bootstrap: retrain due (last run %s, interval %d weeks)",
                last_str[:10], self._retrain_weeks,
            )
        return due

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
        use_simple = self.operation_mode == "simple" and (
            self.case_tmp_entity or self.outlet_tmp_entity
        )
        chunk_start = self.start_date
        frames = []
        while chunk_start < end:
            chunk_end = min(chunk_start + timedelta(days=_CHUNK_DAYS), end)
            try:
                if use_simple:
                    df_chunk = fetch_consumption_chunk_simple_mode(
                        client=self._client,
                        relay_entity=self.relay_entity,
                        case_tmp_entity=self.case_tmp_entity,
                        inlet_tmp_entity=self.inlet_tmp_entity,
                        outlet_tmp_entity=self.outlet_tmp_entity,
                        power_entity=self.power_entity,
                        coupling=self.coupling,
                        T_set=self.boiler_set_tmp,
                        cold_water_tmp=self.cold_water_tmp,
                        standby_w=self.standby_w,
                        boiler_volume=self.boiler_volume,
                        meas_state=self.meas_state,
                        meas_temp=self.meas_temp,
                        meas_power=self.meas_power,
                        start=chunk_start,
                        end=chunk_end,
                    )
                else:
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
                hdo_start = max(self.start_date, end - timedelta(weeks=self.hdo_history_weeks))
                presence_sources = []
                for meas, entity in (
                    (self.meas_power, self.power_entity),
                    (self.meas_temp, self.case_tmp_entity),
                    (self.meas_temp, self.water_tmp_entity),
                    (self.meas_temp, self.inlet_tmp_entity),
                    (self.meas_temp, self.outlet_tmp_entity),
                    (self.meas_flow, self.flow_entity),
                ):
                    pair = (meas, entity)
                    if entity and pair not in presence_sources:
                        presence_sources.append(pair)
                n = seed_hdo_learner(
                    client=self._client,
                    relay_entity=self.relay_entity,
                    meas_state=self.meas_state,
                    hdo_learner=hdo_learner,
                    start=hdo_start,
                    end=end,
                    presence_sources=presence_sources,
                )
                summary["hdo_observations"] = n
                summary["hdo_history_weeks"] = self.hdo_history_weeks
                logger.info("InfluxDB bootstrap: seeded HDO learner with %d observations", n)
            except Exception as e:
                logger.warning("InfluxDB bootstrap: HDO seeding failed: %s", e)

        self.store.set("influx_last_bootstrap", datetime.now().isoformat())
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
