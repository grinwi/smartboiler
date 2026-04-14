# Created as a part of Master's Thesis "Using machine learning methods to save energy in a smart home"
# Faculty of Information Technology, Brno University of Technology, 2024
# Author: Adam Grünwald
#
# Main controller — dual-workflow architecture:
#   ForecastWorkflow: runs hourly — collects data, updates predictor, plans day
#   ControlWorkflow:  runs every 60s — executes heating plan, observes HDO
#
# Extracted helpers:
#   temperature_estimator.py — multi-level water temp estimation
#   legionella_protector.py  — periodic legionella heating cycle

import logging
import os
import sys
import threading
import time
from collections import deque
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

HEATING_TIMEOUT_MINUTES = 180
CONTROL_LOOP_INTERVAL_S = 60
FORECAST_LOOP_INTERVAL_S = 3600     # 1 hour
DATA_COLLECT_LOOKBACK_H = 6
MIN_TRAINING_DAYS_DEFAULT = 30

# Thermostat-trip debounce: don't record a second calibration within this window
_CALIB_DEBOUNCE_S = 5 * 60   # 5 min
_THERMOSTAT_TRIP_CONFIRM_S = 60.0
_CASE_STABLE_WINDOW_S = 5 * 60.0
_THERMOSTAT_TRIP_POWER_W = 50.0       # effective zero-power threshold
_THERMOSTAT_TRIP_HOLD_MAX_S = 15 * 60.0
_CASE_RISE_TOLERANCE_C = 0.1
_HDO_UNAVAILABLE_STATES = {"unavailable", "unknown"}


def _is_hdo_unavailable_state(state: Optional[str]) -> bool:
    return str(state or "").strip().lower() in _HDO_UNAVAILABLE_STATES


class SmartBoilerController:
    """
    Orchestrates data collection, prediction, scheduling, and boiler control.

    ForecastWorkflow (hourly):
      1. Collect new entity data from HA into local cache
      2. Update rolling histogram predictor
      3. Fetch spot electricity prices
      4. Run greedy heating scheduler → 24h heating plan
      5. Persist state

    ControlWorkflow (every 60s):
      1. Read current boiler temperature (via TemperatureEstimator)
      2. Check legionella protection (via LegionellaProtector)
      3. Execute heating plan (turn on/off relay)
      4. Observe relay + power for HDO learning
    """

    def __init__(self, options: Dict):
        # ── Imports (deferred so module loads fast) ───────────────────────
        from smartboiler.ha_client import HAClient
        from smartboiler.state_store import StateStore
        from smartboiler.ha_data_collector import HADataCollector
        from smartboiler.predictor import RollingHistogramPredictor
        from smartboiler.scheduler import HeatingScheduler, BoilerParams
        from smartboiler.spot_price import SpotPriceFetcher
        from smartboiler.thermal_model import ThermalModel
        from smartboiler.temperature_estimator import TemperatureEstimator
        from smartboiler.legionella_protector import LegionellaProtector
        from smartboiler.web_server import (
            set_state_provider,
            set_extra_provider,
            set_calendar_manager,
            set_influx_bootstrap_handlers,
            run_dashboard,
        )

        self._run_dashboard = run_dashboard
        self._set_influx_bootstrap_handlers = set_influx_bootstrap_handlers
        self._options = dict(options)

        # ── Config ────────────────────────────────────────────────────────
        self.boiler_switch_entity_id = options["boiler_switch_entity_id"]
        self.boiler_power_entity_id = options.get("boiler_power_entity_id") or None
        self.boiler_water_flow_entity_id = options.get("boiler_water_flow_entity_id") or None
        self.boiler_water_temp_entity_id = options.get("boiler_water_temp_entity_id") or None
        self.boiler_case_tmp_entity_id   = options.get("boiler_case_tmp_entity_id") or None
        self.boiler_inlet_tmp_entity_id  = options.get("boiler_inlet_tmp_entity_id") or None
        self.boiler_outlet_tmp_entity_id = options.get("boiler_outlet_tmp_entity_id") or None
        self.boiler_area_tmp_entity_id   = options.get("boiler_area_tmp_entity_id") or None
        self.boiler_direct_tmp_entity_id = options.get("boiler_direct_tmp_entity_id") or None
        self.pv_surplus_entity_id = options.get("pv_surplus_entity_id") or None

        # PV / Solar (FVE)
        self.has_pv = bool(options.get("has_pv", False))
        self.pv_installed_power_kw = float(options.get("pv_installed_power_kw", 0.0))
        self.pv_power_entity_id = options.get("pv_power_entity_id") or None
        self.pv_forecast_entity_id = options.get("pv_forecast_entity_id") or None

        # Battery
        self.has_battery = bool(options.get("has_battery", False))
        self.battery_capacity_kwh = float(options.get("battery_capacity_kwh", 0.0))
        self.battery_soc_entity_id = options.get("battery_soc_entity_id") or None
        self.battery_soc_unit = options.get("battery_soc_unit", "percent")
        self.battery_max_charge_kw = float(options.get("battery_max_charge_kw", 0.0))
        self.battery_priority = options.get("battery_priority", "battery_first")

        self.person_entity_ids: List[str] = options.get("person_entity_ids") or []

        self.calendar_entity_id = options.get("calendar_entity_id") or ""
        self.vacation_mode = options.get("vacation_mode", "min_temp")
        self.vacation_min_tmp = float(options.get("vacation_min_temp", 30))

        self.boiler_volume_l = float(options.get("boiler_volume", 120))
        self.boiler_set_tmp = float(options.get("boiler_set_tmp", 60))
        self.boiler_min_tmp = float(options.get("boiler_min_operation_tmp", 37))
        self.boiler_watt = float(options.get("boiler_watt_power", 2000))
        self.area_tmp = float(options.get("average_boiler_surroundings_temp", 20))

        self.thermal_window_days = float(options.get("thermal_model_window_days", 7.0))
        self.thermal_mass_ratio  = float(options.get("thermal_mass_ratio", 0.3))

        self.operation_mode         = options.get("operation_mode", "standard")
        self.cold_water_temp        = float(options.get("cold_water_temp", 10.0))
        self.thermal_coupling       = float(options.get("thermal_coupling_ratio", 0.45))
        self.boiler_standby_w       = float(options.get("boiler_standby_watts", 50.0))
        self.draw_detection_thr_c   = float(options.get("draw_detection_threshold_c", 2.0))

        self.has_spot_price = bool(options.get("has_spot_price", False))
        self.spot_price_region = options.get("spot_price_region", "CZ")
        self.hdo_explicit_schedule = options.get("hdo_explicit_schedule", "") or ""
        self.hdo_history_weeks = int(options.get("hdo_history_weeks", 3))
        self.hdo_decay_weeks = int(options.get("hdo_decay_weeks", 2))
        self.prediction_conservatism = options.get("prediction_conservatism", "medium")
        self.min_training_days = int(options.get("min_training_days", MIN_TRAINING_DAYS_DEFAULT))

        data_path = os.getenv("DATA_PATH", "/data")

        # ── Core components ───────────────────────────────────────────────
        supervisor_token = os.getenv("SUPERVISOR_TOKEN", "")
        self.ha = HAClient(token=supervisor_token)
        self.store = StateStore(data_dir=data_path)

        self.collector = HADataCollector(
            ha=self.ha,
            store=self.store,
            relay_entity_id=self.boiler_switch_entity_id,
            relay_power_entity_id=self.boiler_power_entity_id,
            water_flow_entity_id=self.boiler_water_flow_entity_id,
            water_temp_out_entity_id=self.boiler_water_temp_entity_id,
            boiler_case_tmp_entity_id=self.boiler_case_tmp_entity_id,
            boiler_volume_l=self.boiler_volume_l,
            boiler_set_tmp=self.boiler_set_tmp,
        )

        self.predictor = RollingHistogramPredictor(conservatism=self.prediction_conservatism)
        saved_predictor = self.store.load_pickle("predictor")
        if saved_predictor is not None:
            self.predictor = saved_predictor
            logger.info("Predictor restored from disk.")

        self.hdo_learner = self._new_hdo_learner()
        saved_hdo = self.store.load_pickle("hdo_learner")
        if saved_hdo is not None:
            self.hdo_learner = saved_hdo
            self.hdo_learner.decay_weeks = self.hdo_decay_weeks
            self.hdo_learner.history_weeks = self.hdo_history_weeks
            self.hdo_learner.set_explicit_schedule(self.hdo_explicit_schedule)
            logger.info("HDO learner restored from disk.")

        boiler_params = BoilerParams(
            capacity_l=self.boiler_volume_l,
            wattage_w=self.boiler_watt,
            set_tmp=self.boiler_set_tmp,
            min_tmp=self.boiler_min_tmp,
            area_tmp=self.area_tmp,
            standby_loss_w=self.boiler_standby_w,
            battery_capacity_kwh=self.battery_capacity_kwh if self.has_battery else 0.0,
            battery_soc_kwh=0.0,           # refreshed before each plan from live entity
            battery_max_charge_kw=self.battery_max_charge_kw,
            battery_priority=self.battery_priority,
        )
        self.scheduler = HeatingScheduler(boiler_params)

        self.spot_fetcher = SpotPriceFetcher(country=self.spot_price_region) if self.has_spot_price else None

        # Calendar manager (optional)
        self.calendar: Optional[Any] = None
        if self.calendar_entity_id:
            from smartboiler.calendar_manager import CalendarManager
            self.calendar = CalendarManager(
                ha_client=self.ha,
                calendar_entity_id=self.calendar_entity_id,
                vacation_mode=self.vacation_mode,
            )
            logger.info("Calendar integration enabled: %s", self.calendar_entity_id)

        # Thermal model (learned Newton's-law cooling)
        self.thermal_model = ThermalModel(
            window_days=self.thermal_window_days,
            mass_ratio=self.thermal_mass_ratio,
        )
        saved_thermal = self.store.load_pickle("thermal_model")
        if saved_thermal is not None:
            self.thermal_model = saved_thermal
            self.thermal_model.window_days = self.thermal_window_days
            self.thermal_model.mass_ratio  = self.thermal_mass_ratio
            logger.info("Thermal model restored from disk. %s", self.thermal_model.diagnostics())

        # ── Extracted helpers ─────────────────────────────────────────────
        self.temp_estimator = TemperatureEstimator(
            ha=self.ha,
            switch_entity_id=self.boiler_switch_entity_id,
            power_entity_id=self.boiler_power_entity_id,
            case_tmp_entity_id=self.boiler_case_tmp_entity_id,
            area_tmp_entity_id=self.boiler_area_tmp_entity_id,
            direct_tmp_entity_id=self.boiler_direct_tmp_entity_id,
            thermal_model=self.thermal_model,
            boiler_set_tmp=self.boiler_set_tmp,
            area_tmp_default=self.area_tmp,
        )

        self.legionella = LegionellaProtector(
            store=self.store,
            ha=self.ha,
            switch_entity_id=self.boiler_switch_entity_id,
        )

        # ── Simple Mode: flowless consumption estimator ───────────────────
        self.flow_estimator = None
        if self.operation_mode == "simple":
            from smartboiler.consumption_estimator import FlowlessConsumptionEstimator
            self.flow_estimator = FlowlessConsumptionEstimator(
                vol_L            = self.boiler_volume_l,
                relay_w          = self.boiler_watt,
                T_cold           = self.cold_water_temp,
                coupling         = self.thermal_coupling,
                T_amb_default    = self.area_tmp,
                standby_w        = self.boiler_standby_w,
                draw_threshold_c = self.draw_detection_thr_c,
            )
            saved_fe = self.store.load_pickle("flow_estimator")
            if saved_fe is not None:
                self.flow_estimator = saved_fe
                # Always keep physical params in sync with config
                self.flow_estimator.vol_L         = self.boiler_volume_l
                self.flow_estimator.relay_w       = self.boiler_watt
                self.flow_estimator.T_cold        = self.cold_water_temp
                self.flow_estimator.coupling      = self.thermal_coupling
                self.flow_estimator.T_amb_default = self.area_tmp
                self.flow_estimator.standby_w     = self.boiler_standby_w
                logger.info("FlowlessConsumptionEstimator restored. %s", self.flow_estimator.diagnostics())
            else:
                logger.info("FlowlessConsumptionEstimator initialised (no saved state).")

        # ── Mutable state ─────────────────────────────────────────────────
        self._heating_plan: List[bool] = [False] * 24
        self._plan_slots: List = []
        self._forecast_24h: List[float] = [0.0] * 24
        self._spot_prices: Dict[int, Optional[float]] = {}
        self._last_boiler_tmp: Optional[float] = self.store.get_last_boiler_tmp()
        last_tmp_updated_at = self.store.get_last_boiler_tmp_updated_at()
        self._last_boiler_tmp_updated_at: Optional[datetime] = (
            last_tmp_updated_at
            if isinstance(last_tmp_updated_at, datetime)
            else None
        )
        self._plan_generated_at: Optional[datetime] = None
        self._last_calib_ts: float = 0.0
        self._pending_trip_started_at: Optional[float] = None
        self._prev_relay_on: Optional[bool] = None
        self._prev_power_w: Optional[float] = None
        self._case_tmp_history = deque()
        self._lock = threading.Lock()
        self._influx_bootstrap_lock = threading.Lock()
        self._influx_bootstrap_status: Dict[str, Any] = {
            "available": True,
            "configured": False,
            "running": False,
            "source": "",
            "last_started_at": "",
            "last_finished_at": "",
            "last_error": "",
            "last_summary": {},
        }

        # ── Web dashboard ─────────────────────────────────────────────────
        set_state_provider(self._get_dashboard_state)
        set_extra_provider(self._get_extra_data)
        if self.calendar:
            set_calendar_manager(self.calendar)
        self._set_influx_bootstrap_handlers(
            self.start_influx_bootstrap,
            self.get_influx_bootstrap_status,
        )

    # ── Forecast workflow (hourly) ────────────────────────────────────────

    def run_forecast_workflow(self) -> None:
        """Collect data, update predictor, compute heating plan."""
        logger.info("=== Forecast workflow starting ===")
        try:
            # 1. Collect new data
            df_history = self.collector.collect_and_update(
                lookback_hours=DATA_COLLECT_LOOKBACK_H
            )

            # 2. Update predictor
            self.predictor.update(df_history)
            self.store.save_pickle("predictor", self.predictor)

            # 2b. Simple Mode: finalize yesterday's daily estimate on day rollover
            if self.flow_estimator is not None:
                k_water = self.thermal_model.k_water if self.thermal_model.is_fitted else None
                amb = self.temp_estimator.get_ambient_tmp()
                est_vol = self.flow_estimator.maybe_finalize(k_water=k_water, T_amb=amb)
                if est_vol is not None:
                    self._push_simple_mode_estimate_to_history(est_vol)
                    self.store.save_pickle("flow_estimator", self.flow_estimator)

            # 3. Fetch spot prices
            if self.spot_fetcher:
                try:
                    prices_data = self.spot_fetcher.fetch_today_tomorrow()
                    self.store.set_spot_cache(prices_data)
                    with self._lock:
                        self._spot_prices = prices_data.get("today", {})
                except Exception as e:
                    logger.warning("Spot price fetch failed: %s", e)

            # 4. Get PV surplus forecast (24 zeros if no sensor / PV disabled)
            pv_forecast = self._get_pv_forecast_24h()

            # 5. Refresh battery SoC from live HA entity so the scheduler uses
            #    the current charge level, not the stale value from __init__.
            if self.has_battery and self.battery_soc_entity_id:
                self.scheduler.boiler.battery_soc_kwh = self._get_battery_soc_kwh()

            # 6. Get HDO blocked hours
            hdo_blocked = self.hdo_learner.get_blocked_hours_next_24h()

            # 7. Get consumption forecast
            forecast = self.predictor.predict_next_24h()

            # 8. Get current boiler temperature
            boiler_tmp = self.temp_estimator.get_boiler_tmp(self._last_boiler_tmp) or self.boiler_min_tmp

            # 9. Fetch upcoming calendar events
            now_dt = datetime.now().astimezone()
            calendar_events = (
                self.calendar.get_events(now_dt, now_dt + timedelta(hours=24))
                if self.calendar else []
            )

            # 10. Run scheduler
            with self._lock:
                self._spot_prices_indexed = {
                    i: self._spot_prices.get((now_dt.hour + i) % 24)
                    for i in range(24)
                }
                self._heating_plan, self._plan_slots = self.scheduler.plan(
                    current_tmp=boiler_tmp,
                    consumption_forecast=forecast,
                    pv_forecast=pv_forecast,
                    spot_prices=self._spot_prices_indexed,
                    hdo_blocked=hdo_blocked,
                    calendar_events=calendar_events,
                    vacation_min_tmp=self.vacation_min_tmp,
                )
                self._forecast_24h = forecast
                self._plan_generated_at = datetime.now().astimezone()

            # 10. Persist plan, HDO learner, and thermal model
            plan_serializable = [bool(h) for h in self._heating_plan]
            self.store.set_heating_plan(plan_serializable)
            self.store.save_pickle("hdo_learner", self.hdo_learner)

            self.thermal_model.maybe_refit()
            self.store.save_pickle("thermal_model", self.thermal_model)
            logger.debug("Thermal model: %s", self.thermal_model.diagnostics())

            logger.info(
                "Plan: %d heating hours out of 24; predictor_ready=%s",
                sum(self._heating_plan),
                self.predictor.has_enough_data(self.min_training_days),
            )
        except Exception as e:
            logger.error("Forecast workflow error: %s", e, exc_info=True)

    def _get_pv_forecast_24h(self) -> List[float]:
        """Get PV surplus for next 24 hours.

        Source priority (first non-zero wins):
          1. pv_forecast_entity_id  — a dedicated forecast sensor (kWh/h, e.g. Solcast)
          2. pv_power_entity_id     — current PV production (W), applied as flat 24h proxy
          3. pv_surplus_entity_id   — legacy surplus sensor (W), same flat proxy

        When has_pv is False, returns all zeros regardless of entity configuration.
        The flat-24h proxy is intentionally conservative: it spreads current production
        across 24 hours; the scheduler will heat during those hours for free, but overnight
        the value will be re-read as zero once darkness falls.
        """
        if not self.has_pv:
            return [0.0] * 24

        # 1. Forecast entity (kWh per hour value — no unit conversion needed)
        if self.pv_forecast_entity_id:
            val = self.ha.get_state_value(self.pv_forecast_entity_id, default=None)
            if val is not None:
                kwh = max(0.0, float(val))
                # Sanity check: some integrations report in Wh instead of kWh
                if self.pv_installed_power_kw > 0 and kwh > self.pv_installed_power_kw * 24:
                    kwh /= 1000.0
                logger.debug("PV forecast from entity: %.3f kWh/h", kwh)
                return [kwh] * 24

        # 2. Current power sensor (W → kWh/h)
        for entity_id, label in (
            (self.pv_power_entity_id, "pv_power"),
            (self.pv_surplus_entity_id, "pv_surplus"),
        ):
            if entity_id:
                val = self.ha.get_state_value(entity_id, default=None)
                if val is not None:
                    surplus = max(0.0, float(val) / 1000.0)
                    logger.debug("PV forecast from %s: %.3f kWh/h", label, surplus)
                    return [surplus] * 24

        return [0.0] * 24

    def _get_battery_soc_kwh(self) -> float:
        """Return current battery state of charge in kWh.

        Reads battery_soc_entity_id and converts if needed:
          percent: soc_kwh = battery_capacity_kwh * value / 100
          kwh:     soc_kwh = value  (direct reading)
        """
        if not self.battery_soc_entity_id:
            return 0.0
        val = self.ha.get_state_value(self.battery_soc_entity_id, default=None)
        if val is None:
            return 0.0
        try:
            v = float(val)
        except (ValueError, TypeError):
            return 0.0
        if self.battery_soc_unit == "percent":
            soc = self.battery_capacity_kwh * max(0.0, min(100.0, v)) / 100.0
        else:
            soc = max(0.0, v)
        logger.debug("Battery SoC: %.2f kWh (raw=%.1f %s)", soc, v, self.battery_soc_unit)
        return soc

    # ── Control workflow (every 60s) ──────────────────────────────────────

    def _get_boiler_tmp(self) -> float:
        """Return best available boiler temperature (all estimation levels)."""
        tmp = self.temp_estimator.get_boiler_tmp(self._last_boiler_tmp)
        if tmp is not None:
            self._last_boiler_tmp = tmp
            self._last_boiler_tmp_updated_at = datetime.now().astimezone()
            self.store.set_last_boiler_tmp(tmp, updated_at=self._last_boiler_tmp_updated_at)
        return self._last_boiler_tmp or self.boiler_min_tmp

    def _get_temperature_estimation_data(self) -> Dict:
        """Return a rich diagnostics payload for the web estimator view."""
        report = self.temp_estimator.get_boiler_tmp_report(
            self._last_boiler_tmp,
            last_known_updated_at=self._last_boiler_tmp_updated_at,
        )
        report["operation_mode"] = self.operation_mode
        report["boiler_min_tmp"] = self.boiler_min_tmp
        report["boiler_set_tmp"] = self.boiler_set_tmp
        report["updated_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
        return report

    def _reset_pending_trip(self) -> None:
        """Forget the currently tracked thermostat-trip calibration candidate."""
        self._pending_trip_started_at = None

    @staticmethod
    def _is_thermostat_trip_low_power(relay_on: bool, power_w: Optional[float]) -> bool:
        """Return True when the relay is on but heater draw is effectively zero."""
        return (
            relay_on
            and power_w is not None
            and float(power_w) < _THERMOSTAT_TRIP_POWER_W
        )

    def _record_case_tmp_sample(self, case_tmp: Optional[float], now_ts: float) -> None:
        """Keep a short rolling history of case temperature for stability checks."""
        if case_tmp is None:
            return
        self._case_tmp_history.append((now_ts, float(case_tmp)))
        cutoff = now_ts - _CASE_STABLE_WINDOW_S
        while self._case_tmp_history and self._case_tmp_history[0][0] < cutoff:
            self._case_tmp_history.popleft()

    def _case_tmp_stable_for_calibration(self, now_ts: float) -> bool:
        """
        Return True when the case temperature has not increased in the last 5 min.

        We require full 5-minute coverage and reject any upward step above the
        noise tolerance. This avoids calibrating too early while the casing is
        still soaking heat after the thermostat has opened.
        """
        if not self._case_tmp_history:
            return False

        cutoff = now_ts - _CASE_STABLE_WINDOW_S
        recent = [(ts, tmp) for ts, tmp in self._case_tmp_history if ts >= cutoff]
        if len(recent) < 2:
            return False
        if recent[0][0] > cutoff:
            return False

        prev_tmp = recent[0][1]
        for _ts, tmp in recent[1:]:
            if tmp > prev_tmp + _CASE_RISE_TOLERANCE_C:
                return False
            prev_tmp = tmp
        return True

    def _track_thermostat_trip_candidate(
        self,
        *,
        relay_on: bool,
        power_w: Optional[float],
        case_tmp: Optional[float],
        amb: Optional[float],
        now_ts: float,
    ) -> None:
        """
        Track a thermostat-trip calibration candidate from a heater cut-off event.

        The candidate starts only when the heater was drawing power and then,
        with the relay still ON, the measured power drops to ~0 W. Calibration
        is committed only after:
          - the zero-power state lasted at least 1 minute, and
          - the case temperature has not increased in the last 5 minutes.

        The calibration uses the current temperatures at that stable moment.
        """
        low_power = self._is_thermostat_trip_low_power(relay_on, power_w)
        prev_heating = (
            self._prev_relay_on is True
            and self._prev_power_w is not None
            and float(self._prev_power_w) >= _THERMOSTAT_TRIP_POWER_W
        )
        transition_to_zero = prev_heating and low_power

        if transition_to_zero:
            self._pending_trip_started_at = now_ts
            logger.debug(
                "Thermal calibration candidate started: relay still ON, power dropped from %.1f W to %.1f W.",
                float(self._prev_power_w),
                float(power_w),
            )

        if not low_power:
            self._reset_pending_trip()
            return

        if self._pending_trip_started_at is None:
            return

        zero_power_duration_s = now_ts - self._pending_trip_started_at
        if zero_power_duration_s > _THERMOSTAT_TRIP_HOLD_MAX_S:
            logger.info(
                "Thermal calibration candidate expired after %.0fs without a stable case plateau.",
                zero_power_duration_s,
            )
            self._reset_pending_trip()
            return

        stable_case = self._case_tmp_stable_for_calibration(now_ts)
        debounced = (now_ts - self._last_calib_ts) > _CALIB_DEBOUNCE_S

        if (
            zero_power_duration_s >= _THERMOSTAT_TRIP_CONFIRM_S
            and stable_case
            and debounced
            and case_tmp is not None
            and amb is not None
        ):
            self.thermal_model.observe_calibration(
                T_set=self.boiler_set_tmp,
                T_case=float(case_tmp),
                T_amb=float(amb),
                timestamp=now_ts,
            )
            self._last_calib_ts = now_ts
            logger.info(
                "Thermal calibration confirmed after %.0fs of zero-power hold with current stable case temp %.1f°C (T_amb=%.1f°C).",
                zero_power_duration_s,
                float(case_tmp),
                float(amb),
            )
            self._reset_pending_trip()

    def _should_hold_relay_for_trip_calibration(
        self,
        *,
        relay_on: bool,
        power_w: Optional[float],
    ) -> bool:
        """
        Keep the relay ON while waiting for post-trip calibration confirmation.

        Without this hold, the controller would switch the relay off
        immediately after the thermostat opens and the required 1-minute
        zero-power window could never be observed reliably in production.
        """
        return (
            self._pending_trip_started_at is not None
            and self._is_thermostat_trip_low_power(relay_on, power_w)
        )

    def run_control_workflow(self) -> None:
        """Execute heating plan; perform legionella check; observe HDO."""
        try:
            boiler_tmp = self._get_boiler_tmp()

            # Read raw relay state once — drives both HDO learning and thermal model.
            relay_state_obj = self.ha.get_state(self.boiler_switch_entity_id)
            relay_state_str = relay_state_obj["state"] if relay_state_obj else None
            relay_on = relay_state_str == "on"
            relay_unavailable = _is_hdo_unavailable_state(relay_state_str)

            power_w: Optional[float] = None
            if self.boiler_power_entity_id:
                power_raw = self.ha.get_state_value(self.boiler_power_entity_id, default=None)
                if power_raw is not None:
                    try:
                        power_w = float(power_raw)
                    except (TypeError, ValueError):
                        power_w = None

            # HDO observation
            self.hdo_learner.observe(datetime.now().astimezone(), relay_unavailable)

            # ── Thermal model observations ────────────────────────────────
            now_ts = time.time()
            if self.boiler_case_tmp_entity_id:
                amb = self.temp_estimator.get_ambient_tmp()
                case_tmp_raw = self.ha.get_state_value(self.boiler_case_tmp_entity_id)
                case_tmp = None
                if case_tmp_raw is not None:
                    case_tmp = float(case_tmp_raw)
                    self._record_case_tmp_sample(case_tmp, now_ts)

                self._track_thermostat_trip_candidate(
                    relay_on=relay_on,
                    power_w=power_w,
                    case_tmp=case_tmp,
                    amb=amb,
                    now_ts=now_ts,
                )

                # Passive-cooling sample (only when relay is OFF)
                if case_tmp is not None and not relay_on:
                    self.thermal_model.observe_case_tmp(case_tmp, amb, timestamp=now_ts)
            else:
                self._reset_pending_trip()

            hold_for_trip_calibration = self._should_hold_relay_for_trip_calibration(
                relay_on=relay_on,
                power_w=power_w,
            )
            # Keep a one-tick memory of an ON+low-power thermostat trip so the
            # controller can release the relay cleanly even if heater draw
            # resumes before the 60 s calibration window completes.
            recent_thermostat_trip = (
                relay_on
                and self._prev_relay_on is True
                and self._prev_power_w is not None
                and float(self._prev_power_w) < _THERMOSTAT_TRIP_POWER_W
            )
            self._prev_relay_on = relay_on
            self._prev_power_w = power_w

            # ── Simple Mode: tick the flowless estimator ──────────────────
            if self.flow_estimator is not None:
                case_tmp_for_est = None
                if self.boiler_case_tmp_entity_id:
                    case_tmp_for_est = self.ha.get_state_value(self.boiler_case_tmp_entity_id)
                inlet_tmp_for_est = None
                if self.boiler_inlet_tmp_entity_id:
                    inlet_tmp_for_est = self.ha.get_state_value(self.boiler_inlet_tmp_entity_id)
                amb_for_est = self.temp_estimator.get_ambient_tmp()
                self.flow_estimator.tick(
                    relay_on  = relay_on,
                    T_case    = float(case_tmp_for_est) if case_tmp_for_est is not None else None,
                    T_amb     = amb_for_est,
                    dt_s      = float(CONTROL_LOOP_INTERVAL_S),
                    T_in      = float(inlet_tmp_for_est) if inlet_tmp_for_est is not None else None,
                )

            # Freeze protection (always takes priority)
            if boiler_tmp < 5.0:
                logger.warning("Freeze protection: turning boiler ON (tmp=%.1f)", boiler_tmp)
                self.ha.turn_on(self.boiler_switch_entity_id)
                return

            # Legionella protection
            if self.legionella.check_and_act(boiler_tmp):
                if hold_for_trip_calibration:
                    logger.debug(
                        "Holding relay ON after thermostat cut-off so calibration can be confirmed."
                    )
                    self.ha.turn_on(self.boiler_switch_entity_id)
                return

            if hold_for_trip_calibration:
                logger.debug(
                    "Holding relay ON during post-trip calibration window (power %.1f W).",
                    float(power_w) if power_w is not None else -1.0,
                )
                self.ha.turn_on(self.boiler_switch_entity_id)
                return

            # ── Calendar event override ───────────────────────────────────
            if self.calendar:
                now_tz = datetime.now().astimezone()
                active_evt = self.calendar.get_active_event(now_tz)
                if active_evt:
                    etype = active_evt.event_type
                    if etype == "vacation_off":
                        logger.info("Calendar [%s]: boiler OFF (%s)", etype, active_evt.summary)
                        self.ha.turn_off(self.boiler_switch_entity_id)
                        return
                    elif etype == "vacation_min":
                        if boiler_tmp < self.vacation_min_tmp:
                            logger.info("Calendar [%s]: heating to min %.1f°C", etype, self.vacation_min_tmp)
                            self.ha.turn_on(self.boiler_switch_entity_id)
                        else:
                            self.ha.turn_off(self.boiler_switch_entity_id)
                        return
                    elif etype in ("boost_max", "boost_temp"):
                        target = (
                            active_evt.target_temp
                            if etype == "boost_temp" and active_evt.target_temp
                            else self.boiler_set_tmp
                        )
                        if boiler_tmp < target:
                            logger.info("Calendar [%s]: heating to %.1f°C", etype, target)
                            self.ha.turn_on(self.boiler_switch_entity_id)
                        else:
                            self.ha.turn_off(self.boiler_switch_entity_id)
                        return

            # Execute current heating plan
            current_hour_idx = self._current_plan_hour_index()
            with self._lock:
                should_heat = (
                    self._heating_plan[current_hour_idx]
                    if self._heating_plan and current_hour_idx < len(self._heating_plan)
                    else False
                )

            # Override: if below min_tmp, heat regardless of plan
            if boiler_tmp < self.boiler_min_tmp:
                logger.info(
                    "Below min_tmp (%.1f < %.1f): forcing heat ON",
                    boiler_tmp, self.boiler_min_tmp,
                )
                should_heat = True

            # Don't keep relay ON once the boiler reached its target temperature.
            # Two independent signals for "boiler is fully charged":
            #   1. Temperature sensor at or above set_tmp
            #   2. Thermostat trip: relay is already ON but power is effectively 0 W
            #      (the internal thermostat cut the element; the sensor may lag
            #       by ~0.5–1 °C relative to the actual water temperature)
            thermostat_tripped = self._is_thermostat_trip_low_power(relay_on, power_w)
            if should_heat and (
                boiler_tmp >= self.boiler_set_tmp
                or thermostat_tripped
                or recent_thermostat_trip
            ):
                logger.debug(
                    "Boiler fully charged (tmp=%.1f, tripped=%s, recent_trip=%s): relay OFF",
                    boiler_tmp, thermostat_tripped, recent_thermostat_trip,
                )
                should_heat = False

            if should_heat:
                self.ha.turn_on(self.boiler_switch_entity_id)
                logger.debug("Boiler ON (plan hour %d, tmp=%.1f)", current_hour_idx, boiler_tmp)
            else:
                self.ha.turn_off(self.boiler_switch_entity_id)
                logger.debug("Boiler OFF (plan hour %d, tmp=%.1f)", current_hour_idx, boiler_tmp)

        except Exception as e:
            logger.error("Control workflow error: %s", e, exc_info=True)

    def _current_plan_hour_index(self) -> int:
        """Return the index into the 24h plan for the current time."""
        if self._plan_generated_at is None:
            return 0
        elapsed_h = int((datetime.now().astimezone() - self._plan_generated_at).total_seconds() // 3600)
        return min(elapsed_h, 23)

    # ── Dashboard state assembly ──────────────────────────────────────────

    def _get_dashboard_state(self) -> Dict:
        """Assemble current state dict for web dashboard."""
        with self._lock:
            plan_copy = list(self._heating_plan)
            slots_copy = list(self._plan_slots)
            forecast_copy = list(self._forecast_24h)
            spot_copy = dict(self._spot_prices)

        boiler_tmp = self._last_boiler_tmp
        relay_on = self.ha.is_entity_on(self.boiler_switch_entity_id)
        last_leg = self.store.get_last_legionella_heating()

        if relay_on is None:
            boiler_status = "unavailable"
        elif not relay_on:
            boiler_status = "off"
        else:
            power_w = 0.0
            if self.boiler_power_entity_id:
                power_w = self.ha.get_state_value(self.boiler_power_entity_id, default=0.0) or 0.0
            boiler_status = "on_heating" if float(power_w) >= 50 else "on_idle"

        plan_slots_json = []
        if slots_copy:
            for i, slot in enumerate(slots_copy):
                plan_slots_json.append({
                    "label": slot.dt.strftime("%H:00"),
                    "heating": bool(plan_copy[i]) if i < len(plan_copy) else False,
                    "hdo_blocked": bool(slot.hdo_blocked),
                    "pv_free": bool(slot.pv_surplus_kwh > 0.1),
                    "calendar_mode": slot.calendar_mode,
                })

        available_entities = [
            {"entity_id": s.get("entity_id", ""),
             "friendly_name": s.get("attributes", {}).get("friendly_name", "")}
            for s in self.ha.get_all_states()
        ]

        simple_mode_diag = (
            self.flow_estimator.diagnostics()
            if self.flow_estimator is not None
            else None
        )

        return {
            "boiler_temp": round(boiler_tmp, 1) if boiler_tmp is not None else None,
            "relay_on": relay_on,
            "boiler_status": boiler_status,
            "set_tmp": self.boiler_set_tmp,
            "min_tmp": self.boiler_min_tmp,
            "operation_mode": self.operation_mode,
            "simple_mode": simple_mode_diag,
            "heating_until": None,
            "last_legionella": last_leg.strftime("%Y-%m-%d") if last_leg else None,
            "predictor_has_data": bool(self.predictor.has_enough_data(self.min_training_days)),
            "forecast_24h": [round(v, 4) for v in forecast_copy],
            "plan_slots": plan_slots_json,
            "spot_prices_today": {str(k): round(v, 2) for k, v in spot_copy.items()},
            "hdo_schedule": self.hdo_learner.get_weekly_schedule(),
            "available_entities": available_entities,
            "calendar_events": self.calendar.upcoming_events_json() if self.calendar else [],
            "calendar_entity_id": self.calendar_entity_id,
            "sys_info": [
                ["Boiler volume", f"{int(self.boiler_volume_l)} L"],
                ["Boiler power", f"{int(self.boiler_watt)} W"],
                ["Operation mode", self.operation_mode],
                ["Conservatism", self.prediction_conservatism],
                ["HDO observations", str(self.hdo_learner.observation_count())],
                ["History rows", str(len(self.store.load_consumption_history()))],
                ["Spot prices", self.spot_price_region if self.has_spot_price else "disabled"],
                ["Plan generated", self._plan_generated_at.strftime("%H:%M") if self._plan_generated_at else "—"],
            ],
        }

    # ── Extra data providers (history / accuracy / predictor) ─────────────

    def _get_extra_data(self, endpoint: str, params: Dict) -> Dict:
        if endpoint == "history":
            return self._get_history_data(params.get("period", "7d"))
        if endpoint == "accuracy":
            return self._get_accuracy_data()
        if endpoint == "predictor":
            return self._get_predictor_data()
        if endpoint == "temperature_estimation":
            return self._get_temperature_estimation_data()
        return {}

    def _get_history_data(self, period: str = "7d") -> Dict:
        import pandas as pd

        empty: Dict = {"labels": [], "consumption": [], "relay_on": [], "power_w": [],
                       "daily_stats": [], "period": period}
        df = self.store.load_consumption_history()
        if df.empty:
            return empty

        days_map = {"1d": 1, "7d": 7, "30d": 30, "90d": 90}
        days = days_map.get(period, 7)
        cutoff = pd.Timestamp.now() - pd.Timedelta(days=days)
        df_f = df[df.index >= cutoff].copy()
        if df_f.empty:
            return empty

        for col in ("relay_on", "power_w"):
            if col not in df_f.columns:
                df_f[col] = 0.0
        df_f["relay_on"] = df_f["relay_on"].astype(float)
        df_f["consumed_kwh"] = df_f["consumed_kwh"].clip(lower=0.0)
        df_f = df_f.fillna(0.0)

        freq = "1h" if days <= 7 else ("3h" if days <= 30 else "1D")
        label_fmt = "%H:%M" if days <= 1 else ("%m-%d %H:00" if days <= 30 else "%m-%d")

        agg = df_f.resample(freq).agg(
            {"consumed_kwh": "sum", "relay_on": "mean", "power_w": "mean"}
        ).fillna(0.0)

        daily = df_f.resample("1D").agg(
            {"consumed_kwh": "sum", "relay_on": "sum", "power_w": "mean"}
        ).fillna(0.0)

        daily_stats = [
            {
                "date": ts.strftime("%Y-%m-%d"),
                "consumption_kwh": round(float(row["consumed_kwh"]), 3),
                "heating_hours": int(round(float(row["relay_on"]))),
                "avg_power_w": round(float(row["power_w"]), 0),
            }
            for ts, row in daily.iterrows()
        ]

        return {
            "labels":       [ts.strftime(label_fmt) for ts in agg.index],
            "consumption":  [round(float(v), 4) for v in agg["consumed_kwh"]],
            "relay_on":     [round(float(v), 3) for v in agg["relay_on"]],
            "power_w":      [round(float(v), 1) for v in agg["power_w"]],
            "daily_stats":  daily_stats,
            "period":       period,
        }

    def _get_accuracy_data(self) -> Dict:
        import numpy as np

        df = self.store.load_consumption_history()
        if df.empty or "consumed_kwh" not in df.columns:
            return {"by_hour": [], "mae": None, "bias": None}

        by_hour = []
        all_errors = []
        for hour in range(24):
            mask = df.index.hour == hour
            actual_vals = df.loc[mask, "consumed_kwh"].clip(lower=0).dropna().tolist()
            if not actual_vals:
                continue
            weekdays = df.loc[mask].index.weekday.unique().tolist()
            pred_mean = float(np.mean(
                [self.predictor.predict_hour(int(wd), hour) for wd in weekdays]
            )) if weekdays else 0.0
            act_mean = float(np.mean(actual_vals))
            all_errors.append(pred_mean - act_mean)
            by_hour.append({
                "hour":        hour,
                "predicted":   round(pred_mean, 4),
                "actual_mean": round(act_mean, 4),
                "count":       len(actual_vals),
            })

        if all_errors:
            mae  = round(float(np.mean([abs(e) for e in all_errors])), 4)
            bias = round(float(np.mean(all_errors)), 4)
        else:
            mae = bias = None
        return {"by_hour": by_hour, "mae": mae, "bias": bias}

    def _get_predictor_data(self) -> Dict:
        day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        summary = self.predictor.get_histogram_summary()
        grid: Dict = {}
        for wd_idx, day in enumerate(day_names):
            grid[day] = {}
            for hour in range(24):
                pred = self.predictor.predict_hour(wd_idx, hour)
                slot = summary.get(day, {}).get(hour, {})
                grid[day][str(hour)] = {
                    "predicted": round(pred, 4),
                    "p50":       slot.get("p50", 0.0),
                    "p75":       slot.get("p75", 0.0),
                    "count":     slot.get("count", 0),
                }
        return {
            "heatmap":         grid,
            "quantile":        self.predictor.quantile,
            "total_samples":   self.predictor._total_samples,
            "global_fallback": round(self.predictor._global_fallback, 4),
        }

    # ── Simple Mode helpers ───────────────────────────────────────────────

    def _push_simple_mode_estimate_to_history(self, est_vol_L: float) -> None:
        """
        Convert a daily volume estimate (Simple Mode) into synthetic hourly rows
        and append to consumption history so the predictor can learn from them.
        The energy is spread uniformly over 24 hours (best approximation without
        time-of-use data).
        """
        import pandas as pd

        T_set = (
            self.flow_estimator.T_set_calibrated
            if self.flow_estimator and self.flow_estimator.T_set_calibrated
            else self.boiler_set_tmp
        )
        T_cold    = self.cold_water_temp
        dT_useful = max(T_set - T_cold, 2.0)
        est_kwh   = est_vol_L * 1.0 * 4186.0 * dT_useful / 3.6e6
        hourly_kwh = est_kwh / 24.0

        from datetime import timedelta
        yesterday = datetime.now().astimezone() - timedelta(days=1)
        midnight  = yesterday.replace(hour=0, minute=0, second=0, microsecond=0)
        idx = pd.date_range(midnight, periods=24, freq="1h", tz=midnight.tzinfo)
        df_synthetic = pd.DataFrame(
            {"consumed_kwh": [hourly_kwh] * 24, "relay_on": [0.0] * 24, "power_w": [0.0] * 24},
            index=idx,
        )
        self.store.append_consumption(df_synthetic)
        logger.info(
            "Simple Mode: pushed %.1fL (%.3f kWh, %.5f kWh/h) to consumption history",
            est_vol_L, est_kwh, hourly_kwh,
        )

    # ── HDO bootstrap from HA history ─────────────────────────────────────

    def _new_hdo_learner(self):
        from smartboiler.hdo_learner import HDOLearner

        learner = HDOLearner(
            decay_weeks=self.hdo_decay_weeks,
            history_weeks=self.hdo_history_weeks,
        )
        learner.set_explicit_schedule(self.hdo_explicit_schedule)
        return learner

    def _bootstrap_hdo_from_ha(self) -> None:
        """Seed HDO learner from HA REST API relay history on fresh startup."""
        if self.hdo_learner.observation_count() > 0:
            return
        start = datetime.now().astimezone() - timedelta(weeks=self.hdo_history_weeks)
        logger.info(
            "Bootstrapping HDO learner from HA history (last %d weeks)…",
            self.hdo_history_weeks,
        )
        try:
            history = self.ha.get_history(self.boiler_switch_entity_id, start)
        except Exception as e:
            logger.warning("HA HDO bootstrap failed to fetch history: %s", e)
            return
        if not history:
            logger.info("HA HDO bootstrap: no relay history returned.")
            return
        # Parse all state transitions into (dt, is_unavailable) pairs
        parsed: List[tuple] = []
        for entry in history:
            state_str = entry.get("state", "")
            last_changed = entry.get("last_changed") or entry.get("last_updated")
            if not last_changed:
                continue
            try:
                dt = datetime.fromisoformat(last_changed.replace("Z", "+00:00")).astimezone()
            except Exception:
                continue
            parsed.append((dt, _is_hdo_unavailable_state(state_str)))

        # Fill every 5-minute slot within each "unavailable" period.
        # HA history only records state *transitions*, so if HDO is active
        # 13:05–13:35 only two observations would be written (the start and
        # end transitions).  Slots in between would have zero observations and
        # would never cross MIN_WEEKS_TO_TRUST.  We synthesise one observation
        # per 5-minute slot for the full duration of each unavailable span.
        from smartboiler.hdo_learner import SLOT_MINUTES
        slot_seconds = SLOT_MINUTES * 60
        count = 0
        for i, (dt, is_unavail) in enumerate(parsed):
            if is_unavail:
                # Determine how long this unavailable period lasted
                end_dt = parsed[i + 1][0] if i + 1 < len(parsed) else datetime.now().astimezone()
                duration_s = (end_dt - dt).total_seconds()
                # Emit one observation per 5-min slot for the entire span
                cursor = 0.0
                while cursor < max(duration_s, slot_seconds):
                    slot_dt = dt + timedelta(seconds=cursor)
                    self.hdo_learner.observe(slot_dt, True)
                    count += 1
                    cursor += slot_seconds
            else:
                # Available — record one observation at the transition point only
                self.hdo_learner.observe(dt, False)
                count += 1

        self.store.save_pickle("hdo_learner", self.hdo_learner)
        logger.info(
            "HDO learner seeded from HA history: %d observations from %d transitions.",
            count, len(parsed),
        )

    # ── InfluxDB bootstrap helpers ───────────────────────────────────────

    def _refresh_after_influx_bootstrap(self) -> Dict[str, Any]:
        """Refresh predictor/dashboard state from the newly imported history."""
        df_history = self.store.load_consumption_history()
        self.predictor.update(df_history)
        self.store.save_pickle("predictor", self.predictor)

        forecast = self.predictor.predict_next_24h()
        with self._lock:
            self._forecast_24h = forecast

        info = {
            "predictor_total_samples": getattr(self.predictor, "_total_samples", 0),
            "predictor_ready": bool(self.predictor.has_enough_data(self.min_training_days)),
            "hdo_total_observations": int(self.hdo_learner.observation_count()),
            "hdo_weekly_blocked_hours": int(
                sum(len(hours) for hours in self.hdo_learner.get_weekly_schedule().values())
            ),
        }
        if info["hdo_total_observations"] > 0 and info["hdo_weekly_blocked_hours"] == 0:
            logger.info(
                "Post-bootstrap HDO state: observations imported, but no blocked hours "
                "met the trust/confidence threshold. This usually means the Influx relay "
                "history has no 'unavailable' periods or the pattern is too weak."
            )
        logger.info("Post-bootstrap refresh complete: %s", info)
        return info

    def _build_influx_bootstrapper(self, options_override: Optional[Dict] = None):
        from smartboiler.influx_bootstrap import InfluxBootstrapper

        options = dict(self._options)
        if options_override:
            options.update(options_override)

        self.thermal_model.window_days = float(
            options.get("thermal_model_window_days", self.thermal_window_days)
        )
        self.thermal_model.mass_ratio = float(
            options.get("thermal_mass_ratio", self.thermal_mass_ratio)
        )

        return InfluxBootstrapper(
            options=options,
            store=self.store,
            thermal_model=self.thermal_model,
            boiler_set_tmp=float(options.get("boiler_set_tmp", self.boiler_set_tmp)),
            boiler_min_tmp=float(options.get("boiler_min_operation_tmp", self.boiler_min_tmp)),
            boiler_watt=float(options.get("boiler_watt_power", self.boiler_watt)),
            boiler_volume=float(options.get("boiler_volume", self.boiler_volume_l)),
            area_tmp=float(options.get("average_boiler_surroundings_temp", self.area_tmp)),
            cold_water_tmp=float(options.get("cold_water_temp", self.cold_water_temp)),
            operation_mode=options.get("operation_mode", self.operation_mode),
            coupling=float(options.get("thermal_coupling_ratio", self.thermal_coupling)),
            standby_w=float(options.get("boiler_standby_watts", self.boiler_standby_w)),
        )

    def get_influx_bootstrap_status(self) -> Dict[str, Any]:
        with self._influx_bootstrap_lock:
            status = dict(self._influx_bootstrap_status)
        status["configured"] = self._build_influx_bootstrapper().is_configured()
        return status

    def start_influx_bootstrap(
        self,
        options_override: Optional[Dict] = None,
        source: str = "manual",
    ) -> Dict[str, Any]:
        options = dict(self._options)
        if options_override:
            options.update(options_override)
            self._options = dict(options)

        bootstrapper = self._build_influx_bootstrapper(options)
        configured = bootstrapper.is_configured()
        if not configured:
            status = self.get_influx_bootstrap_status()
            status["configured"] = False
            return {
                "ok": False,
                "started": False,
                "available": True,
                "error": "InfluxDB bootstrap is not configured for the active mode.",
                "status": status,
            }

        with self._influx_bootstrap_lock:
            if self._influx_bootstrap_status["running"]:
                status = dict(self._influx_bootstrap_status)
                status["configured"] = configured
                return {
                    "ok": False,
                    "started": False,
                    "available": True,
                    "error": "InfluxDB bootstrap is already running.",
                    "status": status,
                }

            started_at = datetime.now().isoformat()
            self._influx_bootstrap_status.update({
                "available": True,
                "configured": configured,
                "running": True,
                "source": source,
                "last_started_at": started_at,
                "last_finished_at": "",
                "last_error": "",
                "last_summary": {},
            })

        def _run_bootstrap() -> None:
            logger.info("InfluxDB bootstrap (%s) starting in background…", source)
            try:
                seeded_hdo = self._new_hdo_learner()
                summary = bootstrapper.run(hdo_learner=seeded_hdo)
                self.hdo_learner = seeded_hdo
                self.store.save_pickle("thermal_model", self.thermal_model)
                self.store.save_pickle("hdo_learner", self.hdo_learner)
                summary.update(self._refresh_after_influx_bootstrap())
                logger.info("InfluxDB bootstrap (%s) done: %s", source, summary)
                with self._influx_bootstrap_lock:
                    self._influx_bootstrap_status.update({
                        "available": True,
                        "configured": True,
                        "running": False,
                        "source": source,
                        "last_finished_at": datetime.now().isoformat(),
                        "last_error": "",
                        "last_summary": summary,
                    })
            except Exception as exc:
                logger.error("InfluxDB bootstrap (%s) failed: %s", source, exc)
                with self._influx_bootstrap_lock:
                    self._influx_bootstrap_status.update({
                        "available": True,
                        "configured": configured,
                        "running": False,
                        "source": source,
                        "last_finished_at": datetime.now().isoformat(),
                        "last_error": str(exc),
                    })

        thread = threading.Thread(
            target=_run_bootstrap,
            daemon=True,
            name=f"influx-bootstrap-{source}",
        )
        thread.start()
        return {
            "ok": True,
            "started": True,
            "available": True,
            "status": self.get_influx_bootstrap_status(),
        }

    # ── Main loop ─────────────────────────────────────────────────────────

    def start(self) -> None:
        """Start both workflows and web dashboard."""
        dash_thread = threading.Thread(
            target=self._run_dashboard,
            kwargs={"host": "0.0.0.0", "port": 8099},
            daemon=True,
            name="dashboard",
        )
        dash_thread.start()
        logger.info("Dashboard started on :8099")

        if self._build_influx_bootstrapper().should_run():
            self.start_influx_bootstrap(source="startup")

        self._bootstrap_hdo_from_ha()
        self.run_forecast_workflow()

        last_forecast_run = datetime.now().astimezone()

        logger.info("Starting control loop (interval: %ds)", CONTROL_LOOP_INTERVAL_S)
        while True:
            try:
                self.run_control_workflow()
            except Exception as e:
                logger.error("Unhandled control error: %s", e)

            if datetime.now().astimezone() - last_forecast_run >= timedelta(seconds=FORECAST_LOOP_INTERVAL_S):
                try:
                    self.run_forecast_workflow()
                    last_forecast_run = datetime.now().astimezone()
                except Exception as e:
                    logger.error("Unhandled forecast error: %s", e)

                # Periodic InfluxDB retrain check (every predictor_retrain_weeks)
                if self._build_influx_bootstrapper().should_run():
                    self.start_influx_bootstrap(source="retrain")

            time.sleep(CONTROL_LOOP_INTERVAL_S)


# ── Entry point ───────────────────────────────────────────────────────────

def _setup_logging(level_str: str) -> None:
    level = getattr(logging, level_str.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        stream=sys.stdout,
    )


def _wait_for_setup() -> Dict:
    """Start the web server and block until setup wizard completes.

    Returns the completed config dict.
    """
    import threading
    from smartboiler.web_server import run_dashboard

    logger.info(
        "Setup not complete — starting web server on :8099. "
        "Open the SmartBoiler add-on UI to complete setup."
    )

    t = threading.Thread(target=run_dashboard, daemon=True)
    t.start()

    from smartboiler.setup_config import load_setup_config, is_setup_complete
    while True:
        time.sleep(5)
        cfg = load_setup_config()
        if is_setup_complete(cfg):
            logger.info("Setup complete — starting controller.")
            return cfg


if __name__ == "__main__":
    from smartboiler.setup_config import load_setup_config, is_setup_complete

    # Basic logging before we know the configured level
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        stream=sys.stdout,
    )

    options = load_setup_config()

    if not is_setup_complete(options):
        options = _wait_for_setup()

    _setup_logging(options.get("logging_level", "INFO"))

    controller = SmartBoilerController(options)
    controller.start()
