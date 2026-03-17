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

import json
import logging
import os
import sys
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

HEATING_TIMEOUT_MINUTES = 180
CONTROL_LOOP_INTERVAL_S = 60
FORECAST_LOOP_INTERVAL_S = 3600     # 1 hour
DATA_COLLECT_LOOKBACK_H = 6
MIN_TRAINING_DAYS_DEFAULT = 30

# Thermostat-trip debounce: don't record a second calibration within this window
_CALIB_DEBOUNCE_S = 5 * 60   # 5 min


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
        from smartboiler.hdo_learner import HDOLearner
        from smartboiler.scheduler import HeatingScheduler, BoilerParams
        from smartboiler.spot_price import SpotPriceFetcher
        from smartboiler.thermal_model import ThermalModel
        from smartboiler.temperature_estimator import TemperatureEstimator
        from smartboiler.legionella_protector import LegionellaProtector
        from smartboiler.web_server import (
            set_state_provider, set_extra_provider, set_calendar_manager, run_dashboard,
        )

        self._run_dashboard = run_dashboard

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

        self.hdo_learner = HDOLearner()
        self.hdo_learner.set_explicit_schedule(self.hdo_explicit_schedule)
        saved_hdo = self.store.load_pickle("hdo_learner")
        if saved_hdo is not None:
            self.hdo_learner = saved_hdo
            self.hdo_learner.set_explicit_schedule(self.hdo_explicit_schedule)
            logger.info("HDO learner restored from disk.")

        boiler_params = BoilerParams(
            capacity_l=self.boiler_volume_l,
            wattage_w=self.boiler_watt,
            set_tmp=self.boiler_set_tmp,
            min_tmp=self.boiler_min_tmp,
            area_tmp=self.area_tmp,
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

        # ── InfluxDB bootstrap (periodic retrain, runs in background) ────────
        from smartboiler.influx_bootstrap import InfluxBootstrapper
        self._bootstrapper = InfluxBootstrapper(
            options=options,
            store=self.store,
            thermal_model=self.thermal_model if self.boiler_case_tmp_entity_id else None,
            boiler_set_tmp=self.boiler_set_tmp,
            boiler_min_tmp=self.boiler_min_tmp,
            boiler_watt=self.boiler_watt,
            boiler_volume=self.boiler_volume_l,
            area_tmp=self.area_tmp,
            cold_water_tmp=self.cold_water_temp,
            operation_mode=self.operation_mode,
            coupling=self.thermal_coupling,
            standby_w=self.boiler_standby_w,
        )

        # ── Mutable state ─────────────────────────────────────────────────
        self._heating_plan: List[bool] = [False] * 24
        self._plan_slots: List = []
        self._forecast_24h: List[float] = [0.0] * 24
        self._spot_prices: Dict[int, Optional[float]] = {}
        self._last_boiler_tmp: Optional[float] = self.store.get_last_boiler_tmp()
        self._plan_generated_at: Optional[datetime] = None
        self._last_calib_ts: float = 0.0
        self._lock = threading.Lock()

        # ── Web dashboard ─────────────────────────────────────────────────
        set_state_provider(self._get_dashboard_state)
        set_extra_provider(self._get_extra_data)
        if self.calendar:
            set_calendar_manager(self.calendar)

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

            # 4. Get PV surplus forecast (24 zeros if no sensor)
            pv_forecast = self._get_pv_forecast_24h()

            # 5. Get HDO blocked hours
            hdo_blocked = self.hdo_learner.get_blocked_hours_next_24h()

            # 6. Get consumption forecast
            forecast = self.predictor.predict_next_24h()

            # 7. Get current boiler temperature
            boiler_tmp = self.temp_estimator.get_boiler_tmp(self._last_boiler_tmp) or self.boiler_min_tmp

            # 8. Fetch upcoming calendar events
            now_dt = datetime.now().astimezone()
            calendar_events = (
                self.calendar.get_events(now_dt, now_dt + timedelta(hours=24))
                if self.calendar else []
            )

            # 9. Run scheduler
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
        """Get PV surplus for next 24 hours (zeroes if no sensor)."""
        if not self.pv_surplus_entity_id:
            return [0.0] * 24
        val = self.ha.get_state_value(self.pv_surplus_entity_id, default=0.0)
        surplus = max(0.0, float(val) / 1000.0)  # convert W → kWh/h approximation
        return [surplus] * 24

    # ── Control workflow (every 60s) ──────────────────────────────────────

    def _get_boiler_tmp(self) -> float:
        """Return best available boiler temperature (all estimation levels)."""
        tmp = self.temp_estimator.get_boiler_tmp(self._last_boiler_tmp)
        if tmp is not None:
            self._last_boiler_tmp = tmp
            self.store.set_last_boiler_tmp(tmp)
        return self._last_boiler_tmp or self.boiler_min_tmp

    def run_control_workflow(self) -> None:
        """Execute heating plan; perform legionella check; observe HDO."""
        try:
            boiler_tmp = self._get_boiler_tmp()

            # Read raw relay state once — drives both HDO learning and thermal model.
            relay_state_obj = self.ha.get_state(self.boiler_switch_entity_id)
            relay_state_str = relay_state_obj["state"] if relay_state_obj else None
            relay_on = relay_state_str == "on"
            relay_unavailable = relay_state_str == "unavailable"

            power_w = 0.0
            if self.boiler_power_entity_id:
                power_w = self.ha.get_state_value(self.boiler_power_entity_id, default=0.0) or 0.0

            # HDO observation
            self.hdo_learner.observe(datetime.now().astimezone(), relay_unavailable)

            # ── Thermal model observations ────────────────────────────────
            if self.boiler_case_tmp_entity_id:
                case_tmp_raw = self.ha.get_state_value(self.boiler_case_tmp_entity_id)
                if case_tmp_raw is not None:
                    case_tmp = float(case_tmp_raw)
                    amb = self.temp_estimator.get_ambient_tmp()

                    # Thermostat-trip calibration: relay ON + power ≈ 0 W
                    now_ts = time.time()
                    trip = relay_on and float(power_w) < 50
                    debounced = (now_ts - self._last_calib_ts) > _CALIB_DEBOUNCE_S
                    if trip and debounced:
                        self.thermal_model.observe_calibration(
                            T_set=self.boiler_set_tmp,
                            T_case=case_tmp,
                            T_amb=amb,
                            timestamp=now_ts,
                        )
                        self._last_calib_ts = now_ts

                    # Passive-cooling sample (only when relay is OFF)
                    if not relay_on:
                        self.thermal_model.observe_case_tmp(case_tmp, amb, timestamp=now_ts)

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

    def _bootstrap_hdo_from_ha(self) -> None:
        """Seed HDO learner from HA REST API relay history on fresh startup."""
        if self.hdo_learner.observation_count() > 0:
            return
        from smartboiler.hdo_learner import HISTORY_WEEKS
        start = datetime.now().astimezone() - timedelta(weeks=HISTORY_WEEKS)
        logger.info("Bootstrapping HDO learner from HA history (last %d weeks)…", HISTORY_WEEKS)
        try:
            history = self.ha.get_history(self.boiler_switch_entity_id, start)
        except Exception as e:
            logger.warning("HA HDO bootstrap failed to fetch history: %s", e)
            return
        if not history:
            logger.info("HA HDO bootstrap: no relay history returned.")
            return
        count = 0
        for entry in history:
            state_str = entry.get("state", "")
            last_changed = entry.get("last_changed") or entry.get("last_updated")
            if not last_changed:
                continue
            try:
                dt = datetime.fromisoformat(last_changed.replace("Z", "+00:00")).astimezone()
            except Exception:
                continue
            self.hdo_learner.observe(dt, state_str == "unavailable")
            count += 1
        self.store.save_pickle("hdo_learner", self.hdo_learner)
        logger.info("HDO learner seeded from HA history: %d state transitions ingested.", count)

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

        if self._bootstrapper.should_run():
            import threading
            def _run_bootstrap():
                logger.info("InfluxDB bootstrap starting in background…")
                try:
                    summary = self._bootstrapper.run(hdo_learner=self.hdo_learner)
                    self.store.save_pickle("thermal_model", self.thermal_model)
                    self.store.save_pickle("hdo_learner", self.hdo_learner)
                    logger.info("InfluxDB bootstrap done: %s", summary)
                except Exception as exc:
                    logger.error("InfluxDB bootstrap failed: %s", exc)
            threading.Thread(target=_run_bootstrap, daemon=True, name="influx-bootstrap").start()

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
                if self._bootstrapper.should_run():
                    import threading
                    def _retrain():
                        logger.info("InfluxDB periodic retrain starting in background…")
                        try:
                            summary = self._bootstrapper.run(hdo_learner=self.hdo_learner)
                            self.store.save_pickle("thermal_model", self.thermal_model)
                            self.store.save_pickle("hdo_learner", self.hdo_learner)
                            logger.info("InfluxDB retrain done: %s", summary)
                        except Exception as exc:
                            logger.error("InfluxDB retrain failed: %s", exc)
                    threading.Thread(target=_retrain, daemon=True, name="influx-retrain").start()

            time.sleep(CONTROL_LOOP_INTERVAL_S)


# ── Entry point ───────────────────────────────────────────────────────────

def _load_options() -> Dict:
    options_path = os.getenv("OPTIONS_PATH", "/data/options.json")
    p = Path(options_path)
    if p.exists():
        with p.open() as f:
            return json.load(f)
    logger.warning("options.json not found at %s; using defaults", options_path)
    return {}


def _setup_logging(level_str: str) -> None:
    level = getattr(logging, level_str.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        stream=sys.stdout,
    )


if __name__ == "__main__":
    options = _load_options()
    _setup_logging(options.get("logging_level", "INFO"))

    if not options.get("boiler_switch_entity_id"):
        logger.error("boiler_switch_entity_id is required in options.json")
        sys.exit(1)

    controller = SmartBoilerController(options)
    controller.start()
