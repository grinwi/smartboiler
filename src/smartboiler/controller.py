# Created as a part of Master's Thesis "Using machine learning methods to save energy in a smart home"
# Faculty of Information Technology, Brno University of Technology, 2024
# Author: Adam Grünwald
#
# Main controller — dual-workflow architecture:
#   ForecastWorkflow: runs hourly — collects data, updates predictor, plans day
#   ControlWorkflow:  runs every 60s — executes heating plan, observes HDO

import json
import logging
import os
import sys
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

LEGIONELLA_INTERVAL_DAYS = 21
HEATING_TIMEOUT_MINUTES = 180
CONTROL_LOOP_INTERVAL_S = 60
FORECAST_LOOP_INTERVAL_S = 3600     # 1 hour
DATA_COLLECT_LOOKBACK_H = 6
MIN_TRAINING_DAYS_DEFAULT = 30


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
      1. Read current boiler temperature
      2. Check legionella protection
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
        from smartboiler.web_server import set_state_provider, run_dashboard

        self._run_dashboard = run_dashboard

        # ── Config ────────────────────────────────────────────────────────
        self.boiler_switch_entity_id = options["boiler_switch_entity_id"]
        self.boiler_power_entity_id = options.get("boiler_power_entity_id") or None
        self.boiler_water_flow_entity_id = options.get("boiler_water_flow_entity_id") or None
        self.boiler_water_temp_entity_id = options.get("boiler_water_temp_entity_id") or None
        self.boiler_case_tmp_entity_id = options.get("boiler_case_tmp_entity_id") or None
        self.boiler_direct_tmp_entity_id = options.get("boiler_direct_tmp_entity_id") or None
        self.pv_surplus_entity_id = options.get("pv_surplus_entity_id") or None
        self.energy_tariff_entity_id = options.get("energy_tariff_entity_id") or None
        self.person_entity_ids: List[str] = options.get("person_entity_ids") or []

        self.boiler_volume_l = float(options.get("boiler_volume", 120))
        self.boiler_set_tmp = float(options.get("boiler_set_tmp", 60))
        self.boiler_min_tmp = float(options.get("boiler_min_operation_tmp", 37))
        self.boiler_watt = float(options.get("boiler_watt_power", 2000))
        self.area_tmp = float(options.get("average_boiler_surroundings_temp", 20))
        self.boiler_case_max_tmp = float(options.get("boiler_case_max_tmp", 40))

        self.has_spot_price = bool(options.get("has_spot_price", False))
        self.spot_price_region = options.get("spot_price_region", "CZ")
        self.has_hdo = bool(options.get("hdo", False))
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
        # Restore trained predictor from disk if available
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

        # ── Mutable state ─────────────────────────────────────────────────
        self._heating_plan: List[bool] = [False] * 24
        self._plan_slots: List = []
        self._forecast_24h: List[float] = [0.0] * 24
        self._spot_prices: Dict[int, Optional[float]] = {}
        self._last_boiler_tmp: Optional[float] = None
        self._plan_generated_at: Optional[datetime] = None
        self._lock = threading.Lock()

        # ── Web dashboard ─────────────────────────────────────────────────
        set_state_provider(self._get_dashboard_state)

    # ── Temperature estimation ────────────────────────────────────────────

    def _get_boiler_tmp(self) -> Optional[float]:
        """Three-level temperature estimation strategy."""
        # Level 1: direct water temperature sensor
        if self.boiler_direct_tmp_entity_id:
            val = self.ha.get_state_value(self.boiler_direct_tmp_entity_id)
            if val is not None:
                return float(val)

        # Level 2: infer from power feedback
        # When relay ON and power drops to ~0 → thermostat cut off at set_tmp
        # (exponential cooling model after that)
        if self.boiler_power_entity_id and self._last_boiler_tmp is not None:
            power = self.ha.get_state_value(self.boiler_power_entity_id)
            relay_on = self.ha.is_entity_on(self.boiler_switch_entity_id)
            if relay_on and power is not None and float(power) < 50:
                # Thermostat just tripped → water is at set_tmp
                return float(self.boiler_set_tmp)

        # Level 3: linear interpolation from case temperature sensor
        if self.boiler_case_tmp_entity_id:
            case_tmp = self.ha.get_state_value(self.boiler_case_tmp_entity_id)
            if case_tmp is not None:
                return self._interpolate_from_case_tmp(float(case_tmp))

        return self._last_boiler_tmp  # last known

    def _interpolate_from_case_tmp(self, case_tmp: float) -> float:
        """Linear interpolation: case_tmp → water_tmp."""
        area = self.area_tmp
        case_max = self.boiler_case_max_tmp
        set_tmp = self.boiler_set_tmp

        if case_tmp <= area or case_max <= area:
            return case_tmp
        ratio = (case_tmp - area) / (case_max - area)
        return ratio * (set_tmp - area) + area

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
            boiler_tmp = self._get_boiler_tmp() or self.boiler_min_tmp

            # 8. Run scheduler
            with self._lock:
                self._spot_prices_indexed = {
                    i: self._spot_prices.get((datetime.now().hour + i) % 24)
                    for i in range(24)
                }
                self._heating_plan, self._plan_slots = self.scheduler.plan(
                    current_tmp=boiler_tmp,
                    consumption_forecast=forecast,
                    pv_forecast=pv_forecast,
                    spot_prices=self._spot_prices_indexed,
                    hdo_blocked=hdo_blocked,
                )
                self._forecast_24h = forecast
                self._plan_generated_at = datetime.now()

            # 9. Persist plan and HDO learner
            plan_serializable = [bool(h) for h in self._heating_plan]
            self.store.set_heating_plan(plan_serializable)
            self.store.save_pickle("hdo_learner", self.hdo_learner)

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
        # Read current surplus; assume it stays constant for forecast
        val = self.ha.get_state_value(self.pv_surplus_entity_id, default=0.0)
        surplus = max(0.0, float(val) / 1000.0)  # convert W → kWh/h approximation
        return [surplus] * 24

    # ── Control workflow (every 60s) ──────────────────────────────────────

    def run_control_workflow(self) -> None:
        """Execute heating plan; perform legionella check; observe HDO."""
        try:
            boiler_tmp = self._get_boiler_tmp()
            if boiler_tmp is not None:
                self._last_boiler_tmp = boiler_tmp
            else:
                boiler_tmp = self._last_boiler_tmp or self.boiler_min_tmp

            relay_on = self.ha.is_entity_on(self.boiler_switch_entity_id) or False
            power_w = 0.0
            if self.boiler_power_entity_id:
                power_w = self.ha.get_state_value(self.boiler_power_entity_id, default=0.0) or 0.0

            # HDO observation
            self.hdo_learner.observe(datetime.now(), relay_on, float(power_w))

            # Freeze protection (always takes priority)
            if boiler_tmp < 5.0:
                logger.warning("Freeze protection: turning boiler ON (tmp=%.1f)", boiler_tmp)
                self.ha.turn_on(self.boiler_switch_entity_id)
                return

            # Legionella protection
            last_leg = self.store.get_last_legionella_heating()
            if datetime.now() - last_leg > timedelta(days=LEGIONELLA_INTERVAL_DAYS):
                logger.info("Legionella protection: heating to 65°C (tmp=%.1f)", boiler_tmp)
                self.ha.turn_on(self.boiler_switch_entity_id)
                if boiler_tmp >= 65.0:
                    self.store.set_last_legionella_heating(datetime.now())
                    self.ha.turn_off(self.boiler_switch_entity_id)
                    logger.info("Legionella protection complete.")
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
        hours_elapsed = int((datetime.now() - self._plan_generated_at).total_seconds() / 3600)
        return min(hours_elapsed, 23)

    # ── Dashboard state ───────────────────────────────────────────────────

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

        plan_slots_json = []
        if slots_copy:
            for i, slot in enumerate(slots_copy):
                plan_slots_json.append({
                    "label": slot.dt.strftime("%H:00"),
                    "heating": plan_copy[i] if i < len(plan_copy) else False,
                    "hdo_blocked": slot.hdo_blocked,
                    "pv_free": slot.pv_surplus_kwh > 0.1,
                })

        available_entities = [
            {"entity_id": s.get("entity_id", ""),
             "friendly_name": s.get("attributes", {}).get("friendly_name", "")}
            for s in self.ha.get_all_states()
        ]

        return {
            "boiler_temp": round(boiler_tmp, 1) if boiler_tmp is not None else None,
            "relay_on": relay_on,
            "set_tmp": self.boiler_set_tmp,
            "min_tmp": self.boiler_min_tmp,
            "heating_until": None,  # plan-based, no explicit deadline
            "last_legionella": last_leg.strftime("%Y-%m-%d") if last_leg else None,
            "predictor_has_data": self.predictor.has_enough_data(self.min_training_days),
            "forecast_24h": [round(v, 4) for v in forecast_copy],
            "plan_slots": plan_slots_json,
            "spot_prices_today": {str(k): round(v, 2) for k, v in spot_copy.items()},
            "hdo_schedule": self.hdo_learner.get_weekly_schedule(),
            "available_entities": available_entities,
            "sys_info": [
                ["Boiler volume", f"{int(self.boiler_volume_l)} L"],
                ["Boiler power", f"{int(self.boiler_watt)} W"],
                ["Conservatism", self.prediction_conservatism],
                ["HDO observations", str(self.hdo_learner.observation_count())],
                ["History rows", str(len(self.store.load_consumption_history()))],
                ["Spot prices", self.spot_price_region if self.has_spot_price else "disabled"],
                ["Plan generated", self._plan_generated_at.strftime("%H:%M") if self._plan_generated_at else "—"],
            ],
        }

    # ── Main loops ────────────────────────────────────────────────────────

    def start(self) -> None:
        """Start both workflows and web dashboard."""
        # Dashboard in background thread
        dash_thread = threading.Thread(
            target=self._run_dashboard,
            kwargs={"host": "0.0.0.0", "port": 8099},
            daemon=True,
            name="dashboard",
        )
        dash_thread.start()
        logger.info("Dashboard started on :8099")

        # Run initial forecast immediately
        self.run_forecast_workflow()

        last_forecast_run = datetime.now()

        logger.info("Starting control loop (interval: %ds)", CONTROL_LOOP_INTERVAL_S)
        while True:
            try:
                self.run_control_workflow()
            except Exception as e:
                logger.error("Unhandled control error: %s", e)

            # Re-run forecast every hour
            if datetime.now() - last_forecast_run >= timedelta(seconds=FORECAST_LOOP_INTERVAL_S):
                try:
                    self.run_forecast_workflow()
                    last_forecast_run = datetime.now()
                except Exception as e:
                    logger.error("Unhandled forecast error: %s", e)

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

    # Validate required options
    if not options.get("boiler_switch_entity_id"):
        logger.error("boiler_switch_entity_id is required in options.json")
        sys.exit(1)

    controller = SmartBoilerController(options)
    controller.start()
