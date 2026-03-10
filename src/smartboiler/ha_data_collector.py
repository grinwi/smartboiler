# Created as a part of Master's Thesis "Using machine learning methods to save energy in a smart home"
# Faculty of Information Technology, Brno University of Technology, 2024
# Author: Adam Grünwald
#
# Collects hourly consumption data from Home Assistant entity history
# and maintains the local cache used by the predictor and HDO learner.

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from smartboiler.ha_client import HAClient
from smartboiler.state_store import StateStore

logger = logging.getLogger(__name__)


class HADataCollector:
    """Reads entity history from HA, computes hourly consumption, updates cache."""

    def __init__(
        self,
        ha: HAClient,
        store: StateStore,
        relay_entity_id: str,
        relay_power_entity_id: Optional[str] = None,
        water_flow_entity_id: Optional[str] = None,
        water_temp_out_entity_id: Optional[str] = None,
        boiler_case_tmp_entity_id: Optional[str] = None,
        boiler_volume_l: float = 120.0,
        boiler_set_tmp: float = 60.0,
        cold_water_tmp: float = 10.0,
    ):
        self.ha = ha
        self.store = store
        self.relay_entity_id = relay_entity_id
        self.relay_power_entity_id = relay_power_entity_id
        self.water_flow_entity_id = water_flow_entity_id
        self.water_temp_out_entity_id = water_temp_out_entity_id
        self.boiler_case_tmp_entity_id = boiler_case_tmp_entity_id
        self.boiler_volume_l = boiler_volume_l
        self.boiler_set_tmp = boiler_set_tmp
        self.cold_water_tmp = cold_water_tmp

    def collect_and_update(self, lookback_hours: int = 6) -> pd.DataFrame:
        """Fetch the last lookback_hours of data from HA and update local cache.

        Returns the updated full consumption history DataFrame.
        """
        end = datetime.now()
        start = end - timedelta(hours=lookback_hours)

        df_new = self._fetch_hourly(start, end)
        if df_new.empty:
            logger.info("No new consumption data fetched.")
            return self.store.load_consumption_history()

        df_all = self.store.append_consumption(df_new)
        self.store.set_last_data_collection(datetime.now())
        logger.info(
            "Collected %d new hourly rows; total history: %d rows",
            len(df_new),
            len(df_all),
        )
        return df_all

    def _fetch_hourly(self, start: datetime, end: datetime) -> pd.DataFrame:
        """Fetch entity history and aggregate to hourly kWh rows."""
        # Get relay on/off state changes
        relay_history = self.ha.get_history(self.relay_entity_id, start, end)
        # Get power history (optional, used for HDO detection)
        power_history: List[Dict] = []
        if self.relay_power_entity_id:
            power_history = self.ha.get_history(self.relay_power_entity_id, start, end)

        # Build minute-resolution series for relay state and power
        relay_series = self._states_to_series(relay_history, start, end, dtype="bool")
        power_series = self._states_to_series(power_history, start, end, dtype="float")

        # Compute consumed kWh per hour via water flow sensor or fallback
        if self.water_flow_entity_id and self.water_temp_out_entity_id:
            flow_history = self.ha.get_history(self.water_flow_entity_id, start, end)
            temp_history = self.ha.get_history(self.water_temp_out_entity_id, start, end)
            consumed_series = self._compute_consumed_kwh_from_flow(
                flow_history, temp_history, start, end
            )
        else:
            # Fallback: estimate from relay on-time (very rough)
            consumed_series = self._estimate_consumed_from_relay(relay_series)

        # Resample to hourly
        df = pd.DataFrame(
            {
                "consumed_kwh": consumed_series,
                "relay_on": relay_series,
                "power_w": power_series,
            }
        )
        df.index = pd.to_datetime(df.index)
        df_hourly = df.resample("1h").agg(
            {
                "consumed_kwh": "sum",
                "relay_on": "mean",
                "power_w": "mean",
            }
        )
        df_hourly["relay_on"] = df_hourly["relay_on"] > 0.5
        return df_hourly.dropna(how="all")

    def _states_to_series(
        self,
        history: List[Dict],
        start: datetime,
        end: datetime,
        dtype: str = "float",
    ) -> pd.Series:
        """Convert HA history state objects to a minute-resolution pandas Series."""
        minutes = pd.date_range(start, end, freq="1min")
        values = pd.Series(index=minutes, dtype=float)

        if not history:
            return values

        for i, state_obj in enumerate(history):
            ts_str = state_obj.get("last_changed") or state_obj.get("last_updated", "")
            raw = state_obj.get("state", "")
            try:
                ts = pd.to_datetime(ts_str, utc=True).tz_localize(None)
                if dtype == "bool":
                    val = 1.0 if raw in ("on", "true", "1") else 0.0
                else:
                    val = float(raw)
            except (ValueError, TypeError):
                continue

            # Find next state change time (or end)
            if i + 1 < len(history):
                next_ts_str = history[i + 1].get("last_changed") or history[i + 1].get("last_updated", "")
                try:
                    next_ts = pd.to_datetime(next_ts_str, utc=True).tz_localize(None)
                except Exception:
                    next_ts = end
            else:
                next_ts = end

            mask = (values.index >= ts) & (values.index < next_ts)
            values[mask] = val

        return values

    def _compute_consumed_kwh_from_flow(
        self,
        flow_history: List[Dict],
        temp_history: List[Dict],
        start: datetime,
        end: datetime,
    ) -> pd.Series:
        """Compute consumed kWh from water flow (L/min) and outlet temperature."""
        minutes = pd.date_range(start, end, freq="1min")
        flow = self._states_to_series(flow_history, start, end, dtype="float").fillna(0)
        temp_out = self._states_to_series(temp_history, start, end, dtype="float").fillna(
            self.boiler_set_tmp
        )
        # Q = m * c * ΔT  where m = flow (L/min) * density (1 kg/L)
        # kWh per minute = flow_l_min * 4.186 kJ/(kg·K) * (T_out - T_cold) / 3600
        delta_t = (temp_out - self.cold_water_tmp).clip(lower=0)
        kwh_per_min = flow * 4.186 * delta_t / 3600.0
        consumed = pd.Series(kwh_per_min.values, index=minutes)
        return consumed

    def _estimate_consumed_from_relay(self, relay_series: pd.Series) -> pd.Series:
        """Very rough fallback: assume each ON minute means the boiler heated.
        Sets consumed_kwh = 0 (we can't measure consumption without a flow sensor)."""
        return pd.Series(0.0, index=relay_series.index)

    def get_current_readings(self) -> Dict:
        """Return current entity values for the dashboard."""
        result: Dict = {}

        relay_state = self.ha.get_state(self.relay_entity_id)
        result["relay_on"] = (
            relay_state.get("state") in ("on", "true", "1") if relay_state else None
        )

        if self.relay_power_entity_id:
            result["power_w"] = self.ha.get_state_value(self.relay_power_entity_id)

        if self.boiler_case_tmp_entity_id:
            result["boiler_case_tmp"] = self.ha.get_state_value(self.boiler_case_tmp_entity_id)

        if self.water_temp_out_entity_id:
            result["water_temp_out"] = self.ha.get_state_value(self.water_temp_out_entity_id)

        return result
