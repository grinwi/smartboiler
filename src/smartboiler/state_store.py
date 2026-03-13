# Created as a part of Master's Thesis "Using machine learning methods to save energy in a smart home"
# Faculty of Information Technology, Brno University of Technology, 2024
# Author: Adam Grünwald
#
# Persistent storage layer — survives add-on restarts via /data/ directory.

import json
import logging
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)

CONSUMPTION_HISTORY_DAYS = 90


class StateStore:
    """Manages persistent state in /data/ directory."""

    def __init__(self, data_dir: str = "/data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._state_path = self.data_dir / "state.json"
        self._consumption_path = self.data_dir / "consumption_history.pkl"
        self._state: Dict[str, Any] = self._load_json_state()

    # ── JSON key-value state ──────────────────────────────────────────────

    def _load_json_state(self) -> Dict[str, Any]:
        if self._state_path.exists():
            try:
                with self._state_path.open() as f:
                    return json.load(f)
            except Exception as e:
                logger.warning("Could not load state.json: %s", e)
        return {}

    def _save_json_state(self) -> None:
        try:
            with self._state_path.open("w") as f:
                json.dump(self._state, f, default=str, indent=2)
        except Exception as e:
            logger.error("Could not save state.json: %s", e)

    def get(self, key: str, default=None):
        return self._state.get(key, default)

    def set(self, key: str, value) -> None:
        self._state[key] = value
        self._save_json_state()

    # ── Consumption history ───────────────────────────────────────────────

    def load_consumption_history(self) -> pd.DataFrame:
        """Load hourly consumption DataFrame from pickle.

        Columns: consumed_kwh, relay_on, power_w  (DatetimeIndex)
        """
        if self._consumption_path.exists():
            try:
                with self._consumption_path.open("rb") as f:
                    df = pickle.load(f)
                if isinstance(df, pd.DataFrame):
                    return df
            except Exception as e:
                logger.warning("Could not load consumption history: %s", e)
        return pd.DataFrame(
            columns=["consumed_kwh", "relay_on", "power_w"],
            dtype=float,
        )

    def save_consumption_history(self, df: pd.DataFrame) -> None:
        try:
            with self._consumption_path.open("wb") as f:
                pickle.dump(df, f)
        except Exception as e:
            logger.error("Could not save consumption history: %s", e)

    def append_consumption(self, df_new: pd.DataFrame) -> pd.DataFrame:
        """Merge new rows into history; deduplicate; keep last 90 days."""
        df_existing = self.load_consumption_history()
        df_combined = pd.concat([df_existing, df_new])
        df_combined = df_combined[~df_combined.index.duplicated(keep="last")]
        df_combined = df_combined.sort_index()
        cutoff = pd.Timestamp.now() - pd.Timedelta(days=CONSUMPTION_HISTORY_DAYS)
        df_combined = df_combined[df_combined.index >= cutoff]
        self.save_consumption_history(df_combined)
        return df_combined

    # ── Generic pickle helpers (learner / predictor objects) ──────────────

    def load_pickle(self, name: str) -> Optional[Any]:
        path = self.data_dir / f"{name}.pkl"
        if path.exists():
            try:
                with path.open("rb") as f:
                    return pickle.load(f)
            except Exception as e:
                logger.warning("Could not load %s.pkl: %s", name, e)
        return None

    def save_pickle(self, name: str, obj: Any) -> None:
        path = self.data_dir / f"{name}.pkl"
        try:
            with path.open("wb") as f:
                pickle.dump(obj, f)
        except Exception as e:
            logger.error("Could not save %s.pkl: %s", name, e)

    # ── Typed convenience accessors ───────────────────────────────────────

    def get_last_legionella_heating(self) -> datetime:
        val = self._state.get("last_legionella_heating")
        if val:
            try:
                return datetime.fromisoformat(val)
            except Exception:
                pass
        # Default: now (so legionella check doesn't immediately trigger)
        return datetime.now().astimezone()

    def set_last_legionella_heating(self, dt: datetime) -> None:
        self.set("last_legionella_heating", dt.isoformat())

    def get_last_data_collection(self) -> datetime:
        val = self._state.get("last_data_collection")
        if val:
            try:
                return datetime.fromisoformat(val)
            except Exception:
                pass
        return datetime(2000, 1, 1)

    def set_last_data_collection(self, dt: datetime) -> None:
        self.set("last_data_collection", dt.isoformat())

    def get_heating_plan(self) -> List:
        return self._state.get("heating_plan", [])

    def set_heating_plan(self, plan: List) -> None:
        self.set("heating_plan", plan)

    def get_spot_cache(self) -> Dict:
        return self._state.get("spot_cache", {})

    def set_spot_cache(self, cache: Dict) -> None:
        self.set("spot_cache", cache)

    def get_heating_until(self) -> Optional[datetime]:
        val = self._state.get("heating_until")
        if val:
            try:
                return datetime.fromisoformat(val)
            except Exception:
                pass
        return None

    def set_heating_until(self, dt: Optional[datetime]) -> None:
        self.set("heating_until", dt.isoformat() if dt else None)

    def get_last_boiler_tmp(self) -> Optional[float]:
        val = self._state.get("last_boiler_tmp")
        try:
            return float(val) if val is not None else None
        except (ValueError, TypeError):
            return None

    def set_last_boiler_tmp(self, tmp: float) -> None:
        self.set("last_boiler_tmp", tmp)
