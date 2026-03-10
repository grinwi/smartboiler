# Created as a part of Master's Thesis "Using machine learning methods to save energy in a smart home"
# Faculty of Information Technology, Brno University of Technology, 2024
# Author: Adam Grünwald
#
# Rolling histogram predictor — hist_p75 per weekday×hour over last 12 weeks.
# Much more data-efficient than LSTM; works from day 1 with a global fallback.

import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

HISTORY_WEEKS = 12
MIN_SAMPLES_FOR_SLOT = 4  # need at least 4 observations to trust a slot


class RollingHistogramPredictor:
    """Predicts hourly hot water consumption via weekly histogram quantile."""

    def __init__(self, conservatism: str = "medium"):
        """
        Args:
            conservatism: "low" (p50), "medium" (p75), "high" (p90)
        """
        q_map = {"low": 0.50, "medium": 0.75, "high": 0.90}
        self.quantile = q_map.get(conservatism, 0.75)
        # (weekday, hour) → list of kWh values from non-zero consumption hours
        self._hist: Dict[Tuple[int, int], List[float]] = {}
        self._global_fallback: float = 0.0
        self._total_samples: int = 0

    def update(self, df: pd.DataFrame) -> None:
        """Rebuild histogram from DataFrame with DatetimeIndex and 'consumed_kwh' column.

        Keeps only the last HISTORY_WEEKS weeks of data.
        """
        if df is None or df.empty or "consumed_kwh" not in df.columns:
            return

        cutoff = pd.Timestamp.now() - pd.Timedelta(weeks=HISTORY_WEEKS)
        df_recent = df[df.index >= cutoff].copy()
        # Only include hours with meaningful consumption (filter sensor noise)
        df_nonzero = df_recent[df_recent["consumed_kwh"] > 0.005]

        self._hist = {}
        for ts, row in df_nonzero.iterrows():
            key = (ts.weekday(), ts.hour)
            self._hist.setdefault(key, []).append(float(row["consumed_kwh"]))

        self._total_samples = len(df_recent)

        all_vals = df_nonzero["consumed_kwh"].values
        if len(all_vals) > 0:
            self._global_fallback = float(np.quantile(all_vals, self.quantile))
        else:
            self._global_fallback = 0.0

        logger.info(
            "Predictor updated: %d total hours, %d non-zero slots, global_p%.0f=%.4f kWh",
            self._total_samples,
            len(self._hist),
            self.quantile * 100,
            self._global_fallback,
        )

    def predict_hour(self, weekday: int, hour: int) -> float:
        """Return predicted consumption (kWh) for a specific weekday+hour.

        Falls back to global quantile if the slot has too few samples.
        """
        key = (weekday, hour)
        values = self._hist.get(key, [])
        if len(values) >= MIN_SAMPLES_FOR_SLOT:
            return float(np.quantile(values, self.quantile))
        return self._global_fallback

    def predict_next_24h(self, from_dt: Optional[datetime] = None) -> List[float]:
        """Return predicted consumption for the next 24 hours (starting from_dt)."""
        if from_dt is None:
            from_dt = datetime.now()
        base = from_dt.replace(minute=0, second=0, microsecond=0)
        predictions = []
        for i in range(24):
            dt = base + pd.Timedelta(hours=i)
            predictions.append(self.predict_hour(dt.weekday(), dt.hour))
        return predictions

    def has_enough_data(self, min_days: int = 30) -> bool:
        """True if we have enough history to make meaningful predictions."""
        return self._total_samples >= min_days * 20  # ~20 usable hours/day

    def get_histogram_summary(self) -> Dict:
        """Return per-day summary for dashboard display."""
        day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        summary: Dict = {}
        for (weekday, hour), values in self._hist.items():
            day = day_names[weekday]
            summary.setdefault(day, {})[hour] = {
                "p50": round(float(np.quantile(values, 0.5)), 4) if values else 0.0,
                "p75": round(float(np.quantile(values, 0.75)), 4) if values else 0.0,
                "count": len(values),
            }
        return summary
