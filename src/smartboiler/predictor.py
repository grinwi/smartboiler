# Created as a part of Master's Thesis "Using machine learning methods to save energy in a smart home"
# Faculty of Information Technology, Brno University of Technology, 2024
# Author: Adam Grünwald
#
# Rolling histogram predictor — per-(weekday × hour) quantile over last 12 weeks.
#
# Why quantile instead of mean?
#   Hot water consumption within a given slot is right-skewed: most hours have zero
#   or minimal use, but occasional heavy draws (showers, baths) pull the mean up.
#   A quantile estimate (p75 by default) is much more robust to this skew and gives
#   a conservative "plan for this much" number that's rarely exceeded.
#
# Why not LSTM?
#   With < 6 months of data (typical for a new installation) an LSTM offers no
#   measurable accuracy improvement over a well-tuned histogram.  The histogram
#   requires no training, no GPU, and < 1 ms per prediction cycle.
#
# Fallback strategy:
#   If a (weekday, hour) slot has fewer than MIN_SAMPLES_FOR_SLOT observations
#   (new installation, or rarely-used slot), the global quantile of all non-zero
#   consumption hours is returned instead.  This is intentionally conservative —
#   it avoids under-heating during the learning period.

import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

HISTORY_WEEKS = 12
# Minimum observations before trusting a per-slot prediction.
# 4 is intentionally low — for a slot that fires e.g. every Saturday morning,
# 4 observations correspond to just one month of data, which is already enough
# to distinguish "this household showers at this time" from background noise.
# Increasing this delays adaptation and keeps more slots in fallback mode longer.
MIN_SAMPLES_FOR_SLOT = 4


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
            from_dt = datetime.now().astimezone()
        base = from_dt.replace(minute=0, second=0, microsecond=0)
        predictions = []
        for i in range(24):
            dt = base + pd.Timedelta(hours=i)
            predictions.append(self.predict_hour(dt.weekday(), dt.hour))
        return predictions

    def has_enough_data(self, min_days: int = 30) -> bool:
        """True if we have enough history to make meaningful predictions.

        The threshold of 20 usable hours/day is a rough heuristic: a household
        typically has consumption in ~4-8 distinct hours per day, but the
        DataFrame contains all 24 hourly rows.  Multiplying min_days × 20 gives
        a conservative total-sample threshold that corresponds roughly to
        min_days of collection with a non-trivial usage pattern.
        """
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
