# Created as a part of Master's Thesis "Using machine learning methods to save energy in a smart home"
# Faculty of Information Technology, Brno University of Technology, 2024
# Author: Adam Grünwald
#
# HDO (ripple control) pattern learner.
# Detects when relay is ON but measured power ≈ 0W → HDO is blocking.
# Builds a weekday×hour confidence map that adapts as the utility changes schedules.

import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

POWER_ZERO_THRESHOLD_W = 50.0  # below this with relay ON = HDO blocking
MIN_CONFIDENCE_TO_BLOCK = 0.6   # 60% of weighted observations = blocked
HISTORY_WEEKS = 8               # forget observations older than this


class HDOLearner:
    """Infers HDO blocking schedule from power consumption observations."""

    def __init__(self, decay_weeks: int = 4):
        """
        Args:
            decay_weeks: Exponential decay half-life for weighting recent vs old observations.
        """
        self.decay_weeks = decay_weeks
        # (weekday, hour) → [(unix_timestamp, is_blocked), ...]
        self._observations: Dict[Tuple[int, int], List[Tuple[float, bool]]] = {}
        self._explicit_blocked: List[Tuple[int, int]] = []

    def set_explicit_schedule(self, blocked_hours: str) -> None:
        """Parse an explicit HDO schedule from config string.

        Format: "22:00-06:00,13:00-15:00" (start-end pairs, can wrap midnight).
        Explicit schedule takes priority over learned schedule.
        """
        self._explicit_blocked = []
        if not blocked_hours or blocked_hours.strip() in ("", "empty"):
            return
        for segment in blocked_hours.split(","):
            segment = segment.strip()
            if "-" not in segment:
                continue
            try:
                start_str, end_str = segment.split("-", 1)
                start_h = int(start_str.split(":")[0])
                end_h = int(end_str.split(":")[0])
                if start_h < end_h:
                    hours = range(start_h, end_h)
                elif start_h > end_h:  # wraps midnight
                    hours = list(range(start_h, 24)) + list(range(0, end_h))
                else:
                    continue  # zero-length, skip
                for h in hours:
                    for wd in range(7):
                        self._explicit_blocked.append((wd, h))
            except Exception as e:
                logger.warning("Could not parse HDO segment '%s': %s", segment, e)

    def observe(self, dt: datetime, relay_on: bool, power_w: float) -> None:
        """Record one observation. Call each minute when relay state is known."""
        if not relay_on:
            return  # only meaningful when relay should be on
        is_blocked = power_w < POWER_ZERO_THRESHOLD_W
        key = (dt.weekday(), dt.hour)
        self._observations.setdefault(key, []).append((dt.timestamp(), is_blocked))
        # Prune old observations
        cutoff = datetime.now().timestamp() - HISTORY_WEEKS * 7 * 24 * 3600
        self._observations[key] = [
            (ts, b) for ts, b in self._observations[key] if ts > cutoff
        ]

    def is_blocked(self, weekday: int, hour: int) -> bool:
        """Return True if HDO is likely blocking at this weekday+hour."""
        if (weekday, hour) in self._explicit_blocked:
            return True
        key = (weekday, hour)
        obs = self._observations.get(key, [])
        if len(obs) < 3:
            return False  # insufficient data
        now = datetime.now().timestamp()
        decay_s = self.decay_weeks * 7 * 24 * 3600
        weights = [np.exp(-(now - ts) / decay_s) for ts, _ in obs]
        total_w = sum(weights)
        blocked_w = sum(w for w, (_, b) in zip(weights, obs) if b)
        confidence = blocked_w / total_w if total_w > 0 else 0.0
        return confidence >= MIN_CONFIDENCE_TO_BLOCK

    def get_blocked_hours_next_24h(self, from_dt: Optional[datetime] = None) -> List[bool]:
        """Return 24 booleans for whether each upcoming hour is HDO-blocked."""
        if from_dt is None:
            from_dt = datetime.now()
        result = []
        for i in range(24):
            total_hours = from_dt.hour + i
            weekday = (from_dt.weekday() + total_hours // 24) % 7
            hour = total_hours % 24
            result.append(self.is_blocked(weekday, hour))
        return result

    def get_weekly_schedule(self) -> Dict[str, List[int]]:
        """Return inferred+explicit HDO schedule as day→blocked hours list."""
        day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        schedule: Dict[str, List[int]] = {day: [] for day in day_names}
        for wd in range(7):
            for h in range(24):
                if self.is_blocked(wd, h):
                    schedule[day_names[wd]].append(h)
        return schedule

    def observation_count(self) -> int:
        return sum(len(v) for v in self._observations.values())
