# Created as a part of Master's Thesis "Using machine learning methods to save energy in a smart home"
# Faculty of Information Technology, Brno University of Technology, 2024
# Author: Adam Grünwald
#
# HDO (ripple control) pattern learner.
#
# HDO blocking signal: the boiler switch entity reports state "unavailable" in HA.
# This happens because the HDO relay physically cuts the circuit, making the smart
# plug/switch unable to communicate.
#
# NOTE: relay available + power = 0  means the boiler thermostat target temperature
# was reached — this is NOT HDO and must NOT be treated as a blocking signal.
#
# Resolution: 5-minute slots (288 per day).  HDO schedules like "12:05-13:35" can
# only be represented correctly at this granularity.
#
# Trust policy: a slot is only considered blocked when observations span at least
# MIN_WEEKS_TO_TRUST distinct ISO calendar weeks, ensuring we never rely on a single
# anomalous reading.

import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

logger = logging.getLogger(__name__)

SLOT_MINUTES = 5                # resolution: 5 minutes
SLOTS_PER_DAY = 24 * 60 // SLOT_MINUTES   # 288 slots per day
HISTORY_WEEKS = 3               # rolling observation window
MIN_WEEKS_TO_TRUST = 2          # need observations from ≥2 distinct calendar weeks
MIN_CONFIDENCE_TO_BLOCK = 0.7   # ≥70% of weighted observations must be "unavailable"


def _now_ts() -> float:
    return datetime.now(tz=timezone.utc).timestamp()


def _parse_hhmm(s: str) -> Tuple[int, int]:
    """Parse "HH:MM" or "HH" into (hour, minute)."""
    parts = s.strip().split(":")
    h = int(parts[0])
    m = int(parts[1]) if len(parts) > 1 else 0
    return h, m


class HDOLearner:
    """Infers HDO blocking schedule from relay-unavailable observations.

    Call observe() every control-loop cycle (e.g. every 60 s) with the
    current raw state of the boiler switch entity.  The learner accumulates
    per-slot evidence and uses exponential decay so that schedule changes
    propagate within a few weeks.
    """

    def __init__(self, decay_weeks: int = 2, history_weeks: int = HISTORY_WEEKS):
        """
        Args:
            decay_weeks: Exponential decay half-life (weeks) for weighting
                         recent vs old observations.
            history_weeks: Rolling retention window for observations.
        """
        self.decay_weeks = decay_weeks
        self.history_weeks = history_weeks
        # (weekday, slot_idx) → [(unix_ts, is_unavailable, (iso_year, iso_week)), ...]
        self._observations: Dict[
            Tuple[int, int],
            List[Tuple[float, bool, Tuple[int, int]]]
        ] = {}
        # Set of (weekday, slot_idx) explicitly blocked by config string
        self._explicit_blocked: Set[Tuple[int, int]] = set()

    # ── Explicit schedule ─────────────────────────────────────────────────────

    def set_explicit_schedule(self, schedule_str: Optional[str]) -> None:
        """Parse an explicit HDO schedule from a config string.

        Format: "HH:MM-HH:MM[,HH:MM-HH:MM,...]"  (5-minute resolution).
        Examples:
            "22:00-06:00"          night window, wraps midnight
            "12:05-13:35"          sub-hour window
            "22:00-06:00,13:00-15:00"  two segments

        Applies to all weekdays.  Calling again replaces the previous schedule.
        Explicit schedule always takes priority over learned schedule.
        """
        self._explicit_blocked = set()
        if not schedule_str or schedule_str.strip().lower() in ("", "empty"):
            return
        for segment in schedule_str.split(","):
            segment = segment.strip()
            if "-" not in segment:
                continue
            try:
                start_str, end_str = segment.split("-", 1)
                s_h, s_m = _parse_hhmm(start_str)
                e_h, e_m = _parse_hhmm(end_str)
                start_slot = (s_h * 60 + s_m) // SLOT_MINUTES
                end_slot = (e_h * 60 + e_m) // SLOT_MINUTES
                if start_slot == end_slot:
                    continue  # zero-length segment — skip
                if start_slot < end_slot:
                    slot_range: List[int] = list(range(start_slot, end_slot))
                else:  # wraps midnight
                    slot_range = list(range(start_slot, SLOTS_PER_DAY)) + list(range(0, end_slot))
                for s in slot_range:
                    for wd in range(7):
                        self._explicit_blocked.add((wd, s))
            except Exception as exc:
                logger.warning("Could not parse HDO segment '%s': %s", segment, exc)

    # ── Observation ───────────────────────────────────────────────────────────

    def _slot_of(self, dt: datetime) -> Tuple[int, int]:
        """Return (weekday, slot_index) for a datetime (uses local/aware time)."""
        slot_idx = (dt.hour * 60 + dt.minute) // SLOT_MINUTES
        return (dt.weekday(), slot_idx)

    def _cutoff_ts(self, now_ts: Optional[float] = None) -> float:
        base = _now_ts() if now_ts is None else now_ts
        return base - self.history_weeks * 7 * 24 * 3600

    def _prune_all(self, now_ts: Optional[float] = None) -> None:
        cutoff_ts = self._cutoff_ts(now_ts)
        pruned: Dict[Tuple[int, int], List[Tuple[float, bool, Tuple[int, int]]]] = {}
        for key, obs in self._observations.items():
            kept = [(t, b, yw) for t, b, yw in obs if t > cutoff_ts]
            if kept:
                pruned[key] = kept
        self._observations = pruned

    def clear_observations(self) -> None:
        """Drop learned observations while keeping the explicit schedule."""
        self._observations = {}

    def observe(self, dt: datetime, relay_unavailable: bool) -> None:
        """Record one observation.

        Args:
            dt: Timestamp of the observation (timezone-aware preferred).
            relay_unavailable: True when the HA entity state == "unavailable",
                indicating HDO has physically cut the relay circuit.
                False when the relay is available (state "on" or "off"),
                regardless of whether the boiler is actually heating.
        """
        key = self._slot_of(dt)
        ts = dt.timestamp()
        iso = dt.isocalendar()
        year_week: Tuple[int, int] = (int(iso[0]), int(iso[1]))
        self._observations.setdefault(key, []).append((ts, relay_unavailable, year_week))
        self._prune_all()

    # ── Blocking queries ──────────────────────────────────────────────────────

    def _is_slot_blocked(self, weekday: int, slot_idx: int) -> bool:
        """Return True if the 5-minute slot is HDO-blocked."""
        if (weekday, slot_idx) in self._explicit_blocked:
            return True
        key = (weekday, slot_idx)
        obs = self._observations.get(key, [])
        if not obs:
            return False
        # Require data from at least MIN_WEEKS_TO_TRUST distinct calendar weeks
        distinct_weeks = len({yw for _, _, yw in obs})
        if distinct_weeks < MIN_WEEKS_TO_TRUST:
            return False
        # Exponentially-weighted confidence
        now_ts = _now_ts()
        decay_s = self.decay_weeks * 7 * 24 * 3600
        weights = [np.exp(-(now_ts - t) / decay_s) for t, _, _ in obs]
        total_w = sum(weights)
        blocked_w = sum(wt for wt, (_, b, _) in zip(weights, obs) if b)
        confidence = blocked_w / total_w if total_w > 0 else 0.0
        return bool(confidence >= MIN_CONFIDENCE_TO_BLOCK)

    def is_blocked_at(self, dt: datetime) -> bool:
        """Return True if HDO is likely blocking at this exact datetime."""
        self._prune_all()
        key = self._slot_of(dt)
        return self._is_slot_blocked(*key)

    def is_blocked(self, weekday: int, hour: int) -> bool:
        """Return True if HDO is blocking at any 5-min slot within this hour.

        Backward-compatible interface used by the scheduler and dashboard.
        """
        self._prune_all()
        start_slot = (hour * 60) // SLOT_MINUTES   # = hour * 12
        return any(self._is_slot_blocked(weekday, s) for s in range(start_slot, start_slot + 12))

    def get_blocked_hours_next_24h(self, from_dt: Optional[datetime] = None) -> List[bool]:
        """Return 24 booleans — one per upcoming hour — indicating HDO blocking."""
        self._prune_all()
        if from_dt is None:
            from_dt = datetime.now().astimezone()
        result = []
        for i in range(24):
            total_hours = from_dt.hour + i
            weekday = (from_dt.weekday() + total_hours // 24) % 7
            hour = total_hours % 24
            result.append(bool(self.is_blocked(weekday, hour)))
        return result

    def get_weekly_schedule(self) -> Dict[str, List[int]]:
        """Return inferred+explicit HDO schedule as {day_name → [blocked hours]}."""
        self._prune_all()
        day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        schedule: Dict[str, List[int]] = {day: [] for day in day_names}
        for wd in range(7):
            for h in range(24):
                if self.is_blocked(wd, h):
                    schedule[day_names[wd]].append(h)
        return schedule

    def observation_count(self) -> int:
        """Total number of stored observations across all slots."""
        self._prune_all()
        return sum(len(v) for v in self._observations.values())
