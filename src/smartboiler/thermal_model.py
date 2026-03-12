# Created as a part of Master's Thesis "Using machine learning methods to save energy in a smart home"
# Faculty of Information Technology, Brno University of Technology, 2024
# Author: Adam Grünwald
#
# Learned thermal model for boiler water temperature estimation.
#
# Physics (Newton's law of cooling):
#   After the thermostat trips (relay ON, power → 0 W) the water is at T_set.
#   Both water and the boiler case then cool exponentially:
#
#     T_water(t) = T_amb + (T_set  - T_amb) * exp(-k_w * dt)
#     T_case(t)  = T_amb + (C0     - T_amb) * exp(-k_c * dt)
#
#   k_case  is fitted from observed case-temperature samples using log-linear
#           regression (only samples recorded while relay is OFF = passive cooling).
#
#   k_water is derived as  k_water = k_case * mass_ratio  because the water in
#           the tank has far more thermal mass than the case shell, so it cools
#           much more slowly (typical ratio 0.2–0.4; default 0.3).
#
# Estimation workflow:
#   Given current T_case and T_amb, and the most-recent calibration event
#   (timestamp t0, T_set, C0=T_case at trip):
#     1. Infer elapsed time from case-sensor decay:
#          dt = -ln((T_case - T_amb) / (C0 - T_amb)) / k_c
#     2. Estimate water temperature:
#          T_water = T_amb + (T_set - T_amb) * exp(-k_w * dt)
#     3. If model not fitted yet → simple proportional fallback.
#
# Seasonal adaptation:
#   All data older than `window_days` (default 7) is pruned on every calibration,
#   and the model is re-fitted automatically.  This lets the model adapt when the
#   boiler is in an unheated space (garage / basement) where ambient temperature
#   varies throughout the year.

import logging
import math
import time
from dataclasses import dataclass
from typing import Callable, List, Optional

import numpy as np

_LOGGER = logging.getLogger(__name__)

# Minimum data requirements before attempting a fit
_MIN_CALIB_EVENTS = 1
_MIN_SAMPLES_FOR_FIT = 8

# Rate-limit on case-temp samples stored (seconds); avoids storing 10 k rows/week
SAMPLE_MIN_INTERVAL_S = 15 * 60  # 15 min

# Safety bounds for k values  [1/hour]
_K_MIN = 0.005   # cools at most over ~8 days
_K_MAX = 3.0     # cools in under 20 minutes (unrealistic → reject)


@dataclass
class _CalibEvent:
    """Thermostat-trip event: relay ON, power ≈ 0 W → T_water == T_set."""
    ts: float          # unix timestamp
    T_set: float       # water temp (=thermostat set point) at this moment
    T_case: float      # case sensor reading at this moment
    T_amb: float       # ambient temp at this moment


@dataclass
class _CaseSample:
    """One case-sensor reading recorded during passive cooling."""
    ts: float
    T_case: float
    T_amb: float
    calib_idx: int     # index into self.calib_events that preceded this sample


class ThermalModel:
    """
    Learns and estimates boiler water temperature from a case temperature sensor.

    Persistence: serialise with ``to_dict()`` / ``from_dict()``; store via
    ``StateStore.save_pickle("thermal_model", model)``.

    Typical usage in the control loop:
        # relay ON + power ≈ 0 W → thermostat tripped
        model.observe_calibration(T_set, T_case, T_amb)

        # relay OFF → passive cooling; call every ~15 min
        model.observe_case_tmp(T_case, T_amb)

        # hourly forecast workflow
        model.maybe_refit()

        # anywhere
        T_est = model.estimate_water_tmp(T_case, T_amb)
    """

    def __init__(
        self,
        window_days: float = 7.0,
        mass_ratio: float = 0.3,
        clock: Optional[Callable[[], float]] = None,
    ):
        """
        Args:
            window_days:  Rolling window for data retention and re-fitting.
                          Reset to 7 days so seasonal ambient changes are tracked.
            mass_ratio:   k_water / k_case.  Water has ~10× the thermal mass of
                          the case shell → cools 3-5× slower → ratio ≈ 0.2-0.4.
            clock:        Callable returning current unix timestamp.  Defaults to
                          ``time.time``.  Override in tests to control "now".
        """
        self.window_days = window_days
        self.mass_ratio = mass_ratio
        self._clock: Callable[[], float] = clock if clock is not None else time.time

        self._calib_events: List[_CalibEvent] = []
        self._samples: List[_CaseSample] = []

        self.k_case: Optional[float] = None   # fitted  [1/h]
        self.k_water: Optional[float] = None  # derived [1/h]
        self._last_fit_ts: float = 0.0
        self._last_sample_ts: float = 0.0
        self._n_fit_samples: int = 0          # how many samples were used in last fit

    # ── Public API ────────────────────────────────────────────────────────

    def observe_calibration(
        self,
        T_set: float,
        T_case: float,
        T_amb: float,
        timestamp: Optional[float] = None,
    ) -> None:
        """
        Record a thermostat-trip calibration event.
        Call this when: relay is ON and power sensor reads < 50 W.
        """
        ts = timestamp if timestamp is not None else self._clock()
        event = _CalibEvent(ts=ts, T_set=T_set, T_case=T_case, T_amb=T_amb)
        self._calib_events.append(event)
        _LOGGER.info(
            "ThermalModel: calibration  T_set=%.1f  T_case=%.1f  T_amb=%.1f",
            T_set, T_case, T_amb,
        )
        self._prune()
        # Trigger a re-fit after every calibration event if we have enough data
        self._try_fit()

    def observe_case_tmp(
        self,
        T_case: float,
        T_amb: float,
        timestamp: Optional[float] = None,
    ) -> None:
        """
        Record a case-temperature sample during passive cooling (relay OFF).
        Rate-limited to SAMPLE_MIN_INTERVAL_S; safe to call every 60 s.
        """
        ts = timestamp if timestamp is not None else self._clock()
        if ts - self._last_sample_ts < SAMPLE_MIN_INTERVAL_S:
            return
        if not self._calib_events:
            return
        idx = len(self._calib_events) - 1
        self._samples.append(_CaseSample(ts=ts, T_case=T_case, T_amb=T_amb, calib_idx=idx))
        self._last_sample_ts = ts

    def maybe_refit(self) -> None:
        """
        Re-fit the model if at least ``window_days`` have elapsed since last fit.
        Call from the hourly forecast workflow.
        """
        age_days = (self._clock() - self._last_fit_ts) / 86400.0
        if age_days >= self.window_days:
            self._try_fit()

    def estimate_water_tmp(
        self,
        T_case: float,
        T_amb: float,
        timestamp: Optional[float] = None,
    ) -> Optional[float]:
        """
        Estimate current water temperature from case sensor and ambient.

        Returns None if no calibration has been observed yet.
        Returns a value clamped to [T_amb, T_set].
        """
        if not self._calib_events:
            return None

        calib = self._calib_events[-1]
        ts = timestamp if timestamp is not None else self._clock()

        # Guard: calibration in the future (clock jump)
        if ts < calib.ts:
            return float(calib.T_set)

        # ── Fitted model ─────────────────────────────────────────────────
        if self.k_case is not None and self.k_water is not None:
            c0_adj = calib.T_case - T_amb
            c_adj  = T_case        - T_amb

            if c0_adj > 0.5 and 0 < c_adj < c0_adj:
                # Infer elapsed time from case-sensor reading
                elapsed_h = -math.log(c_adj / c0_adj) / self.k_case
            else:
                # Case sensor unusable (already at ambient or above C0) →
                # fall back to time-based estimate using calibration timestamp
                elapsed_h = (ts - calib.ts) / 3600.0

            T_water = T_amb + (calib.T_set - T_amb) * math.exp(-self.k_water * elapsed_h)
            return float(max(T_amb, min(calib.T_set, T_water)))

        # ── Unfitted fallback: proportional scaling ───────────────────────
        # T_water / T_case ≈ T_set / C0  (relative to ambient)
        c0_adj = calib.T_case - calib.T_amb
        if c0_adj <= 0:
            return None
        ratio = (calib.T_set - calib.T_amb) / c0_adj
        T_water = T_amb + ratio * (T_case - T_amb)
        return float(max(T_amb, min(calib.T_set, T_water)))

    # ── Diagnostics ───────────────────────────────────────────────────────

    @property
    def is_fitted(self) -> bool:
        return self.k_case is not None

    @property
    def last_calibration_ts(self) -> Optional[float]:
        return self._calib_events[-1].ts if self._calib_events else None

    def diagnostics(self) -> dict:
        calib = self._calib_events[-1] if self._calib_events else None
        return {
            "fitted": self.is_fitted,
            "k_case": round(self.k_case, 4) if self.k_case else None,
            "k_water": round(self.k_water, 4) if self.k_water else None,
            "mass_ratio": self.mass_ratio,
            "window_days": self.window_days,
            "n_calib_events": len(self._calib_events),
            "n_samples": len(self._samples),
            "n_fit_samples": self._n_fit_samples,
            "last_calib_T_set": round(calib.T_set, 1) if calib else None,
            "last_calib_T_case": round(calib.T_case, 1) if calib else None,
        }

    # ── Persistence ───────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "window_days": self.window_days,
            "mass_ratio": self.mass_ratio,
            "k_case": self.k_case,
            "k_water": self.k_water,
            "last_fit_ts": self._last_fit_ts,
            "n_fit_samples": self._n_fit_samples,
            "calib_events": [
                {"ts": e.ts, "T_set": e.T_set, "T_case": e.T_case, "T_amb": e.T_amb}
                for e in self._calib_events
            ],
            "samples": [
                {"ts": s.ts, "T_case": s.T_case, "T_amb": s.T_amb, "calib_idx": s.calib_idx}
                for s in self._samples
            ],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ThermalModel":
        m = cls(
            window_days=d.get("window_days", 7.0),
            mass_ratio=d.get("mass_ratio", 0.3),
        )
        m.k_case = d.get("k_case")
        m.k_water = d.get("k_water")
        m._last_fit_ts = d.get("last_fit_ts", 0.0)
        m._n_fit_samples = d.get("n_fit_samples", 0)
        for ev in d.get("calib_events", []):
            m._calib_events.append(
                _CalibEvent(ev["ts"], ev["T_set"], ev["T_case"], ev["T_amb"])
            )
        for s in d.get("samples", []):
            m._samples.append(
                _CaseSample(s["ts"], s["T_case"], s["T_amb"], s["calib_idx"])
            )
        if m._samples:
            m._last_sample_ts = m._samples[-1].ts
        return m

    # ── Internal helpers ──────────────────────────────────────────────────

    def _prune(self) -> None:
        """Drop data outside the rolling window."""
        cutoff = self._clock() - self.window_days * 86400.0
        self._calib_events = [e for e in self._calib_events if e.ts >= cutoff]
        self._samples = [s for s in self._samples if s.ts >= cutoff]

    def _try_fit(self) -> bool:
        """
        Fit k_case from (dt, log_ratio) pairs via log-linear regression.
        Returns True if fit succeeded and updated self.k_case / self.k_water.
        """
        if len(self._calib_events) < _MIN_CALIB_EVENTS:
            return False

        # Build index for quick lookup
        calib_map = {i: e for i, e in enumerate(self._calib_events)}
        cutoff = self._clock() - self.window_days * 86400.0

        xs: List[float] = []  # dt in hours
        ys: List[float] = []  # ln( (T_case - T_amb) / (C0 - T_amb_at_calib) )

        for s in self._samples:
            calib = calib_map.get(s.calib_idx)
            if calib is None or calib.ts < cutoff:
                continue

            dt_h = (s.ts - calib.ts) / 3600.0
            if dt_h <= 0:
                continue

            # Use current sample's ambient (it may have drifted since calibration)
            c0_adj = calib.T_case - s.T_amb   # initial case excess above ambient
            c_adj  = s.T_case    - s.T_amb    # current case excess

            if c0_adj <= 0.5 or c_adj <= 0 or c_adj >= c0_adj:
                # Boiler re-heated, or case already at ambient — skip
                continue

            xs.append(dt_h)
            ys.append(math.log(c_adj / c0_adj))

        if len(xs) < _MIN_SAMPLES_FOR_FIT:
            _LOGGER.debug(
                "ThermalModel: not enough valid samples to fit (%d/%d)",
                len(xs), _MIN_SAMPLES_FOR_FIT,
            )
            return False

        xs_a = np.array(xs)
        ys_a = np.array(ys)

        # log(ratio) = -k * dt  →  force through origin (intercept ≈ 0 by definition)
        # Weighted least squares: weight by 1/dt to prefer early samples (less noise)
        weights = 1.0 / (xs_a + 1.0)
        # numpy polyfit with weights
        coeffs = np.polyfit(xs_a, ys_a, 1, w=weights)
        k_fit = -float(coeffs[0])

        if not (_K_MIN < k_fit < _K_MAX):
            _LOGGER.warning(
                "ThermalModel: fitted k_case=%.4f is outside bounds [%.3f, %.1f]; ignoring",
                k_fit, _K_MIN, _K_MAX,
            )
            return False

        self.k_case = k_fit
        self.k_water = k_fit * self.mass_ratio
        self._last_fit_ts = self._clock()
        self._n_fit_samples = len(xs)

        _LOGGER.info(
            "ThermalModel: fitted k_case=%.4f/h  k_water=%.4f/h  "
            "(mass_ratio=%.2f, %d samples, window=%.0f days)",
            self.k_case, self.k_water, self.mass_ratio,
            self._n_fit_samples, self.window_days,
        )
        return True
