# Created as a part of Master's Thesis "Using machine learning methods to save energy in a smart home"
# Faculty of Information Technology, Brno University of Technology, 2024
# Author: Adam Grünwald
#
# Simple Mode consumption estimator — estimates daily hot-water volume without a flow sensor.
#
# Physical basis (validated in analysis/simple_mode_analysis.ipynb):
#   Excess-cooling event detection fails on real data because the relay fires
#   mid-draw, destroying the Newton baseline.  The energy-balance method works:
#
#     vol_est_L = (relay_on_h × relay_kW − standby_kWh_elapsed)
#                 × 3.6e6 / (C_water × (T_set − T_cold)) × alpha
#
# T_set calibration (no direct NTC probe):
#   At each relay-OFF transition, T_water ≈ T_set (tank just heated).
#   From the case sensor:  T_water = T_amb + (T_case − T_amb) / coupling
#   Running 90th-percentile of these samples converges to the thermostat set-point.
#
# Alpha correction (online EWMA, lr=0.10):
#   Updated when ground-truth flow volume is available (Standard Mode with flow
#   sensor configured).  In pure Simple Mode alpha stays at 1.0 until adjusted.

import logging
from datetime import date, datetime, timedelta
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)

C_WATER = 4186.0   # J / (kg·K)
RHO     = 1.0      # kg / L

_ALPHA_LR  = 0.10
_ALPHA_MIN = 0.5
_ALPHA_MAX = 2.0

# Exported so influx_consumption_importer can use the same constant
_BOILER_CP_KWH_PER_L_K = C_WATER * RHO / 3.6e6  # ≈ 0.001163 kWh/(L·K)

_COUPLING_LR   = 0.15          # EMA learning rate for coupling calibration
_COUPLING_MIN  = 0.10
_COUPLING_MAX  = 0.90
_DRAW_ALPHA_LR = 0.05          # smaller LR when draw-detection is the signal


class FlowlessConsumptionEstimator:
    """
    Estimates daily hot-water consumption from relay ON time and boiler physics.

    Persistence: use ``to_dict()`` / ``from_dict()``; store via
    ``StateStore.save_pickle("flow_estimator", estimator)``.

    Typical usage::

        # Every control loop tick (~60 s):
        estimator.tick(relay_on, T_case, T_amb)

        # Every hour (forecast workflow) — check for day rollover:
        estimator.maybe_finalize(k_water=thermal_model.k_water, T_amb=amb)

        # Dashboard:
        diag = estimator.diagnostics()
    """

    def __init__(
        self,
        vol_L: float = 100.0,
        relay_w: float = 2000.0,
        T_cold: float = 10.0,
        coupling: float = 0.45,
        T_amb_default: float = 20.0,
        standby_w: float = 50.0,
        draw_threshold_c: float = 2.0,
    ):
        """
        Args:
            vol_L:            Boiler tank volume [L].
            relay_w:          Relay heating power [W].
            T_cold:           Cold water inlet temperature [°C] (static fallback).
            coupling:         T_case to T_water coupling ratio (0.3–0.6).
                              T_case ≈ T_amb + coupling × (T_water − T_amb)
            T_amb_default:    Fallback ambient temp when no sensor [°C].
            standby_w:        Standby heat-loss power [W].
            draw_threshold_c: Minimum T_case drop per tick [°C] to detect a draw
                              while the relay is OFF.
        """
        self.vol_L            = float(vol_L)
        self.relay_w          = float(relay_w)
        self.T_cold           = float(T_cold)
        self.coupling         = float(coupling)
        self.T_amb_default    = float(T_amb_default)
        self.standby_w        = float(standby_w)
        self.draw_threshold_c = float(draw_threshold_c)

        # ── Learned state (persisted) ─────────────────────────────────────
        self.alpha: float                     = 1.0
        self.T_set_calibrated: Optional[float] = None
        self.coupling_calibrated: Optional[float] = None  # auto-fitted coupling

        # ── Intraday accumulators ─────────────────────────────────────────
        self._current_day: Optional[date]    = None
        self._relay_on_s: float              = 0.0
        self._elapsed_s: float              = 0.0
        self._T_set_samples: List[float]     = []
        self._coupling_samples: List[float]  = []   # coupling inferred today
        self._T_in_samples: List[float]      = []   # T_in (inlet) readings today
        self._T_cold_live: Optional[float]   = None  # daily median of T_in
        self._was_relay_on: Optional[bool]   = None
        self._prev_T_case: Optional[float]   = None  # for draw detection
        self._draw_vol_today: float          = 0.0   # draw-detected litres today

        # ── History of finalized daily estimates ──────────────────────────
        self._daily_history: List[dict]      = []

    # ── Public API ────────────────────────────────────────────────────────

    def tick(
        self,
        relay_on: bool,
        T_case: Optional[float],
        T_amb: Optional[float],
        dt_s: float = 60.0,
        T_in: Optional[float] = None,
    ) -> None:
        """
        Call once per control loop iteration (~60 s).

        Args:
            relay_on:  True if boiler relay is ON.
            T_case:    Case temperature sensor reading [°C] or None.
            T_amb:     Ambient temperature [°C] or None.
            dt_s:      Elapsed seconds since last tick.
            T_in:      Cold-water inlet probe reading [°C] or None.
                       When provided, overrides the static T_cold config value
                       via a daily median (T_cold_live).
        """
        today = datetime.now().date()
        if self._current_day is None:
            self._current_day = today

        # Day boundary detected — do NOT reset accumulators or advance _current_day
        # here.  Resetting in tick() would prevent maybe_finalize() from ever
        # running (maybe_finalize checks today != _current_day, which tick() would
        # already have updated).  Let _finalize_day() — called by maybe_finalize()
        # from the hourly forecast workflow — own the reset and _current_day update.
        # If the forecast workflow misses a cycle by ≤1 h the error is negligible.

        self._elapsed_s += dt_s
        if relay_on:
            self._relay_on_s += dt_s

        # Accumulate T_in for live T_cold estimation
        if T_in is not None:
            t_in = float(T_in)
            if 0.0 <= t_in <= 30.0:   # sanity: inlet should be cold
                self._T_in_samples.append(t_in)

        just_turned_off = (self._was_relay_on is True) and (not relay_on)

        if just_turned_off and T_case is not None and T_amb is not None:
            T_c = float(T_case)
            T_a = float(T_amb)

            # T_set inference from case sensor
            T_set_sample = self._infer_T_set(T_c, T_a)
            if T_set_sample is not None:
                self._T_set_samples.append(T_set_sample)
                logger.debug("T_set sample at relay-off: %.1f°C", T_set_sample)

            # Coupling calibration: needs T_set already known
            coupling_sample = self._infer_coupling(T_c, T_a)
            if coupling_sample is not None:
                self._coupling_samples.append(coupling_sample)
                logger.debug("coupling sample at relay-off: %.3f", coupling_sample)

        # Draw detection while relay is OFF and T_case is dropping
        if (not relay_on) and T_case is not None and T_amb is not None:
            self._detect_draw(float(T_case), float(T_amb))

        self._was_relay_on = relay_on
        self._prev_T_case  = float(T_case) if T_case is not None else self._prev_T_case

    def maybe_finalize(
        self,
        k_water: Optional[float] = None,
        T_amb: Optional[float] = None,
        true_vol_L: Optional[float] = None,
    ) -> Optional[float]:
        """
        Finalize yesterday's estimate if the date has rolled over.

        Call this from the hourly forecast workflow.  Safe to call every hour;
        only acts when a day boundary has been crossed.

        Args:
            k_water:     Fitted thermal model k_water [1/h] for better standby
                         loss calculation.  Falls back to standby_w if None.
            T_amb:       Current ambient temperature [°C].
            true_vol_L:  Ground-truth daily volume [L] if a flow sensor is
                         available (Standard Mode).  Pass None otherwise.

        Returns:
            Finalized volume estimate [L] if a day was finalised, else None.
        """
        today = datetime.now().date()
        if self._current_day is None or today == self._current_day:
            return None
        return self._finalize_day(
            k_water=k_water,
            T_amb=T_amb,
            true_vol_L=true_vol_L,
        )

    def estimate_today(
        self,
        T_set_override: Optional[float] = None,
        k_water: Optional[float] = None,
        T_amb: Optional[float] = None,
    ) -> float:
        """
        Estimate hot-water consumption [L] accumulated so far today.

        Returns 0.0 if the relay has never turned on today.
        """
        T_amb_eff = float(T_amb) if T_amb is not None else self.T_amb_default
        T_set_eff = (
            float(T_set_override)
            if T_set_override is not None
            else (self.T_set_calibrated or (T_amb_eff + 30.0))
        )
        T_set_eff = max(T_set_eff, T_amb_eff + 5.0)

        relay_on_h = self._relay_on_s / 3600.0
        relay_kwh  = relay_on_h * self.relay_w / 1000.0

        elapsed_h  = max(self._elapsed_s, 1.0) / 3600.0
        if k_water is not None:
            standby_kwh = (
                k_water
                * self.vol_L * RHO * C_WATER
                * max(T_set_eff - T_amb_eff, 1.0)
                / 3.6e6
                * elapsed_h
            )
        else:
            standby_kwh = self.standby_w / 1000.0 * elapsed_h

        net_kwh   = max(relay_kwh - standby_kwh, 0.0)
        T_cold_eff = self._T_cold_live if self._T_cold_live is not None else self.T_cold
        dT_useful = max(T_set_eff - T_cold_eff, 2.0)
        vol_est   = net_kwh * 3.6e6 / (C_WATER * RHO * dT_useful) * self.alpha

        logger.debug(
            "estimate_today: relay_on=%.2fh  relay_kwh=%.3f  standby_kwh=%.3f  "
            "net=%.3f  T_set=%.1f  vol=%.1fL  alpha=%.3f",
            relay_on_h, relay_kwh, standby_kwh, net_kwh, T_set_eff, vol_est, self.alpha,
        )
        return float(max(vol_est, 0.0))

    def diagnostics(self) -> dict:
        return {
            "alpha":               round(self.alpha, 3),
            "T_set_calibrated":    round(self.T_set_calibrated, 1) if self.T_set_calibrated else None,
            "coupling_calibrated": round(self.coupling_calibrated, 3) if self.coupling_calibrated else None,
            "T_cold_live":         round(self._T_cold_live, 1) if self._T_cold_live is not None else None,
            "relay_on_h_today":    round(self._relay_on_s / 3600.0, 2),
            "draw_vol_today_L":    round(self._draw_vol_today, 1),
            "est_vol_today_L":     round(self.estimate_today(), 1),
            "T_set_samples_today": len(self._T_set_samples),
            "daily_history":       self._daily_history[-7:],
        }

    # ── Persistence ───────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "vol_L":               self.vol_L,
            "relay_w":             self.relay_w,
            "T_cold":              self.T_cold,
            "coupling":            self.coupling,
            "T_amb_default":       self.T_amb_default,
            "standby_w":           self.standby_w,
            "draw_threshold_c":    self.draw_threshold_c,
            "alpha":               self.alpha,
            "T_set_calibrated":    self.T_set_calibrated,
            "coupling_calibrated": self.coupling_calibrated,
            "T_cold_live":         self._T_cold_live,
            "daily_history":       self._daily_history,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "FlowlessConsumptionEstimator":
        obj = cls(
            vol_L            = d.get("vol_L",           100.0),
            relay_w          = d.get("relay_w",         2000.0),
            T_cold           = d.get("T_cold",            10.0),
            coupling         = d.get("coupling",           0.45),
            T_amb_default    = d.get("T_amb_default",      20.0),
            standby_w        = d.get("standby_w",          50.0),
            draw_threshold_c = d.get("draw_threshold_c",    2.0),
        )
        obj.alpha               = float(d.get("alpha", 1.0))
        obj.T_set_calibrated    = d.get("T_set_calibrated")
        obj.coupling_calibrated = d.get("coupling_calibrated")
        obj._T_cold_live        = d.get("T_cold_live")
        obj._daily_history      = d.get("daily_history", [])
        return obj

    # ── Internal helpers ──────────────────────────────────────────────────

    def _finalize_day(
        self,
        k_water: Optional[float],
        T_amb: Optional[float],
        true_vol_L: Optional[float],
    ) -> float:
        est_vol = self.estimate_today(k_water=k_water, T_amb=T_amb)

        # Update alpha with ground truth (Standard Mode with flow sensor)
        if true_vol_L is not None and true_vol_L > 0.5 and est_vol > 0.5:
            ratio = float(np.clip(true_vol_L / est_vol, _ALPHA_MIN, _ALPHA_MAX))
            old_alpha = self.alpha
            self.alpha = float(np.clip(
                self.alpha * (1 - _ALPHA_LR) + ratio * _ALPHA_LR,
                _ALPHA_MIN, _ALPHA_MAX,
            ))
            logger.info(
                "alpha updated %.3f → %.3f  (true=%.1fL  est=%.1fL)",
                old_alpha, self.alpha, true_vol_L, est_vol,
            )

        # Update T_cold_live from today's inlet-probe samples
        if self._T_in_samples:
            self._T_cold_live = float(np.median(self._T_in_samples))
            logger.debug("T_cold_live updated to %.1f°C (%d samples)", self._T_cold_live, len(self._T_in_samples))

        # Update T_set calibration from today's case-sensor samples
        if self._T_set_samples:
            T_set_today = float(np.percentile(self._T_set_samples, 90))
            T_set_today = float(np.clip(T_set_today, 30.0, 85.0))
            if self.T_set_calibrated is None:
                self.T_set_calibrated = T_set_today
            else:
                self.T_set_calibrated = self.T_set_calibrated * 0.85 + T_set_today * 0.15
            logger.info(
                "T_set_calibrated → %.1f°C  (%d samples)",
                self.T_set_calibrated, len(self._T_set_samples),
            )

        # Update coupling calibration from today's relay-OFF samples
        if self._coupling_samples:
            coupling_today = float(np.median(self._coupling_samples))
            coupling_today = float(np.clip(coupling_today, _COUPLING_MIN, _COUPLING_MAX))
            if self.coupling_calibrated is None:
                self.coupling_calibrated = coupling_today
            else:
                self.coupling_calibrated = (
                    self.coupling_calibrated * (1 - _COUPLING_LR)
                    + coupling_today * _COUPLING_LR
                )
            logger.info(
                "coupling_calibrated → %.3f  (%d samples)",
                self.coupling_calibrated, len(self._coupling_samples),
            )

        # Cross-check draw-detected volume vs energy balance → nudge alpha
        if (
            true_vol_L is None                   # no flow sensor (pure simple mode)
            and self._draw_vol_today > 2.0        # detected at least a small draw
            and est_vol > 0.5
        ):
            ratio = float(np.clip(self._draw_vol_today / est_vol, _ALPHA_MIN, _ALPHA_MAX))
            old_alpha = self.alpha
            self.alpha = float(np.clip(
                self.alpha * (1 - _DRAW_ALPHA_LR) + ratio * _DRAW_ALPHA_LR,
                _ALPHA_MIN, _ALPHA_MAX,
            ))
            logger.info(
                "alpha (draw correction) %.3f → %.3f  (draw=%.1fL  eb=%.1fL)",
                old_alpha, self.alpha, self._draw_vol_today, est_vol,
            )

        # Record in history (keep last 90 days)
        self._daily_history.append({
            "date":       (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d"),
            "est_vol_L":  round(est_vol, 1),
            "draw_vol_L": round(self._draw_vol_today, 1),
            "true_vol_L": round(true_vol_L, 1) if true_vol_L is not None else None,
            "alpha":      round(self.alpha, 3),
        })
        if len(self._daily_history) > 90:
            self._daily_history = self._daily_history[-90:]

        logger.info(
            "finalize_day: est=%.1fL  draw=%.1fL  true=%s  alpha=%.3f  T_set_cal=%s  coupling=%s",
            est_vol,
            self._draw_vol_today,
            f"{true_vol_L:.1f}L" if true_vol_L is not None else "—",
            self.alpha,
            f"{self.T_set_calibrated:.1f}°C" if self.T_set_calibrated else "—",
            f"{self._effective_coupling:.3f}",
        )

        # Reset accumulators
        self._relay_on_s       = 0.0
        self._elapsed_s        = 0.0
        self._T_set_samples    = []
        self._coupling_samples = []
        self._T_in_samples     = []
        self._draw_vol_today   = 0.0
        self._current_day      = datetime.now().date()

        return est_vol

    @property
    def _effective_coupling(self) -> float:
        """Use auto-calibrated coupling if available, else config value."""
        return self.coupling_calibrated if self.coupling_calibrated is not None else self.coupling

    def _infer_T_set(self, T_case: float, T_amb: float) -> Optional[float]:
        """
        Infer T_water from case sensor at relay-OFF transition.
        T_water = T_amb + (T_case − T_amb) / coupling
        """
        c = self._effective_coupling
        if c <= 0:
            return None
        T_water = T_amb + (T_case - T_amb) / c
        if T_water < T_amb + 2.0 or T_water > 90.0:
            return None
        return float(T_water)

    def _infer_coupling(self, T_case: float, T_amb: float) -> Optional[float]:
        """
        Infer coupling ratio at relay-OFF (T_water ≈ T_set_calibrated).
        Requires T_set_calibrated to be set (from prior relay-OFF EMA).
        coupling = (T_case − T_amb) / (T_set_calibrated − T_amb)
        """
        if self.T_set_calibrated is None:
            return None
        dT_water = self.T_set_calibrated - T_amb
        dT_case  = T_case - T_amb
        if dT_water < 2.0 or dT_case <= 0.0:
            return None
        c = dT_case / dT_water
        if c < _COUPLING_MIN or c > _COUPLING_MAX:
            return None
        return float(c)

    def _detect_draw(self, T_case: float, T_amb: float) -> None:
        """
        Detect a hot-water draw from T_case dropping while relay is OFF.

        When T_case drops by more than draw_threshold_c per tick, cold water
        entered the tank (displacing hot water out the tap).  Estimate volume:
            ΔT_water = ΔT_case / coupling
            V_draw   = vol_L × ΔT_water / (T_water_before − T_cold_eff)
        """
        if self._prev_T_case is None:
            return

        delta_case = self._prev_T_case - T_case   # positive = drop
        if delta_case < self.draw_threshold_c:
            return

        c = self._effective_coupling
        delta_water = delta_case / c if c > 0 else 0.0

        T_water_before = T_amb + (self._prev_T_case - T_amb) / c if c > 0 else 60.0
        T_cold_eff     = self._T_cold_live if self._T_cold_live is not None else self.T_cold
        denom          = max(T_water_before - T_cold_eff, 2.0)

        v_draw = self.vol_L * delta_water / denom
        v_draw = max(v_draw, 0.0)
        if v_draw > 0.1:
            self._draw_vol_today += v_draw
            logger.debug(
                "_detect_draw: ΔT_case=%.2f  ΔT_water=%.2f  V=%.1fL  total_today=%.1fL",
                delta_case, delta_water, v_draw, self._draw_vol_today,
            )
