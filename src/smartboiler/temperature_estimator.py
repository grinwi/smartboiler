# Created as a part of Master's Thesis "Using machine learning methods to save energy in a smart home"
# Faculty of Information Technology, Brno University of Technology, 2024
# Author: Adam Grünwald
#
# Multi-level water temperature estimator for the SmartBoiler controller.
# Tries four estimation levels in order from most to least accurate.

import logging
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)


class TemperatureEstimator:
    """
    Estimates the current water temperature inside the boiler using up to four
    levels of fallback:

      L1  Direct NTC probe on the boiler (most accurate)
      L2  Power-feedback (relay ON):
            power < 50 W  → internal thermostat tripped → T = T_set
            power > 50 W  → actively heating → skip L3 (cooling model invalid
                            during active heating) → fall straight to L4
      L3  Learned thermal model from case-sensor (Newton's-law cooling).
            Only valid when relay is OFF and boiler is cooling passively.
      L4  Last known value (stale, but better than nothing)
    """

    def __init__(
        self,
        ha,                               # HAClient
        switch_entity_id: str,
        power_entity_id: Optional[str],
        case_tmp_entity_id: Optional[str],
        area_tmp_entity_id: Optional[str],
        direct_tmp_entity_id: Optional[str],
        thermal_model,                    # ThermalModel (may be None)
        boiler_set_tmp: float,
        area_tmp_default: float,
    ):
        self._ha = ha
        self._switch_entity_id = switch_entity_id
        self._power_entity_id = power_entity_id
        self._case_tmp_entity_id = case_tmp_entity_id
        self._area_tmp_entity_id = area_tmp_entity_id
        self._direct_tmp_entity_id = direct_tmp_entity_id
        self._thermal_model = thermal_model
        self._boiler_set_tmp = boiler_set_tmp
        self._area_tmp_default = area_tmp_default

    @staticmethod
    def _minutes_since(dt: Optional[datetime]) -> Optional[float]:
        """Return age in minutes for a possibly-naive datetime."""
        if not isinstance(dt, datetime):
            return None
        ref = dt
        if ref.tzinfo is None:
            ref = ref.replace(tzinfo=datetime.now().astimezone().tzinfo)
        return (datetime.now().astimezone() - ref).total_seconds() / 60.0

    def _get_ambient_context(self) -> dict:
        """Return the current ambient temperature plus source metadata."""
        if self._area_tmp_entity_id:
            val = self._ha.get_state_value(self._area_tmp_entity_id)
            if val is not None:
                return {
                    "configured": True,
                    "entity_id": self._area_tmp_entity_id,
                    "value": float(val),
                    "source": "sensor",
                    "default_value": float(self._area_tmp_default),
                }
        return {
            "configured": bool(self._area_tmp_entity_id),
            "entity_id": self._area_tmp_entity_id,
            "value": float(self._area_tmp_default),
            "source": "default",
            "default_value": float(self._area_tmp_default),
        }

    def get_ambient_tmp(self) -> float:
        """Return current ambient temperature near the boiler."""
        return self._get_ambient_context()["value"]

    def get_boiler_tmp_report(
        self,
        last_known: Optional[float] = None,
        last_known_updated_at: Optional[datetime] = None,
    ) -> dict:
        """
        Return a detailed diagnostic report describing how the estimate is made.

        This mirrors ``get_boiler_tmp()`` but adds branch/inputs metadata for
        the dashboard and tests.
        """
        direct_val = None
        if self._direct_tmp_entity_id:
            direct_val = self._ha.get_state_value(self._direct_tmp_entity_id)
            if direct_val is not None:
                direct_val = float(direct_val)

        relay_on = self._ha.is_entity_on(self._switch_entity_id)

        power = None
        if self._power_entity_id:
            power = self._ha.get_state_value(self._power_entity_id)
            if power is not None:
                power = float(power)

        case_tmp = None
        if self._case_tmp_entity_id:
            case_tmp = self._ha.get_state_value(self._case_tmp_entity_id)
            if case_tmp is not None:
                case_tmp = float(case_tmp)

        ambient = self._get_ambient_context()
        thermal_preview = self._build_thermal_model_preview(
            case_tmp=case_tmp,
            ambient_tmp=ambient["value"],
            relay_on=relay_on,
            power_w=power,
        )
        last_known_age_min = self._minutes_since(last_known_updated_at)

        inputs = {
            "switch_entity_id": self._switch_entity_id,
            "direct_tmp": direct_val,
            "power_w": power,
            "case_tmp": case_tmp,
            "ambient_tmp": ambient["value"],
            "set_tmp": float(self._boiler_set_tmp),
            "last_known": float(last_known) if last_known is not None else None,
            "last_known_updated_at": (
                last_known_updated_at.isoformat(timespec="seconds")
                if isinstance(last_known_updated_at, datetime) else None
            ),
            "last_known_age_min": (
                round(last_known_age_min, 1)
                if last_known_age_min is not None else None
            ),
        }

        report = {
            "estimate": float(last_known) if last_known is not None else None,
            "source_key": "last_known",
            "source_level": "L4",
            "source_label": "Last known value",
            "reason": "All live inputs were unavailable, so the cached value is used.",
            "warnings": [],
            "direct_sensor": {
                "configured": bool(self._direct_tmp_entity_id),
                "entity_id": self._direct_tmp_entity_id,
                "value": direct_val,
            },
            "power_feedback": {
                "configured": bool(self._power_entity_id),
                "entity_id": self._power_entity_id,
                "relay_on": relay_on,
                "power_w": power,
                "thermostat_trip_threshold_w": 50.0,
                "thermostat_tripped": bool(relay_on and power is not None and power < 50.0),
                "actively_heating": bool(relay_on and power is not None and power >= 50.0),
            },
            "case_sensor": {
                "configured": bool(self._case_tmp_entity_id),
                "entity_id": self._case_tmp_entity_id,
                "value": case_tmp,
            },
            "ambient": ambient,
            "inputs": inputs,
            "thermal_model_preview": thermal_preview,
        }

        if direct_val is not None:
            report.update({
                "estimate": direct_val,
                "source_key": "direct_sensor",
                "source_level": "L1",
                "source_label": "Direct water sensor",
                "reason": "Using the direct water temperature probe.",
            })
            report["warnings"] = self._build_warnings(
                source_key=report["source_key"],
                thermal_preview=thermal_preview,
                ambient=ambient,
                last_known=last_known,
                last_known_updated_at=last_known_updated_at,
            )
            return report

        if self._power_entity_id:
            if relay_on and power is not None:
                if power < 50:
                    report.update({
                        "estimate": float(self._boiler_set_tmp),
                        "source_key": "thermostat_tripped",
                        "source_level": "L2",
                        "source_label": "Thermostat tripped -> set temperature",
                        "reason": (
                            "Relay is ON but power is below 50 W, so the internal thermostat "
                            "likely opened and water is assumed to be at T_set."
                        ),
                    })
                    report["warnings"] = self._build_warnings(
                        source_key=report["source_key"],
                        thermal_preview=thermal_preview,
                        ambient=ambient,
                        last_known=last_known,
                        last_known_updated_at=last_known_updated_at,
                    )
                    return report
                report.update({
                    "estimate": float(last_known) if last_known is not None else None,
                    "source_key": "actively_heating_last_known",
                    "source_level": "L2",
                    "source_label": "Active heating guard -> last known value",
                    "reason": (
                        "Relay is ON and the element is drawing power. The cooling model is "
                        "invalid while heating, so the app keeps the last known water temperature."
                    ),
                })
                report["warnings"] = self._build_warnings(
                    source_key=report["source_key"],
                    thermal_preview=thermal_preview,
                    ambient=ambient,
                    last_known=last_known,
                    last_known_updated_at=last_known_updated_at,
                )
                return report

        if thermal_preview.get("usable_now") and thermal_preview.get("estimate") is not None:
            report.update({
                "estimate": float(thermal_preview["estimate"]),
                "source_key": "thermal_model",
                "source_level": "L3",
                "source_label": "Case sensor thermal model",
                "reason": (
                    thermal_preview.get("reason")
                    or "Using the case sensor and ambient temperature with the thermal model."
                ),
            })
            report["warnings"] = self._build_warnings(
                source_key=report["source_key"],
                thermal_preview=thermal_preview,
                ambient=ambient,
                last_known=last_known,
                last_known_updated_at=last_known_updated_at,
            )
            return report

        report["warnings"] = self._build_warnings(
            source_key=report["source_key"],
            thermal_preview=thermal_preview,
            ambient=ambient,
            last_known=last_known,
            last_known_updated_at=last_known_updated_at,
        )
        return report

    def get_boiler_tmp(self, last_known: Optional[float] = None) -> Optional[float]:
        """
        Return the best available water temperature estimate.

        Args:
            last_known: Cached value to return as L4 fallback when all sensors fail.
        """
        # Level 1: direct water temperature sensor
        if self._direct_tmp_entity_id:
            val = self._ha.get_state_value(self._direct_tmp_entity_id)
            if val is not None:
                return float(val)

        # Level 2 / active-heating guard.
        # When the relay is ON we have two sub-cases based on the power reading:
        #
        #   power < 50 W  → internal boiler thermostat cut the heating element
        #                   (bimetal/NTC inside the tank reached set_tmp).
        #                   The smart relay stays energised; element is open.
        #                   Water temperature must be at T_set.
        #
        #   power > 50 W  → element is actively drawing current; boiler is heating.
        #                   The thermal model (L3) is a Newton's-law COOLING equation
        #                   fitted on passive cool-down data.  It produces nonsense
        #                   results during active heating (temperature is rising, not
        #                   decaying exponentially).  Skip L3 and fall to L4.
        if self._power_entity_id:
            power = self._ha.get_state_value(self._power_entity_id)
            relay_on = self._ha.is_entity_on(self._switch_entity_id)
            if relay_on and power is not None:
                if float(power) < 50:
                    # Thermostat tripped → water is at set temperature
                    return float(self._boiler_set_tmp)
                else:
                    # Actively heating → thermal model invalid; use last known value
                    return last_known

        # Level 3: learned thermal model from case sensor
        # Only reached when the relay is OFF (boiler cooling passively).
        if self._case_tmp_entity_id and self._thermal_model is not None:
            case_tmp = self._ha.get_state_value(self._case_tmp_entity_id)
            if case_tmp is not None:
                amb = self.get_ambient_tmp()
                est = self._thermal_model.estimate_water_tmp(float(case_tmp), amb)
                if est is not None:
                    return est

        # Level 4: last known value
        return last_known

    def _build_thermal_model_preview(
        self,
        case_tmp: Optional[float],
        ambient_tmp: float,
        relay_on: Optional[bool],
        power_w: Optional[float],
    ) -> dict:
        if not self._case_tmp_entity_id:
            return {
                "configured": False,
                "available": False,
                "usable_now": False,
                "reason": "Case temperature sensor is not configured.",
            }
        if case_tmp is None:
            return {
                "configured": True,
                "available": False,
                "usable_now": False,
                "reason": "Case temperature sensor returned no value.",
            }
        if self._thermal_model is None:
            return {
                "configured": True,
                "available": False,
                "usable_now": False,
                "reason": "Thermal model is not initialised.",
            }

        if hasattr(self._thermal_model, "debug_snapshot"):
            preview = self._thermal_model.debug_snapshot(case_tmp, ambient_tmp)
        else:
            estimate = self._thermal_model.estimate_water_tmp(case_tmp, ambient_tmp)
            preview = {
                "available": estimate is not None,
                "estimate": estimate,
                "mode": "thermal_model_preview",
                "mode_label": "Thermal model preview",
                "reason": "Preview generated from estimate_water_tmp().",
                "equations": [],
                "inputs": {
                    "case_tmp": case_tmp,
                    "ambient_tmp": ambient_tmp,
                },
                "intermediates": {},
                "calibration": None,
                "model": {},
                "recent_calibrations": [],
                "calibration_point": None,
                "current_cycle_samples": [],
                "current_point": None,
                "current_cycle_max_case_tmp": None,
            }

        preview["configured"] = True
        preview["usable_now"] = not (
            relay_on
            and power_w is not None
            and power_w >= 50.0
        )
        if not preview["usable_now"]:
            preview["skip_reason"] = (
                "Relay is ON and power is above 50 W, so the cooling equation is intentionally "
                "disabled during active heating."
            )
        return preview

    def _build_warnings(
        self,
        source_key: str,
        thermal_preview: dict,
        ambient: dict,
        last_known: Optional[float],
        last_known_updated_at: Optional[datetime],
    ) -> list:
        warnings = []
        if source_key == "thermal_model" and ambient.get("source") == "default":
            warnings.append(
                "Ambient temperature is using the configured default, not a live area sensor."
            )
        if (
            source_key == "thermal_model"
            and
            thermal_preview.get("mode") == "proportional_fallback"
            and thermal_preview.get("available")
        ):
            warnings.append(
                "Thermal model is not fitted yet; the case sensor path is using a simple proportional fallback."
            )
        age_min = self._minutes_since(last_known_updated_at)
        if source_key in ("last_known", "actively_heating_last_known") and age_min is not None:
            if age_min >= 30.0 and last_known is not None:
                warnings.append(
                    f"The cached last-known water temperature is stale ({age_min:.0f} minutes old)."
                )
        elif source_key in ("last_known", "actively_heating_last_known") and last_known is not None:
            warnings.append(
                "Last-known fallback has no timestamp metadata, so staleness cannot be shown."
            )
        if (
            source_key == "actively_heating_last_known"
            and not thermal_preview.get("usable_now")
            and thermal_preview.get("configured")
        ):
            warnings.append(
                thermal_preview.get("skip_reason")
                or "Thermal model is currently not usable."
            )
        return warnings
