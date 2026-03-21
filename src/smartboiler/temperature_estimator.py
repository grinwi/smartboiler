# Created as a part of Master's Thesis "Using machine learning methods to save energy in a smart home"
# Faculty of Information Technology, Brno University of Technology, 2024
# Author: Adam Grünwald
#
# Multi-level water temperature estimator for the SmartBoiler controller.
# Tries four estimation levels in order from most to least accurate.

import logging
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

    def get_ambient_tmp(self) -> float:
        """Return current ambient temperature near the boiler."""
        if self._area_tmp_entity_id:
            val = self._ha.get_state_value(self._area_tmp_entity_id)
            if val is not None:
                return float(val)
        return self._area_tmp_default

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
