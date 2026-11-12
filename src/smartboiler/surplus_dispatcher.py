# Real-time PV surplus dispatcher for an on/off heater.
#
# The day-ahead plan works with hourly forecasts, which are always wrong in
# detail: a cloud passes, the kettle runs, the battery fills up early. This
# dispatcher corrects for that every control tick using the *measured* grid
# export, independently of the forecast.
#
# Rules (heater power P):
#   start  — export ≥ START_SHARE × P for TICKS consecutive readings
#            (the surplus would cover most of the heater)
#   stop   — while running, grid import > STOP_IMPORT_SHARE × P for TICKS
#            consecutive readings (less than half of the heater is solar)
#   reset  — when not allowed (HDO, plan already heating, water hot, export
#            worth more than import) or the reading is missing
#
# The gap between start and stop shares plus the consecutive-reading
# requirement is the hysteresis that keeps the relay from chattering.
# While running, the export reading already has the heater's draw
# subtracted, which is why the stop rule looks at import, not export.
# When the boiler's own thermostat cuts the element the draw drops to zero,
# export rises again and the dispatcher simply stays on — harmless.

import logging
from typing import Optional

logger = logging.getLogger(__name__)

START_SHARE = 0.8         # start when surplus covers ≥ 80 % of the heater
STOP_IMPORT_SHARE = 0.5   # stop when > 50 % of the heater comes from the grid
TICKS = 2                 # consecutive readings needed to start or stop


class SurplusDispatcher:
    """Decides, tick by tick, whether surplus solar should run the heater."""

    def __init__(self, heater_w: float):
        self.heater_w = heater_w
        self.active = False
        self._start_count = 0
        self._stop_count = 0

    def update(self, export_w: Optional[float], allowed: bool) -> bool:
        """Feed one reading; return True if the heater should run on surplus.

        Args:
            export_w: measured grid export in W (negative = import), or None
                      if the sensor is unavailable.
            allowed:  False whenever surplus heating must not happen now.
        """
        if not allowed or export_w is None:
            if self.active:
                logger.info("Surplus heating stopped (not allowed or no reading)")
            self.reset()
            return False

        if self.active:
            importing = export_w < -STOP_IMPORT_SHARE * self.heater_w
            self._stop_count = self._stop_count + 1 if importing else 0
            if self._stop_count >= TICKS:
                logger.info("Surplus heating stopped: importing %.0f W", -export_w)
                self.reset()
                return False
            return True

        surplus = export_w >= START_SHARE * self.heater_w
        self._start_count = self._start_count + 1 if surplus else 0
        if self._start_count >= TICKS:
            logger.info("Surplus heating started: exporting %.0f W", export_w)
            self.active = True
            self._start_count = 0
            return True
        return False

    def reset(self) -> None:
        self.active = False
        self._start_count = 0
        self._stop_count = 0
