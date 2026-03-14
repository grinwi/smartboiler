# Created as a part of Master's Thesis "Using machine learning methods to save energy in a smart home"
# Faculty of Information Technology, Brno University of Technology, 2024
# Author: Adam Grünwald
#
# Legionella protection logic — enforces periodic heating to ≥65°C to prevent
# Legionella pneumophila growth, which thrives between 20°C and 50°C.

import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

LEGIONELLA_TARGET_TMP = 65.0       # °C — minimum temperature to kill Legionella
LEGIONELLA_INTERVAL_DAYS = 21      # days between mandatory heating cycles


class LegionellaProtector:
    """
    Enforces a periodic full-heat cycle to prevent Legionella growth.

    Usage in the control loop:
        if self.legionella.check_and_act(boiler_tmp):
            return   # legionella protection took over — skip normal plan
    """

    def __init__(
        self,
        store,               # StateStore — persists last_legionella_heating timestamp
        ha,                  # HAClient
        switch_entity_id: str,
        interval_days: int = LEGIONELLA_INTERVAL_DAYS,
        target_tmp: float = LEGIONELLA_TARGET_TMP,
    ):
        self._store = store
        self._ha = ha
        self._switch_entity_id = switch_entity_id
        self._interval = timedelta(days=interval_days)
        self._target_tmp = target_tmp

    def is_due(self) -> bool:
        """Return True if a legionella cycle is overdue."""
        last = self._store.get_last_legionella_heating()
        return (datetime.now().astimezone() - last) > self._interval

    def check_and_act(self, boiler_tmp: float) -> bool:
        """
        If a legionella cycle is due, command the relay and return True.
        Returns False when legionella protection is not active (caller continues normally).

        When the water reaches `target_tmp`, the cycle is marked complete and
        the relay is turned off.
        """
        if not self.is_due():
            return False

        logger.info(
            "Legionella protection: heating to %.0f°C (current=%.1f°C)",
            self._target_tmp, boiler_tmp,
        )
        self._ha.turn_on(self._switch_entity_id)

        if boiler_tmp >= self._target_tmp:
            now_tz = datetime.now().astimezone()
            self._store.set_last_legionella_heating(now_tz)
            self._ha.turn_off(self._switch_entity_id)
            logger.info("Legionella protection complete — cycle recorded at %s.", now_tz.isoformat())

        return True
