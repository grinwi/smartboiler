# Created as a part of Master's Thesis "Using machine learning methods to save energy in a smart home"
# Faculty of Information Technology, Brno University of Technology, 2024
# Author: Adam Grünwald
#
# Home Assistant REST API client — replaces InfluxDB dependency.
# Reads entity states/history via the HA Supervisor API using SUPERVISOR_TOKEN.

import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

import requests

logger = logging.getLogger(__name__)

HA_BASE_URL = os.getenv("HA_BASE_URL", "http://supervisor/core/api")
SUPERVISOR_TOKEN = os.getenv("SUPERVISOR_TOKEN", "")


class HAClient:
    """Thin wrapper around the Home Assistant REST API."""

    def __init__(
        self,
        base_url: str = HA_BASE_URL,
        token: str = SUPERVISOR_TOKEN,
        timeout: int = 10,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._session = requests.Session()
        self._session.headers.update(
            {
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            }
        )

    def _get(self, path: str, params: Optional[Dict] = None) -> Any:
        url = f"{self.base_url}{path}"
        resp = self._session.get(url, params=params, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def _post(self, path: str, data: Optional[Dict] = None) -> Any:
        url = f"{self.base_url}{path}"
        resp = self._session.post(url, json=data or {}, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    # ── Entity state ─────────────────────────────────────────────────────

    def get_state(self, entity_id: str) -> Optional[Dict]:
        """Return the current state object for an entity, or None on error."""
        try:
            return self._get(f"/states/{entity_id}")
        except Exception as e:
            logger.warning("get_state(%s) failed: %s", entity_id, e)
            return None

    def get_state_value(self, entity_id: str, default=None):
        """Return the numeric state value of an entity, or default on error."""
        state = self.get_state(entity_id)
        if state is None:
            return default
        raw = state.get("state", "")
        try:
            return float(raw)
        except (ValueError, TypeError):
            return raw if raw not in ("unavailable", "unknown", "") else default

    def get_all_states(self, domain: Optional[str] = None) -> List[Dict]:
        """Return all entity states, optionally filtered by domain."""
        try:
            states = self._get("/states")
            if domain:
                states = [
                    s
                    for s in states
                    if s.get("entity_id", "").startswith(f"{domain}.")
                ]
            return states
        except Exception as e:
            logger.warning("get_all_states failed: %s", e)
            return []

    # ── History ───────────────────────────────────────────────────────────

    def get_history(
        self,
        entity_id: str,
        start: datetime,
        end: Optional[datetime] = None,
    ) -> List[Dict]:
        """Return list of historical state objects for an entity."""
        params: Dict[str, str] = {
            "filter_entity_id": entity_id,
            "minimal_response": "true",
        }
        if end:
            params["end_time"] = end.isoformat()
        path = f"/history/period/{start.isoformat()}"
        try:
            result = self._get(path, params=params)
            # HA returns list-of-lists; first element = our entity
            if result and isinstance(result, list) and result[0]:
                return result[0]
            return []
        except Exception as e:
            logger.warning("get_history(%s) failed: %s", entity_id, e)
            return []

    # ── Services ──────────────────────────────────────────────────────────

    def call_service(
        self, domain: str, service: str, entity_id: str, **kwargs
    ) -> bool:
        """Call an HA service (e.g. switch.turn_on)."""
        data = {"entity_id": entity_id, **kwargs}
        try:
            self._post(f"/services/{domain}/{service}", data)
            return True
        except Exception as e:
            logger.error("call_service(%s.%s, %s) failed: %s", domain, service, entity_id, e)
            return False

    def turn_on(self, entity_id: str) -> bool:
        domain = entity_id.split(".")[0]
        return self.call_service(domain, "turn_on", entity_id)

    def turn_off(self, entity_id: str) -> bool:
        domain = entity_id.split(".")[0]
        return self.call_service(domain, "turn_off", entity_id)

    # ── Convenience helpers ───────────────────────────────────────────────

    def is_entity_on(self, entity_id: str) -> Optional[bool]:
        """Return True/False for binary state, or None if unavailable."""
        state = self.get_state(entity_id)
        if state is None:
            return None
        return state.get("state") in ("on", "home", "true", "1")

    def get_attribute(self, entity_id: str, attribute: str, default=None):
        """Return a specific attribute of an entity state."""
        state = self.get_state(entity_id)
        if state is None:
            return default
        return state.get("attributes", {}).get(attribute, default)
