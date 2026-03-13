# Created as a part of Master's Thesis "Using machine learning methods to save energy in a smart home"
# Faculty of Information Technology, Brno University of Technology, 2024
# Author: Adam Grünwald
#
# Google Calendar integration via HA Calendar entity.
#
# Recognised event title keywords (case-insensitive, multilingual):
#   vacation / dovolená / holiday  → heat only to vacation_min_temp (or off)
#   off / boiler off               → relay off; only legionella can override
#   boost / boost 65 / ohřev      → heat to set_tmp (or specific °C)
#
# Events can be created directly in Google Calendar, or via the web dashboard.
# HA Calendar REST API: GET /api/calendars/{entity_id}?start=...&end=...
# HA Create service  : POST /api/services/calendar/create_event  (HA ≥ 2023.11)
# HA Delete service  : POST /api/services/calendar/delete_event  (HA ≥ 2024.6)

import logging
import re
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, List, Optional

logger = logging.getLogger(__name__)

CACHE_TTL_S = 300  # refresh calendar at most every 5 minutes

# ── Keyword tables ──────────────────────────────────────────────────────────

_VACATION_KW = [
    "vacation", "dovolena", "dovolená", "holiday",
    "prázdniny", "prazdniny", "abwesend",
]
_OFF_KW = [
    "boiler off", "kotel off", "off mode",
    "legionella only", "off: boiler", "boiler: off",
]
_BOOST_KW = [
    "boost", "ohrev", "ohřev", "force heat",
    "nahrev", "nahřev", "aufheizen",
]


# ── Data model ──────────────────────────────────────────────────────────────

@dataclass
class BoilerEvent:
    id: str
    summary: str
    start: datetime         # local, tz-naive
    end: datetime           # local, tz-naive
    # vacation_min | vacation_off | boost_max | boost_temp
    event_type: str
    target_temp: Optional[float] = None

    def is_active_at(self, dt: datetime) -> bool:
        return self.start <= dt < self.end

    def covers_hour(self, slot_dt: datetime) -> bool:
        """True if this event overlaps the clock-hour beginning at slot_dt."""
        return self.start < slot_dt + timedelta(hours=1) and self.end > slot_dt


# ── Parsing ─────────────────────────────────────────────────────────────────

def _to_local_naive(raw: str) -> Optional[datetime]:
    """Convert an ISO-8601 date or dateTime string to a local, tz-naive datetime."""
    if not raw:
        return None
    if "T" not in raw:          # all-day event: 'date' field
        raw += "T00:00:00"
    raw = raw.replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(raw)
    except ValueError:
        return None
    if dt.tzinfo is not None:
        utc_offset_s = time.localtime().tm_gmtoff
        local_naive = dt.replace(tzinfo=None) + timedelta(seconds=utc_offset_s) - dt.utcoffset()
        return local_naive
    return dt


def _parse_ha_event(ha_event: dict) -> Optional[BoilerEvent]:
    """Convert a raw HA calendar event dict into a BoilerEvent, or None if unrecognised."""
    try:
        summary = ha_event.get("summary", "") or ""
        s = summary.lower()

        start_b = ha_event.get("start", {})
        end_b   = ha_event.get("end",   {})
        start = _to_local_naive(start_b.get("dateTime") or start_b.get("date", ""))
        end   = _to_local_naive(end_b.get("dateTime")   or end_b.get("date", ""))
        if start is None or end is None:
            return None

        uid = ha_event.get("uid") or ha_event.get("id") or f"{summary}@{start.isoformat()}"

        if any(kw in s for kw in _OFF_KW):
            return BoilerEvent(id=uid, summary=summary, start=start, end=end,
                               event_type="vacation_off")

        if any(kw in s for kw in _VACATION_KW):
            return BoilerEvent(id=uid, summary=summary, start=start, end=end,
                               event_type="vacation_min")

        if any(kw in s for kw in _BOOST_KW):
            # Extract optional temperature: "Boost 65", "boost:65°C", "ohřev 70 C"
            m = re.search(r'\b(\d{2,3})\s*°?\s*[cC]?\b', s)
            if m:
                temp = float(m.group(1))
                if 30.0 <= temp <= 95.0:
                    return BoilerEvent(id=uid, summary=summary, start=start, end=end,
                                       event_type="boost_temp", target_temp=temp)
            return BoilerEvent(id=uid, summary=summary, start=start, end=end,
                               event_type="boost_max")

        return None   # not a SmartBoiler event — ignore
    except Exception as e:
        logger.debug("_parse_ha_event error: %s", e)
        return None


# ── Manager ─────────────────────────────────────────────────────────────────

class CalendarManager:
    """
    Thread-safe bridge between the SmartBoiler controller / scheduler and the
    HA calendar entity.  Caches events for CACHE_TTL_S seconds to avoid
    repeated HTTP calls from the control loop.
    """

    def __init__(
        self,
        ha_client: Any,
        calendar_entity_id: str,
        vacation_mode: str = "min_temp",  # "min_temp" | "off"
    ) -> None:
        self._ha = ha_client
        self._entity_id = calendar_entity_id
        self._vacation_mode = vacation_mode
        self._cache: List[BoilerEvent] = []
        self._cache_until: float = 0.0
        self._lock = threading.Lock()

    # ── Internal fetch / cache ───────────────────────────────────────────────

    def _fetch_raw(self, start: datetime, end: datetime) -> List[BoilerEvent]:
        try:
            raw = self._ha._get(
                f"/calendars/{self._entity_id}",
                params={
                    "start": start.strftime("%Y-%m-%dT%H:%M:%S"),
                    "end":   end.strftime("%Y-%m-%dT%H:%M:%S"),
                },
            )
            result = []
            for item in (raw or []):
                evt = _parse_ha_event(item)
                if evt is not None:
                    result.append(evt)
            return result
        except Exception as e:
            logger.warning("Calendar fetch failed (%s): %s", self._entity_id, e)
            return []

    def _refresh_cache(self) -> None:
        """Fetch a 7-day window; must NOT be called while holding self._lock."""
        now = datetime.now().astimezone()
        events = self._fetch_raw(now - timedelta(hours=1), now + timedelta(days=7))
        with self._lock:
            self._cache = events
            self._cache_until = time.monotonic() + CACHE_TTL_S
        logger.debug("Calendar cache refreshed: %d events", len(events))

    def _get_cached(self) -> List[BoilerEvent]:
        with self._lock:
            if time.monotonic() < self._cache_until:
                return list(self._cache)
        self._refresh_cache()
        with self._lock:
            return list(self._cache)

    # ── Public API ───────────────────────────────────────────────────────────

    def get_events(self, start: datetime, end: datetime) -> List[BoilerEvent]:
        """Return all BoilerEvents whose interval overlaps [start, end)."""
        all_evts = self._get_cached()
        return [e for e in all_evts if e.start < end and e.end > start]

    def get_active_event(self, dt: datetime) -> Optional[BoilerEvent]:
        """
        Return the highest-priority BoilerEvent active at dt, or None.

        Priority: vacation_off > boost > vacation_min
        (legionella is handled by the controller before this is called)
        """
        events = self.get_events(dt, dt + timedelta(seconds=1))
        active = [e for e in events if e.is_active_at(dt)]
        if not active:
            return None
        _priority = {"vacation_off": 0, "boost_max": 1, "boost_temp": 1, "vacation_min": 2}
        return min(active, key=lambda e: _priority.get(e.event_type, 99))

    def create_event(
        self,
        event_type: str,
        start: datetime,
        end: datetime,
        target_temp: Optional[float] = None,
    ) -> bool:
        """Create a boiler event in the HA calendar.  Returns True on success."""
        _labels = {
            "vacation_min": "SmartBoiler: Vacation",
            "vacation_off": "SmartBoiler: Off",
            "boost_max":    "SmartBoiler: Boost",
            "boost_temp":   f"SmartBoiler: Boost {int(target_temp or 0)}\u00b0C",
        }
        summary = _labels.get(event_type, f"SmartBoiler: {event_type}")
        try:
            self._ha._post(
                "/services/calendar/create_event",
                {
                    "entity_id":       self._entity_id,
                    "summary":         summary,
                    "start_date_time": start.strftime("%Y-%m-%d %H:%M:%S"),
                    "end_date_time":   end.strftime("%Y-%m-%d %H:%M:%S"),
                },
            )
            self._invalidate()
            logger.info("Calendar event created: %s %s–%s", summary, start, end)
            return True
        except Exception as e:
            logger.error("create_event failed: %s", e)
            return False

    def delete_event(self, event_id: str) -> bool:
        """Delete an event by its uid.  Returns True on success."""
        try:
            self._ha._post(
                "/services/calendar/delete_event",
                {"entity_id": self._entity_id, "uid": event_id},
            )
            self._invalidate()
            logger.info("Calendar event deleted: %s", event_id)
            return True
        except Exception as e:
            logger.error("delete_event(%s) failed: %s", event_id, e)
            return False

    def _invalidate(self) -> None:
        with self._lock:
            self._cache_until = 0.0

    def upcoming_events_json(self, days: int = 7) -> list:
        """Serialisable list of upcoming events for the web dashboard."""
        now = datetime.now().astimezone()
        events = self.get_events(now - timedelta(hours=1), now + timedelta(days=days))
        _type_labels = {
            "vacation_min": "Vacation (min temp)",
            "vacation_off": "Off (legionella only)",
            "boost_max":    "Boost (max)",
            "boost_temp":   "Boost (target °C)",
        }
        return [
            {
                "id":          e.id,
                "summary":     e.summary,
                "type":        e.event_type,
                "type_label":  _type_labels.get(e.event_type, e.event_type),
                "target_temp": e.target_temp,
                "start":       e.start.strftime("%Y-%m-%d %H:%M"),
                "end":         e.end.strftime("%Y-%m-%d %H:%M"),
                "active":      e.is_active_at(now),
            }
            for e in sorted(events, key=lambda e: e.start)
        ]
