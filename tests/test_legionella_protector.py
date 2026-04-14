"""Tests for LegionellaProtector — focusing on the HDO stale-temperature bug.

Chunk 3 fix: check_and_act must NOT complete a legionella cycle when the relay
is HDO-blocked (relay_active=False), even if the reported temperature meets
the target. A stale cached sensor value while power=0 must not count as proof
of successful heating.

Chunk 9 fix: get_last_legionella_heating() must return a tz-aware datetime even
when the stored ISO string is tz-naive (written by an older version of the code).
"""

from datetime import datetime, timedelta
from unittest.mock import MagicMock

import pytest

from smartboiler.legionella_protector import LegionellaProtector
from smartboiler.state_store import StateStore


class TestLegionellaTimestampTzSafety:
    """Chunk 9: tz-naive stored timestamp must not cause TypeError in is_due()."""

    def test_get_last_legionella_heating_with_tz_naive_string_returns_tz_aware(self, tmp_path):
        """Stored tz-naive ISO string must be returned as tz-aware datetime."""
        store = StateStore(data_dir=str(tmp_path))
        # Simulate what an old code version would write (no UTC offset)
        store.set("last_legionella_heating", "2024-01-15T10:30:00")
        dt = store.get_last_legionella_heating()
        assert dt.tzinfo is not None, "Returned datetime must be tz-aware"

    def test_is_due_does_not_raise_with_tz_naive_stored_string(self, tmp_path):
        """is_due() must not raise TypeError when stored timestamp is tz-naive."""
        store = StateStore(data_dir=str(tmp_path))
        store.set("last_legionella_heating", "2020-01-01T00:00:00")  # tz-naive, very old
        ha = MagicMock()
        leg = LegionellaProtector(store, ha, "switch.boiler")
        # Should not raise TypeError
        result = leg.is_due()
        assert result is True  # 2020 is >21 days ago

    def test_is_due_false_with_tz_naive_recent_timestamp(self, tmp_path):
        """Recent tz-naive stored timestamp must correctly resolve to not-due."""
        from datetime import datetime as _dt
        store = StateStore(data_dir=str(tmp_path))
        # Store a recent tz-naive timestamp (yesterday)
        yesterday_naive = (_dt.now() - timedelta(days=1)).isoformat()
        store.set("last_legionella_heating", yesterday_naive)
        ha = MagicMock()
        leg = LegionellaProtector(store, ha, "switch.boiler")
        result = leg.is_due()
        assert result is False  # 1 day ago < 21-day interval


def _make_protector(days_since_last: int = 25, target_tmp: float = 65.0):
    """Return a LegionellaProtector that is overdue by `days_since_last` days."""
    store = MagicMock()
    store.get_last_legionella_heating.return_value = (
        datetime.now().astimezone() - timedelta(days=days_since_last)
    )
    ha = MagicMock()
    leg = LegionellaProtector(store, ha, "switch.boiler", target_tmp=target_tmp)
    return leg, store, ha


class TestLegionellaHDOGuard:

    def test_does_not_complete_cycle_when_relay_is_hdo_blocked(self):
        """Temperature at target but relay HDO-unavailable — cycle must NOT complete."""
        leg, store, ha = _make_protector()

        result = leg.check_and_act(65.0, relay_active=False)

        assert result is True                              # still active
        store.set_last_legionella_heating.assert_not_called()  # NOT complete
        ha.turn_off.assert_not_called()                   # relay NOT released

    def test_completes_cycle_when_relay_active_and_temp_reached(self):
        """Normal path: relay is active, temp at target → cycle completes."""
        leg, store, ha = _make_protector()

        result = leg.check_and_act(65.0, relay_active=True)

        assert result is True
        store.set_last_legionella_heating.assert_called_once()
        ha.turn_off.assert_called_once_with("switch.boiler")

    def test_does_not_complete_when_temp_below_target_even_if_relay_active(self):
        """Below target temperature — cycle stays active regardless of relay state."""
        leg, store, ha = _make_protector()

        result = leg.check_and_act(60.0, relay_active=True)

        assert result is True
        store.set_last_legionella_heating.assert_not_called()
        ha.turn_off.assert_not_called()

    def test_relay_turned_on_during_active_cycle_regardless_of_relay_state(self):
        """The relay must be commanded ON whenever the cycle is active."""
        leg, store, ha = _make_protector()

        leg.check_and_act(50.0, relay_active=False)

        ha.turn_on.assert_called_with("switch.boiler")

    def test_no_cycle_when_not_due(self):
        """Not overdue — check_and_act returns False and leaves relay alone."""
        leg, store, ha = _make_protector(days_since_last=5)

        result = leg.check_and_act(50.0, relay_active=True)

        assert result is False
        ha.turn_on.assert_not_called()

    def test_default_relay_active_true_preserves_backward_compat(self):
        """Calling check_and_act without relay_active must still work (defaults to True)."""
        leg, store, ha = _make_protector()

        # Should not raise TypeError even without the new kwarg
        result = leg.check_and_act(65.0)

        assert result is True
        store.set_last_legionella_heating.assert_called_once()
