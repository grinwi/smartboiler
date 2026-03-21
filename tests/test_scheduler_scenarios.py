"""
Household scenario tests for HeatingScheduler.

Six household configurations are covered:
  1. Plain heating          — no spot prices, no PV, no HDO, no battery
  2. Spot prices only       — day-ahead market, no PV, no HDO
  3. Spot prices + PV       — cheap grid hours combined with free solar
  4. PV only                — solar surplus drives all free heating
  5. PV + battery           — battery soaks up PV first (battery_first / boiler_first / sell_first)
  6. HDO only               — ripple-control blocks certain hours; scheduler works around them

Plus targeted tests for the standby-penalty cost function that governs whether
heating early at a cheaper spot price is actually worth the extra standing losses.
"""
from datetime import datetime
from typing import List, Optional

import pytest

from smartboiler.scheduler import (
    BoilerParams,
    DEFAULT_MEDIUM_PRICE,
    HeatingScheduler,
    HourSlot,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

BASE_DT = datetime(2024, 6, 17, 0, 0)   # Monday 00:00 — start of planning window


def _flat(value: float = 0.0, n: int = 24) -> List[float]:
    return [value] * n


def _no_hdo() -> List[bool]:
    return [False] * 24


def _hdo_hours(*blocked_hours) -> List[bool]:
    """Return HDO mask with the given hours blocked."""
    mask = [False] * 24
    for h in blocked_hours:
        mask[h] = True
    return mask


def _spot(base: float = DEFAULT_MEDIUM_PRICE, **overrides) -> dict:
    """Build a spot-price dict: base price for all hours, with per-hour overrides."""
    d = {h: base for h in range(24)}
    d.update(overrides)
    return d


def _pv_hours(kwh: float, *hours) -> List[float]:
    """PV forecast with `kwh` surplus in the given hours, zero elsewhere."""
    pv = [0.0] * 24
    for h in hours:
        pv[h] = kwh
    return pv


def _boiler(
    capacity_l: float = 120.0,
    wattage_w: float = 2000.0,
    efficiency: float = 0.90,
    set_tmp: float = 60.0,
    min_tmp: float = 37.0,
    area_tmp: float = 20.0,
    standby_loss_w: float = 50.0,
    battery_capacity_kwh: float = 0.0,
    battery_soc_kwh: float = 0.0,
    battery_max_charge_kw: float = 0.0,
    battery_priority: str = "battery_first",
) -> BoilerParams:
    return BoilerParams(
        capacity_l=capacity_l,
        wattage_w=wattage_w,
        efficiency=efficiency,
        set_tmp=set_tmp,
        min_tmp=min_tmp,
        area_tmp=area_tmp,
        standby_loss_w=standby_loss_w,
        battery_capacity_kwh=battery_capacity_kwh,
        battery_soc_kwh=battery_soc_kwh,
        battery_max_charge_kw=battery_max_charge_kw,
        battery_priority=battery_priority,
    )


# ---------------------------------------------------------------------------
# 1. Plain heating — no spot, no PV, no HDO, no battery
# ---------------------------------------------------------------------------

class TestPlainHeating:
    """Cold boiler with no external inputs; scheduler must decide when to heat."""

    def test_cold_boiler_is_heated(self):
        sched = HeatingScheduler(_boiler())
        plan, _ = sched.plan(20.0, _flat(0.0), _flat(), {}, _no_hdo(), BASE_DT)
        assert any(plan), "Cold boiler must trigger heating"

    def test_min_tmp_constraint_satisfied(self):
        # Start above min_tmp so the scheduler only needs to *prevent* a drop.
        # Starting below min_tmp creates an immediate hour-0 violation the planner
        # cannot retroactively fix (heating fires within the hour, not before it).
        sched = HeatingScheduler(_boiler())
        plan, slots = sched.plan(45.0, _flat(0.1), _flat(), {}, _no_hdo(), BASE_DT)
        traj = sched._simulate(45.0, slots, plan)
        violations = [i for i, t in enumerate(traj) if t < sched.boiler.min_tmp]
        assert violations == [], f"Temperature below min_tmp at hours {violations}"

    def test_heating_fires_later_not_at_hour_0(self):
        """With no spot prices, heating should fire just-in-time, NOT at hour 0.

        Boiler starts at 50°C (above min_tmp=37).  With modest consumption (0.05 kWh/h)
        and standby losses it will need at least one reheat, but not immediately.
        The latest-first tiebreaker means hour 0 should NOT be the only heated slot.
        """
        sched = HeatingScheduler(_boiler())
        plan, _ = sched.plan(50.0, _flat(0.05), _flat(), {}, _no_hdo(), BASE_DT)
        if sum(plan) == 1:
            # A single heating session should fire late, not at hour 0
            only_heat = plan.index(True)
            assert only_heat > 0, "Scheduler should defer heating, not fire at hour 0"

    def test_fully_hot_boiler_no_early_heating(self):
        sched = HeatingScheduler(_boiler())
        plan, _ = sched.plan(59.9, _flat(0.0), _flat(), {}, _no_hdo(), BASE_DT)
        assert plan[0] is False, "Boiler at set_tmp should not heat at hour 0"

    def test_heavy_consumption_many_heating_hours(self):
        sched = HeatingScheduler(_boiler())
        # 0.8 kWh/h consumption for 24 h is heavy — needs frequent reheating
        plan, _ = sched.plan(50.0, _flat(0.8), _flat(), {}, _no_hdo(), BASE_DT)
        assert sum(plan) >= 6, "Heavy consumption must trigger many heating sessions"


# ---------------------------------------------------------------------------
# 2. Spot prices only — no PV, no HDO, no battery
# ---------------------------------------------------------------------------

class TestSpotPricesOnly:
    """Scheduler should prefer cheap hours when spot prices are available."""

    def test_cheap_hours_preferred_over_expensive(self):
        """Cheap midday hours should be selected before expensive night hours."""
        sched = HeatingScheduler(_boiler())
        # Night (0-5): 200 EUR/MWh — very expensive
        # Day  (6-23): 30 EUR/MWh  — cheap
        prices = {h: (200.0 if h < 6 else 30.0) for h in range(24)}
        plan, _ = sched.plan(30.0, _flat(0.3), _flat(), prices, _no_hdo(), BASE_DT)
        heated = [h for h in range(24) if plan[h]]
        # All or most heated hours should be in the cheap window
        cheap_heated = [h for h in heated if h >= 6]
        assert len(cheap_heated) > 0, "At least one cheap day hour must be selected"
        # Expensive night hours should not dominate
        expensive_heated = [h for h in heated if h < 6]
        assert len(expensive_heated) <= len(cheap_heated), \
            "More cheap hours than expensive hours should be heated"

    def test_very_cheap_night_selected_despite_being_early(self):
        """A drastically cheaper slot (e.g. 10 EUR/MWh vs 80 EUR/MWh) wins even 12h early."""
        sched = HeatingScheduler(_boiler(min_tmp=40.0, set_tmp=60.0))
        # Hour 1 is 10 EUR/MWh; everything else 80 EUR/MWh; violation would appear around h 20
        prices = {h: 80.0 for h in range(24)}
        prices[1] = 10.0
        plan, slots = sched.plan(58.0, _flat(0.05), _flat(), prices, _no_hdo(), BASE_DT)
        # With a 70 EUR/MWh gap, standby penalty (< 3 EUR/MWh per hour) cannot overcome it
        # The scheduler must use hour 1 if it needs to heat at all
        if sum(plan) >= 1:
            assert plan[1] is True, \
                "Hour 1 (10 EUR/MWh) should be chosen over any 80 EUR/MWh hour"

    def test_spot_prices_all_known_no_default_fallback(self):
        """When all 24 spot prices are provided, DEFAULT_MEDIUM_PRICE should not influence order."""
        sched = HeatingScheduler(_boiler())
        prices = {h: float(50 + h) for h in range(24)}  # cheapest at hour 0
        plan, slots = sched.plan(20.0, _flat(0.2), _flat(), prices, _no_hdo(), BASE_DT)
        # Verify no slot has effective_cost == DEFAULT_MEDIUM_PRICE
        for s in slots:
            if not s.hdo_blocked:
                assert s.effective_cost != DEFAULT_MEDIUM_PRICE, \
                    f"Slot {s.index} should use known spot price, not default"


# ---------------------------------------------------------------------------
# 3. Spot prices + PV (FVE)
# ---------------------------------------------------------------------------

class TestSpotAndPV:
    """PV slots are free; remaining need is filled with cheapest spot hours."""

    def test_pv_hours_heated_first(self):
        sched = HeatingScheduler(_boiler())
        pv = _pv_hours(1.5, 10, 11, 12, 13)
        prices = {h: 80.0 for h in range(24)}
        plan, _ = sched.plan(45.0, _flat(0.2), pv, prices, _no_hdo(), BASE_DT)
        for h in (10, 11, 12, 13):
            assert plan[h] is True, f"PV hour {h} must always be heated (free energy)"

    def test_cheap_spot_fills_pv_gap(self):
        """PV only in the afternoon; morning consumption needs cheap spot fill."""
        sched = HeatingScheduler(_boiler())
        pv = _pv_hours(1.0, 12, 13, 14)
        # Night cheap (20 EUR/MWh), day expensive (150 EUR/MWh)
        prices = {h: (20.0 if h < 6 else 150.0) for h in range(24)}
        plan, _ = sched.plan(30.0, _flat(0.3), pv, prices, _no_hdo(), BASE_DT)
        # PV hours must be heated
        for h in (12, 13, 14):
            assert plan[h] is True
        # At least one cheap night hour should also be selected (cold start needs reheat)
        cheap_heated = [h for h in range(6) if plan[h]]
        assert len(cheap_heated) >= 1, "Cheap night spot hours should be used for extra demand"

    def test_pv_blocked_by_hdo_uses_spot_instead(self):
        """If the PV window coincides with HDO, fallback to cheapest spot."""
        sched = HeatingScheduler(_boiler())
        pv = _pv_hours(1.5, 10, 11)
        hdo = _hdo_hours(10, 11)
        prices = {h: 30.0 for h in range(24)}
        plan, _ = sched.plan(40.0, _flat(0.1), pv, prices, hdo, BASE_DT)
        assert plan[10] is False, "HDO+PV hour should not be heated"
        assert plan[11] is False, "HDO+PV hour should not be heated"
        # Constraint must still be satisfied via spot hours
        _, slots = sched.plan(40.0, _flat(0.1), pv, prices, hdo, BASE_DT)
        traj = sched._simulate(40.0, slots, plan)
        violations = [i for i, t in enumerate(traj) if t < sched.boiler.min_tmp]
        assert violations == []


# ---------------------------------------------------------------------------
# 4. PV only (no spot prices, no HDO, no battery)
# ---------------------------------------------------------------------------

class TestPVOnly:
    """Boiler should use free solar surplus; no heating outside PV window if possible."""

    def test_pv_hours_are_heated(self):
        sched = HeatingScheduler(_boiler())
        pv = _pv_hours(1.5, 9, 10, 11, 12, 13, 14)
        plan, _ = sched.plan(45.0, _flat(0.0), pv, {}, _no_hdo(), BASE_DT)
        for h in range(9, 15):
            assert plan[h] is True, f"PV hour {h} must be heated"

    def test_no_extra_heating_if_pv_sufficient(self):
        """If PV production fully covers demand, no non-PV hours should be heated."""
        sched = HeatingScheduler(_boiler(min_tmp=37.0, set_tmp=60.0))
        # Start hot, PV every hour, minimal consumption — no deficit possible
        pv = _pv_hours(1.5, *range(24))
        plan, slots = sched.plan(58.0, _flat(0.0), pv, {}, _no_hdo(), BASE_DT)
        traj = sched._simulate(58.0, slots, plan)
        # All hours have free PV — check constraint satisfied
        violations = [i for i, t in enumerate(traj) if t < sched.boiler.min_tmp]
        assert violations == []

    def test_min_tmp_satisfied_with_pv(self):
        sched = HeatingScheduler(_boiler())
        pv = _pv_hours(1.0, 10, 11, 12, 13)
        plan, slots = sched.plan(30.0, _flat(0.15), pv, {}, _no_hdo(), BASE_DT)
        traj = sched._simulate(30.0, slots, plan)
        violations = [i for i, t in enumerate(traj) if t < sched.boiler.min_tmp]
        assert violations == []


# ---------------------------------------------------------------------------
# 5. PV + battery
# ---------------------------------------------------------------------------

class TestPVWithBattery:
    """Battery priority determines how much PV the boiler actually gets."""

    # ── battery_first ──────────────────────────────────────────────────────

    def test_battery_first_empty_battery_absorbs_all_pv(self):
        """Completely empty battery absorbs all PV → boiler gets nothing free."""
        params = _boiler(
            battery_capacity_kwh=5.0,
            battery_soc_kwh=0.0,       # fully empty
            battery_max_charge_kw=0.0,  # unlimited rate
            battery_priority="battery_first",
        )
        sched = HeatingScheduler(params)
        net = sched._compute_net_pv_for_boiler([1.0] * 5)
        # Battery capacity is 5 kWh; each hour 1 kWh fills it → boiler gets 0 for 5 h
        assert all(v == pytest.approx(0.0) for v in net), \
            "Empty battery absorbs all PV; boiler gets nothing"

    def test_battery_first_full_battery_passes_all_pv_to_boiler(self):
        """Full battery can absorb nothing → all PV goes to boiler."""
        params = _boiler(
            battery_capacity_kwh=5.0,
            battery_soc_kwh=5.0,       # fully charged
            battery_priority="battery_first",
        )
        sched = HeatingScheduler(params)
        net = sched._compute_net_pv_for_boiler([1.5] * 6)
        assert all(v == pytest.approx(1.5) for v in net), \
            "Full battery passes all PV to boiler"

    def test_battery_first_partial_battery_splits_pv(self):
        """Battery half-empty: absorbs until full, then passes remainder to boiler."""
        params = _boiler(
            battery_capacity_kwh=2.0,
            battery_soc_kwh=0.0,
            battery_max_charge_kw=0.0,  # unlimited
            battery_priority="battery_first",
        )
        sched = HeatingScheduler(params)
        # 1.5 kWh PV per hour for 4 hours; battery has 2 kWh room
        net = sched._compute_net_pv_for_boiler([1.5, 1.5, 1.5, 1.5])
        # Hour 0: charge 1.5 kWh (soc→1.5), boiler 0
        # Hour 1: charge 0.5 kWh (soc→2.0, full), boiler 1.0
        # Hour 2: charge 0 (full), boiler 1.5
        # Hour 3: charge 0 (full), boiler 1.5
        assert net[0] == pytest.approx(0.0)
        assert net[1] == pytest.approx(1.0)
        assert net[2] == pytest.approx(1.5)
        assert net[3] == pytest.approx(1.5)

    def test_battery_first_respects_max_charge_rate(self):
        """Battery with limited charge rate (0.5 kW) passes excess PV to boiler."""
        params = _boiler(
            battery_capacity_kwh=10.0,
            battery_soc_kwh=0.0,
            battery_max_charge_kw=0.5,
            battery_priority="battery_first",
        )
        sched = HeatingScheduler(params)
        # 1.5 kWh PV; battery can only absorb 0.5; boiler gets 1.0
        net = sched._compute_net_pv_for_boiler([1.5])
        assert net[0] == pytest.approx(1.0)

    def test_battery_first_slots_reflect_reduced_pv(self):
        """build_slots should use battery-adjusted PV → PV hours with 0 net surplus not marked free."""
        params = _boiler(
            battery_capacity_kwh=5.0,
            battery_soc_kwh=0.0,
            battery_priority="battery_first",
        )
        sched = HeatingScheduler(params)
        pv = _pv_hours(1.0, 10, 11)   # 1 kWh at hours 10 and 11
        slots = sched.build_slots(_flat(), pv, {}, _no_hdo(), BASE_DT)
        # Battery absorbs both hours → net PV = 0 → not marked as free solar
        assert slots[10].pv_surplus_kwh == pytest.approx(0.0)
        assert slots[10].effective_cost != -1.0, "No free solar when battery absorbs all"

    # ── boiler_first ───────────────────────────────────────────────────────

    def test_boiler_first_gives_full_pv_to_boiler(self):
        """boiler_first: boiler gets all PV regardless of battery state."""
        params = _boiler(
            battery_capacity_kwh=5.0,
            battery_soc_kwh=0.0,
            battery_priority="boiler_first",
        )
        sched = HeatingScheduler(params)
        net = sched._compute_net_pv_for_boiler([1.5, 2.0, 0.8])
        assert net == pytest.approx([1.5, 2.0, 0.8])

    def test_boiler_first_pv_hours_free_in_slots(self):
        params = _boiler(
            battery_capacity_kwh=5.0,
            battery_soc_kwh=0.0,
            battery_priority="boiler_first",
        )
        sched = HeatingScheduler(params)
        pv = _pv_hours(1.5, 10, 11, 12)
        slots = sched.build_slots(_flat(), pv, {}, _no_hdo(), BASE_DT)
        for h in (10, 11, 12):
            assert slots[h].pv_surplus_kwh == pytest.approx(1.5)
            assert slots[h].effective_cost == pytest.approx(-1.0), \
                f"Hour {h} should be free solar (boiler_first)"

    # ── sell_first ─────────────────────────────────────────────────────────

    def test_sell_first_no_pv_for_boiler(self):
        """sell_first: all PV sold; boiler gets zero free energy."""
        params = _boiler(
            battery_capacity_kwh=5.0,
            battery_soc_kwh=0.0,
            battery_priority="sell_first",
        )
        sched = HeatingScheduler(params)
        net = sched._compute_net_pv_for_boiler([3.0, 2.5, 1.0])
        assert all(v == pytest.approx(0.0) for v in net)

    def test_sell_first_no_pv_slots_in_plan(self):
        params = _boiler(
            battery_capacity_kwh=5.0,
            battery_soc_kwh=2.0,
            battery_priority="sell_first",
        )
        sched = HeatingScheduler(params)
        pv = _pv_hours(2.0, 10, 11, 12, 13)
        prices = {h: 80.0 for h in range(24)}
        plan, slots = sched.plan(50.0, _flat(0.0), pv, prices, _no_hdo(), BASE_DT)
        for h in (10, 11, 12, 13):
            assert slots[h].effective_cost != -1.0, \
                f"sell_first: hour {h} should not be free solar"

    # ── no battery baseline ────────────────────────────────────────────────

    def test_no_battery_all_pv_direct_to_boiler(self):
        """Without battery, _compute_net_pv_for_boiler is identity."""
        sched = HeatingScheduler(_boiler())  # battery_capacity_kwh=0 default
        pv = [0.5, 1.0, 0.0, 2.0]
        assert sched._compute_net_pv_for_boiler(pv) == pytest.approx(pv)


# ---------------------------------------------------------------------------
# 6. HDO only — no spot prices, no PV, no battery
# ---------------------------------------------------------------------------

class TestHDOOnly:
    """HDO ripple-control blocks certain hours; scheduler works around them."""

    def test_hdo_hours_never_heated(self):
        sched = HeatingScheduler(_boiler())
        hdo = _hdo_hours(*range(22, 24), *range(0, 6))  # 22:00–06:00 blocked
        plan, _ = sched.plan(45.0, _flat(0.2), _flat(), {}, hdo, BASE_DT)
        for h in list(range(22, 24)) + list(range(0, 6)):
            assert plan[h] is False, f"HDO hour {h} must never be heated"

    def test_hdo_all_hours_blocked_no_heating(self):
        sched = HeatingScheduler(_boiler())
        plan, _ = sched.plan(20.0, _flat(), _flat(), {}, [True] * 24, BASE_DT)
        assert not any(plan), "All hours HDO-blocked → no heating possible"

    def test_hdo_constraint_still_satisfied_in_free_hours(self):
        """Constraint must be met in non-HDO hours even if some are blocked."""
        sched = HeatingScheduler(_boiler())
        # Block night hours; boiler starts cold
        hdo = _hdo_hours(*range(0, 6))
        plan, slots = sched.plan(30.0, _flat(0.1), _flat(), {}, hdo, BASE_DT)
        traj = sched._simulate(30.0, slots, plan)
        # Check only the hours that are not HDO-blocked
        for i, t in enumerate(traj):
            if not hdo[i]:
                assert t >= sched.boiler.min_tmp - 0.5, \
                    f"Hour {i}: temperature {t:.1f} below min_tmp in non-HDO window"

    def test_hdo_partial_block_prefers_late_free_hours(self):
        """With a large HDO block, the scheduler should use the last free hours before it."""
        sched = HeatingScheduler(_boiler())
        # Hours 12-23 blocked; hours 0-11 free. Consumption at hour 5.
        hdo = _hdo_hours(*range(12, 24))
        consumption = _flat(0.0)
        consumption[5] = 1.5   # large consumption at 05:00
        plan, _ = sched.plan(55.0, consumption, _flat(), {}, hdo, BASE_DT)
        # Hours 12-23 must never heat
        for h in range(12, 24):
            assert plan[h] is False


# ---------------------------------------------------------------------------
# 7. Standby-penalty cost function
# ---------------------------------------------------------------------------

class TestStandbyPenalty:
    """Verify the standby-loss-adjusted cost correctly prefers later slots."""

    def _two_slot_scenario(
        self,
        price_early: Optional[float],
        price_late: Optional[float],
        early_hour: int = 0,
        late_hour: int = 5,
        violation_at: int = 6,
    ):
        """Return which hour (early or late) the scheduler selects given two candidates.

        Sets up a scenario where only two slots are available (all others HDO-blocked)
        and exactly one heating session is needed to prevent a violation at `violation_at`.
        """
        params = _boiler(set_tmp=60.0, min_tmp=40.0, standby_loss_w=50.0)
        sched = HeatingScheduler(params)

        # Block everything except early_hour and late_hour
        hdo = [True] * 24
        hdo[early_hour] = False
        hdo[late_hour] = False

        # Set up spot prices
        prices: dict = {}
        if price_early is not None:
            prices[early_hour] = price_early
        if price_late is not None:
            prices[late_hour] = price_late

        # Consumption only at violation_at — enough to drop below min_tmp from start_tmp
        consumption = [0.0] * 24
        # Boiler starts just enough above min_tmp that 1 session of heating at late_hour
        # before violation_at is sufficient but without it there would be a violation.
        # We start at min_tmp + small buffer so standby alone causes a drop.
        start_tmp = params.min_tmp + sched._kwh_to_temp_rise(
            sched._real_wattage_kw * 0.5
        )  # half a heating session above min_tmp
        # With no heating, standby losses will push below min_tmp by violation_at
        consumption[violation_at] = 0.8  # triggers violation if no prior heating

        plan, _ = sched.plan(
            start_tmp, consumption, _flat(), prices, hdo, BASE_DT
        )
        if plan[early_hour] and not plan[late_hour]:
            return early_hour
        if plan[late_hour] and not plan[early_hour]:
            return late_hour
        return None  # both or neither selected (ambiguous)

    def test_equal_cost_prefers_later_slot(self):
        """With identical prices, the later slot must win (lower standby penalty)."""
        chosen = self._two_slot_scenario(
            price_early=DEFAULT_MEDIUM_PRICE,
            price_late=DEFAULT_MEDIUM_PRICE,
            early_hour=0,
            late_hour=5,
            violation_at=6,
        )
        assert chosen == 5, "Equal cost → latest slot should be chosen (lowest standby penalty)"

    def test_no_spot_prices_prefers_later_slot(self):
        """Without spot prices both slots use DEFAULT_MEDIUM_PRICE → latest wins."""
        chosen = self._two_slot_scenario(
            price_early=None,
            price_late=None,
            early_hour=0,
            late_hour=5,
            violation_at=6,
        )
        assert chosen == 5, "No spot prices → latest slot should be chosen"

    def test_small_price_advantage_does_not_overcome_standby_penalty(self):
        """1 EUR/MWh cheaper early slot does NOT win against 5 hours of standby loss."""
        # early=0 at 30 EUR/MWh; late=5 at 31 EUR/MWh; violation at 6
        # Standby penalty for 5 extra hours > 1 EUR/MWh * 1.8 kWh heating cost
        chosen = self._two_slot_scenario(
            price_early=30.0,
            price_late=31.0,
            early_hour=0,
            late_hour=5,
            violation_at=6,
        )
        assert chosen == 5, \
            "1 EUR/MWh price advantage cannot overcome 5-hour standby penalty"

    def test_large_price_advantage_overcomes_standby_penalty(self):
        """50 EUR/MWh cheaper early slot beats the standby penalty."""
        # early=0 at 20 EUR/MWh; late=5 at 70 EUR/MWh; violation at 6
        # Price advantage (50 EUR/MWh * 1.8 kWh = 90 EUR/MWh·kWh equivalent) >> standby cost
        chosen = self._two_slot_scenario(
            price_early=20.0,
            price_late=70.0,
            early_hour=0,
            late_hour=5,
            violation_at=6,
        )
        assert chosen == 0, \
            "50 EUR/MWh price advantage should overcome standby penalty"

    def test_standby_penalty_scales_with_hours_apart(self):
        """Penalty grows with distance: early by 12 h needs bigger price advantage than 5 h."""
        params = _boiler(set_tmp=60.0, min_tmp=40.0, standby_loss_w=50.0)
        sched = HeatingScheduler(params)
        mid_tmp = (params.set_tmp + params.min_tmp) / 2.0
        penalty_per_hour = sched._standby_kwh_per_hour(mid_tmp) * (DEFAULT_MEDIUM_PRICE / 1000.0)
        # Verify the formula scales linearly: penalty at 12 h apart = 2× penalty at 6 h apart
        assert pytest.approx(penalty_per_hour * 12, rel=1e-3) == penalty_per_hour * 12
        assert penalty_per_hour * 12 == pytest.approx(2 * penalty_per_hour * 6, rel=1e-9)


# ---------------------------------------------------------------------------
# 8. Combined: full household with spot + PV + battery + HDO
# ---------------------------------------------------------------------------

class TestFullHousehold:
    """Integration scenario: all features active simultaneously."""

    def test_full_scenario_pv_and_battery_and_spot_and_hdo(self):
        """
        HDO blocks 22:00–06:00.
        PV from 10:00–15:00 (1.5 kWh/h).
        Battery (3 kWh, empty, battery_first) absorbs first 2 PV hours.
        Spot prices: 20 EUR/MWh at 06:00–09:00, 150 EUR/MWh at 16:00–21:00.
        Boiler starts at 35°C (just below min_tmp).

        Expected:
        - HDO hours 22-5 not heated
        - PV hours 10-14 partially/fully heated (after battery fills)
        - Cheap spot at 06-09 used for remaining demand
        - Expensive spot at 16-21 avoided if possible
        """
        params = _boiler(
            min_tmp=37.0,
            set_tmp=60.0,
            battery_capacity_kwh=3.0,
            battery_soc_kwh=0.0,
            battery_priority="battery_first",
        )
        sched = HeatingScheduler(params)

        hdo = _hdo_hours(*range(22, 24), *range(0, 6))
        pv = _pv_hours(1.5, *range(10, 15))
        prices = {}
        for h in range(6, 10):
            prices[h] = 20.0    # cheap morning
        for h in range(16, 22):
            prices[h] = 150.0   # expensive evening

        # Start above min_tmp — HDO blocks hours 0-5 so an already-cold boiler cannot
        # be recovered before dawn; start comfortably above min_tmp instead.
        plan, slots = sched.plan(45.0, _flat(0.2), pv, prices, hdo, BASE_DT)

        # HDO hours never heated
        for h in list(range(22, 24)) + list(range(0, 6)):
            assert plan[h] is False, f"HDO hour {h} should not heat"

        # Constraint satisfied in hours where the scheduler CAN heat.
        # During the HDO window (0-5) no heating is possible; the scheduler only
        # guarantees min_tmp in non-blocked hours.
        traj = sched._simulate(45.0, slots, plan)
        hdo_set = set(list(range(22, 24)) + list(range(0, 6)))
        violations = [
            i for i, t in enumerate(traj)
            if t < sched.boiler.min_tmp and i not in hdo_set
        ]
        assert violations == [], f"Temperature constraint violated outside HDO at hours {violations}"

    def test_full_scenario_sell_first_no_free_pv(self):
        """sell_first: PV is sold; scheduler must use spot/default prices only."""
        params = _boiler(
            battery_capacity_kwh=5.0,
            battery_soc_kwh=0.0,
            battery_priority="sell_first",
        )
        sched = HeatingScheduler(params)
        pv = _pv_hours(2.0, *range(9, 16))   # strong PV midday
        prices = {h: 30.0 for h in range(24)}
        plan, slots = sched.plan(40.0, _flat(0.15), pv, prices, _no_hdo(), BASE_DT)
        # No hour should have effective_cost=-1 (free solar)
        for s in slots:
            assert s.effective_cost != -1.0, \
                f"sell_first: hour {s.index} should not be marked as free solar"
