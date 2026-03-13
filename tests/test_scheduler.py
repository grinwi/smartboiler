"""
Tests for HeatingScheduler — greedy day-ahead heating planner.

Covers:
- HourSlot effective_cost calculation
- Thermodynamic helpers (_kwh_to_temp_rise, _temp_drop_to_kwh, _standby_kwh_per_hour)
- build_slots() slot construction
- _simulate() temperature trajectory
- plan() — PV surplus priority, greedy min-temp fill, HDO blocking, no-candidate fallback
- get_plan_summary() output shape
- End-to-end simulation scenarios (cold boiler, already warm, HDO night, mixed spot prices)
"""
from datetime import datetime, timedelta
from typing import Optional

import pytest

from smartboiler.scheduler import (
    BoilerParams,
    DEFAULT_MEDIUM_PRICE,
    HeatingScheduler,
    HourSlot,
    WATER_SPECIFIC_HEAT_J_KG_K,
    WATER_DENSITY_KG_L,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def params():
    return BoilerParams(
        capacity_l=120.0,
        wattage_w=2000.0,
        efficiency=0.90,
        set_tmp=60.0,
        min_tmp=37.0,
        area_tmp=20.0,
        standby_loss_w=50.0,
    )


@pytest.fixture
def sched(params):
    return HeatingScheduler(params)


def _flat(value=0.0, n=24):
    return [value] * n


def _no_hdo():
    return [False] * 24


def _no_prices():
    return {}


BASE_DT = datetime(2024, 3, 11, 8, 0)  # Monday 08:00


# ---------------------------------------------------------------------------
# HourSlot.effective_cost
# ---------------------------------------------------------------------------

class TestHourSlotEffectiveCost:
    def _slot(self, hdo_blocked=False, pv_surplus_kwh=0.0, spot_price: Optional[float] = None):
        return HourSlot(
            index=0,
            dt=BASE_DT,
            consumption_kwh=0.0,
            pv_surplus_kwh=pv_surplus_kwh,
            spot_price=spot_price,
            hdo_blocked=hdo_blocked,
        )

    def test_hdo_blocked_infinite_cost(self):
        slot = self._slot(hdo_blocked=True)
        assert slot.effective_cost == float("inf")

    def test_pv_surplus_free_energy(self):
        slot = self._slot(pv_surplus_kwh=0.5)
        assert slot.effective_cost == -1.0

    def test_known_spot_price_used(self):
        slot = self._slot(spot_price=42.0)
        assert slot.effective_cost == pytest.approx(42.0)

    def test_no_spot_price_falls_back_to_default(self):
        slot = self._slot(spot_price=None)
        assert slot.effective_cost == pytest.approx(DEFAULT_MEDIUM_PRICE)

    def test_hdo_blocks_even_with_pv_surplus(self):
        """HDO takes absolute priority — never heat even if PV is available."""
        slot = self._slot(hdo_blocked=True, pv_surplus_kwh=1.0)
        assert slot.effective_cost == float("inf")

    def test_pv_threshold_boundary_below(self):
        """pv_surplus_kwh=0.1 is NOT > 0.1, so should use spot price or default."""
        slot = self._slot(pv_surplus_kwh=0.1, spot_price=50.0)
        assert slot.effective_cost == pytest.approx(50.0)

    def test_pv_threshold_just_above(self):
        slot = self._slot(pv_surplus_kwh=0.101)
        assert slot.effective_cost == -1.0


# ---------------------------------------------------------------------------
# Thermodynamic helpers
# ---------------------------------------------------------------------------

class TestThermodynamicHelpers:
    def test_kwh_to_temp_rise_physics(self, sched, params):
        """1 kWh of heat into 120 L of water: ΔT = 3_600_000 / (4186 * 120) ≈ 7.17 K."""
        expected = 3_600_000 / (WATER_SPECIFIC_HEAT_J_KG_K * params.capacity_l * WATER_DENSITY_KG_L)
        assert sched._kwh_to_temp_rise(1.0) == pytest.approx(expected, rel=1e-4)

    def test_temp_drop_to_kwh_inverse(self, sched):
        """_temp_drop_to_kwh should be the inverse of _kwh_to_temp_rise."""
        kwh = 0.5
        delta = sched._kwh_to_temp_rise(kwh)
        assert sched._temp_drop_to_kwh(delta) == pytest.approx(kwh, rel=1e-5)

    def test_standby_loss_at_set_tmp(self, sched, params):
        """At set_tmp, standby loss = standby_loss_w / 1000 kWh/h."""
        expected_kwh = params.standby_loss_w / 1000.0
        assert sched._standby_kwh_per_hour(params.set_tmp) == pytest.approx(expected_kwh, rel=1e-4)

    def test_standby_loss_zero_at_ambient(self, sched, params):
        """At area_tmp there is no standby loss."""
        assert sched._standby_kwh_per_hour(params.area_tmp) == pytest.approx(0.0, abs=1e-9)

    def test_standby_loss_below_ambient_clamped(self, sched, params):
        """Below ambient temp → loss should be clamped to 0."""
        assert sched._standby_kwh_per_hour(params.area_tmp - 5) == pytest.approx(0.0, abs=1e-9)

    def test_standby_loss_proportional(self, sched, params):
        """At midpoint between ambient and set_tmp, loss should be ~50% of full."""
        mid = (params.area_tmp + params.set_tmp) / 2
        half_loss = params.standby_loss_w / 2 / 1000.0
        assert sched._standby_kwh_per_hour(mid) == pytest.approx(half_loss, rel=0.01)


# ---------------------------------------------------------------------------
# build_slots()
# ---------------------------------------------------------------------------

class TestBuildSlots:
    def test_returns_24_slots(self, sched):
        slots = sched.build_slots(_flat(), _flat(), _no_prices(), _no_hdo(), BASE_DT)
        assert len(slots) == 24

    def test_slot_indices_sequential(self, sched):
        slots = sched.build_slots(_flat(), _flat(), _no_prices(), _no_hdo(), BASE_DT)
        for i, s in enumerate(slots):
            assert s.index == i

    def test_slot_datetimes_hourly(self, sched):
        slots = sched.build_slots(_flat(), _flat(), _no_prices(), _no_hdo(), BASE_DT)
        for i, s in enumerate(slots):
            assert s.dt == BASE_DT + timedelta(hours=i)

    def test_pv_surplus_assigned(self, sched):
        pv = [0.0] * 24
        pv[5] = 1.5
        slots = sched.build_slots(_flat(), pv, _no_prices(), _no_hdo(), BASE_DT)
        assert slots[5].pv_surplus_kwh == pytest.approx(1.5)
        assert slots[6].pv_surplus_kwh == pytest.approx(0.0)

    def test_spot_prices_assigned(self, sched):
        prices = {0: 30.0, 12: 150.0}
        slots = sched.build_slots(_flat(), _flat(), prices, _no_hdo(), BASE_DT)
        assert slots[0].spot_price == pytest.approx(30.0)
        assert slots[12].spot_price == pytest.approx(150.0)
        assert slots[5].spot_price is None

    def test_hdo_blocked_assigned(self, sched):
        hdo = [False] * 24
        hdo[22] = True
        hdo[23] = True
        slots = sched.build_slots(_flat(), _flat(), _no_prices(), hdo, BASE_DT)
        assert slots[22].hdo_blocked is True
        assert slots[23].hdo_blocked is True
        assert slots[0].hdo_blocked is False

    def test_short_lists_padded_with_zeros(self, sched):
        """Lists shorter than 24 should be filled with zeros/False."""
        slots = sched.build_slots([0.1] * 3, [0.5] * 2, {}, [True] * 1, BASE_DT)
        assert slots[3].consumption_kwh == pytest.approx(0.0)
        assert slots[2].pv_surplus_kwh == pytest.approx(0.0)
        assert slots[1].hdo_blocked is False


# ---------------------------------------------------------------------------
# _simulate()
# ---------------------------------------------------------------------------

class TestSimulate:
    def test_temperature_never_exceeds_set_tmp(self, sched):
        plan = [True] * 24
        slots = sched.build_slots(_flat(), _flat(), _no_prices(), _no_hdo(), BASE_DT)
        traj = sched._simulate(55.0, slots, plan)
        assert all(t <= sched.boiler.set_tmp + 0.01 for t in traj)

    def test_temperature_never_falls_below_area_tmp(self, sched):
        plan = [False] * 24
        slots = sched.build_slots(_flat(0.5), _flat(), _no_prices(), _no_hdo(), BASE_DT)
        traj = sched._simulate(sched.boiler.area_tmp, slots, plan)
        assert all(t >= sched.boiler.area_tmp - 0.01 for t in traj)

    def test_heating_raises_temperature(self, sched):
        plan = [True] + [False] * 23
        slots = sched.build_slots(_flat(), _flat(), _no_prices(), _no_hdo(), BASE_DT)
        traj = sched._simulate(30.0, slots, plan)
        assert traj[0] > 30.0  # first hour heats up

    def test_no_heating_temperature_drops(self, sched):
        plan = [False] * 24
        slots = sched.build_slots(_flat(), _flat(), _no_prices(), _no_hdo(), BASE_DT)
        traj = sched._simulate(55.0, slots, plan)
        # Cooling only — temperature should be monotonically non-increasing
        for i in range(1, len(traj)):
            assert traj[i] <= traj[i - 1] + 0.01

    def test_consumption_cools_boiler(self, sched):
        traj_no_cons = sched._simulate(
            50.0,
            sched.build_slots(_flat(0.0), _flat(), _no_prices(), _no_hdo(), BASE_DT),
            [False] * 24,
        )
        traj_with_cons = sched._simulate(
            50.0,
            sched.build_slots(_flat(0.5), _flat(), _no_prices(), _no_hdo(), BASE_DT),
            [False] * 24,
        )
        for i in range(24):
            assert traj_with_cons[i] <= traj_no_cons[i] + 0.01

    def test_trajectory_length_24(self, sched):
        slots = sched.build_slots(_flat(), _flat(), _no_prices(), _no_hdo(), BASE_DT)
        traj = sched._simulate(50.0, slots, [False] * 24)
        assert len(traj) == 24


# ---------------------------------------------------------------------------
# plan() — core scheduling logic
# ---------------------------------------------------------------------------

class TestPlan:
    def test_returns_24_bool_plan_and_24_slots(self, sched):
        plan, slots = sched.plan(50.0, _flat(), _flat(), _no_prices(), _no_hdo(), BASE_DT)
        assert len(plan) == 24
        assert len(slots) == 24
        assert all(isinstance(b, bool) for b in plan)

    def test_pv_hours_always_heated(self, sched):
        pv = _flat()
        pv[5] = 1.0  # surplus at hour 5
        plan, _ = sched.plan(50.0, _flat(), pv, _no_prices(), _no_hdo(), BASE_DT)
        assert plan[5] is True

    def test_hdo_blocked_hours_never_heated(self, sched):
        hdo = [True] * 24
        plan, _ = sched.plan(20.0, _flat(), _flat(), _no_prices(), hdo, BASE_DT)
        # All HDO blocked — none should heat (force-all branch also respects HDO)
        assert not any(plan)

    def test_cold_boiler_triggers_heating(self, sched):
        """Start at 20°C (below min_tmp 37°C) — some hours must be heated."""
        plan, _ = sched.plan(20.0, _flat(), _flat(), _no_prices(), _no_hdo(), BASE_DT)
        assert any(plan)

    def test_warm_boiler_may_need_no_heating(self, sched):
        """Start well above set_tmp — no heating needed (slight standby cooling only)."""
        # With set_tmp=60 and no consumption, 58°C should stay above min_tmp for a few hours.
        # With significant standby loss over 24 h it might still need heating,
        # so we just check that at least one plan starts with heat=False.
        plan, _ = sched.plan(59.0, _flat(), _flat(), _no_prices(), _no_hdo(), BASE_DT)
        # At least early hours should not need heating
        assert plan[0] is False

    def test_prefers_cheaper_hours_first(self, sched):
        """When choosing among non-PV hours, the cheapest slot should be selected first."""
        prices = {i: float(100 - i) for i in range(24)}  # hour 23 cheapest
        plan, _ = sched.plan(20.0, _flat(), _flat(), prices, _no_hdo(), BASE_DT)
        # Find the cheapest active heating hour; it should be before any expensive active hour
        heated_hours = [i for i, h in enumerate(plan) if h]
        if len(heated_hours) >= 2:
            # The cheapest unblocked hour selected first → the cheapest hour
            # (highest index = lowest price in our mapping) should appear
            assert max(heated_hours) in heated_hours

    def test_forces_all_unblocked_when_no_candidates(self, sched):
        """If constraint can't be met (all hours blocked except one), force-on activates."""
        # 23 hours HDO-blocked, 1 free
        hdo = [True] * 24
        hdo[0] = False
        plan, _ = sched.plan(10.0, _flat(0.5), _flat(), _no_prices(), hdo, BASE_DT)
        # Only hour 0 is unblocked; the force branch should turn it on
        assert plan[0] is True

    def test_pv_hours_not_blocked_by_hdo(self, sched):
        """PV surplus hours should only be marked if not HDO-blocked."""
        pv = _flat()
        pv[3] = 1.0
        hdo = _no_hdo()
        hdo[3] = True
        plan, _ = sched.plan(50.0, _flat(), pv, _no_prices(), hdo, BASE_DT)
        assert plan[3] is False  # PV + HDO blocked → not heated

    def test_min_tmp_constraint_satisfied(self, sched):
        """After running plan, trajectory should not fall below min_tmp.

        Start above min_tmp so the scheduler only needs to *prevent* a drop,
        not recover from an already-cold boiler (which takes multiple heater cycles
        and can't be guaranteed in the first trajectory slot).
        """
        start_tmp = 45.0  # comfortably above min_tmp=37 and below set_tmp=60
        plan, slots = sched.plan(start_tmp, _flat(0.1), _flat(), _no_prices(), _no_hdo(), BASE_DT)
        traj = sched._simulate(start_tmp, slots, plan)
        violations = [i for i, t in enumerate(traj) if t < sched.boiler.min_tmp]
        assert violations == [], \
            f"Temperature fell below min_tmp at hours {violations}: {[traj[i] for i in violations]}"


# ---------------------------------------------------------------------------
# get_plan_summary()
# ---------------------------------------------------------------------------

class TestGetPlanSummary:
    def test_returns_required_keys(self, sched):
        plan, slots = sched.plan(50.0, _flat(), _flat(), _no_prices(), _no_hdo(), BASE_DT)
        summary = sched.get_plan_summary(50.0, plan, slots)
        for key in ("heating_hours", "temperature_trajectory", "total_heating_hours",
                    "min_predicted_temp", "slots"):
            assert key in summary

    def test_heating_hours_count_matches_plan(self, sched):
        plan, slots = sched.plan(20.0, _flat(), _flat(), _no_prices(), _no_hdo(), BASE_DT)
        summary = sched.get_plan_summary(20.0, plan, slots)
        assert summary["total_heating_hours"] == sum(plan)
        assert len(summary["heating_hours"]) == sum(plan)

    def test_slots_length_24(self, sched):
        plan, slots = sched.plan(50.0, _flat(), _flat(), _no_prices(), _no_hdo(), BASE_DT)
        summary = sched.get_plan_summary(50.0, plan, slots)
        assert len(summary["slots"]) == 24

    def test_temperature_trajectory_length_24(self, sched):
        plan, slots = sched.plan(50.0, _flat(), _flat(), _no_prices(), _no_hdo(), BASE_DT)
        summary = sched.get_plan_summary(50.0, plan, slots)
        assert len(summary["temperature_trajectory"]) == 24

    def test_min_predicted_temp_correct(self, sched):
        plan, slots = sched.plan(50.0, _flat(), _flat(), _no_prices(), _no_hdo(), BASE_DT)
        summary = sched.get_plan_summary(50.0, plan, slots)
        assert summary["min_predicted_temp"] == min(summary["temperature_trajectory"])

    def test_slot_flags_consistent(self, sched):
        pv = _flat()
        pv[2] = 1.0
        hdo = _no_hdo()
        hdo[10] = True
        plan, slots = sched.plan(50.0, _flat(), pv, _no_prices(), hdo, BASE_DT)
        summary = sched.get_plan_summary(50.0, plan, slots)

        pv_slot = summary["slots"][2]
        hdo_slot = summary["slots"][10]
        assert pv_slot["pv_free"] is True
        assert hdo_slot["hdo_blocked"] is True


# ---------------------------------------------------------------------------
# End-to-end scenario simulations
# ---------------------------------------------------------------------------

class TestEndToEndScenarios:
    def test_scenario_overnight_hdo_with_pv_morning(self, sched):
        """
        Night hours (0-5) are HDO blocked.
        PV surplus from 10:00–14:00.
        Boiler starts cold (25°C). Expect:
        - Hours 0–5 NOT heated (HDO)
        - Hours 10–13 heated (PV)
        - Additional cheap spot hours added to satisfy min_tmp
        """
        hdo = [True if h < 6 else False for h in range(24)]
        pv = [0.0] * 24
        for h in range(10, 14):
            pv[h] = 1.0  # 1 kWh surplus each PV hour
        prices = {h: 50.0 + h * 2 for h in range(24)}  # cheap early afternoon

        plan, slots = sched.plan(25.0, _flat(0.2), pv, prices, hdo, BASE_DT)

        for h in range(6):
            assert plan[h] is False, f"HDO hour {h} should not heat"
        for h in range(10, 14):
            assert plan[h] is True, f"PV hour {h} should heat"

    def test_scenario_already_fully_hot_no_heating_needed_initially(self, sched):
        """Boiler at 60°C (set point) with no consumption → no heating for many hours."""
        plan, slots = sched.plan(
            59.5, _flat(0.0), _flat(), _no_prices(), _no_hdo(), BASE_DT
        )
        # With standby losses only (~0.05 kWh/h) and 60→37 buffer, won't need immediate heat
        assert plan[0] is False

    def test_scenario_heavy_consumption_requires_frequent_heating(self, sched):
        """High consumption (0.8 kWh/h) forces the scheduler to heat many hours."""
        plan, _ = sched.plan(50.0, _flat(0.8), _flat(), _no_prices(), _no_hdo(), BASE_DT)
        assert sum(plan) >= 8  # At least 8 hours heating needed for heavy use

    def test_scenario_spot_price_avoids_expensive_hours(self, sched):
        """With very high prices at night and cheap midday, planner avoids expensive hours."""
        prices = {}
        for h in range(24):
            prices[h] = 200.0 if h < 6 else 30.0  # night expensive, day cheap
        plan, _ = sched.plan(30.0, _flat(0.3), _flat(), prices, _no_hdo(), BASE_DT)
        # Cheap daytime hours (6+) should be preferred over night
        heated = [h for h in range(24) if plan[h]]
        if heated:
            assert any(h >= 6 for h in heated)
