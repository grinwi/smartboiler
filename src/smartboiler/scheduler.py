# Created as a part of Master's Thesis "Using machine learning methods to save energy in a smart home"
# Faculty of Information Technology, Brno University of Technology, 2024
# Author: Adam Grünwald
#
# Greedy day-ahead heating scheduler.
# Sorts hours by effective cost (PV free → cheap spot → expensive → HDO blocked)
# and marks cheapest hours as heating=True until the min_tmp constraint is satisfied.
#
# Standby-loss-adjusted cost:
#   heating early at a cheap price P₁ incurs extra standby losses while waiting for
#   consumption — this is factored in so that a slot slightly cheaper than a later one
#   is only preferred if the price advantage exceeds the accrued standby energy cost.

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    from smartboiler.calendar_manager import BoilerEvent

logger = logging.getLogger(__name__)

WATER_SPECIFIC_HEAT_J_KG_K = 4186.0
WATER_DENSITY_KG_L = 1.0
DEFAULT_MEDIUM_PRICE = 75.0  # EUR/MWh used when no spot price available


@dataclass
class BoilerParams:
    capacity_l: float = 120.0
    wattage_w: float = 2000.0
    efficiency: float = 0.90
    set_tmp: float = 60.0
    min_tmp: float = 37.0
    area_tmp: float = 20.0
    # Approximate standby heat loss at set_tmp; scales with temp delta
    standby_loss_w: float = 50.0
    # Battery / PV integration (optional — all zero/default means no battery)
    battery_capacity_kwh: float = 0.0   # 0 = no battery installed
    battery_soc_kwh: float = 0.0        # current state of charge at planning time
    battery_max_charge_kw: float = 0.0  # max charge rate (kW); 0 = unlimited
    # How to allocate PV surplus when a battery is present:
    #   battery_first — charge battery first, boiler gets the remainder
    #   boiler_first  — boiler gets all PV; battery absorbs whatever is left over
    #   sell_first    — sell everything; no free PV for boiler or battery
    battery_priority: str = "battery_first"


@dataclass
class HourSlot:
    index: int              # 0 = current hour, 1 = next hour, ...
    dt: datetime
    consumption_kwh: float
    pv_surplus_kwh: float   # free solar surplus available this hour (after battery allocation)
    spot_price: Optional[float]   # EUR/MWh; None = unknown
    hdo_blocked: bool
    # Calendar override: vacation_min | vacation_off | boost_max | boost_temp | None
    calendar_mode: Optional[str] = None
    calendar_target_tmp: Optional[float] = None
    effective_cost: float = field(init=False)

    def __post_init__(self):
        if self.hdo_blocked or self.calendar_mode == "vacation_off":
            self.effective_cost = float("inf")
        elif self.calendar_mode in ("boost_max", "boost_temp"):
            self.effective_cost = -2.0   # forced heat — higher priority than PV
        elif self.pv_surplus_kwh > 0.1:
            self.effective_cost = -1.0   # free solar
        elif self.spot_price is not None:
            self.effective_cost = self.spot_price
        else:
            self.effective_cost = DEFAULT_MEDIUM_PRICE


# ── Calendar helpers ──────────────────────────────────────────────────────────

def _calendar_mode_for_hour(
    slot_dt: datetime,
    calendar_events: Optional[List],
) -> Tuple[Optional[str], Optional[float]]:
    """Return (calendar_mode, target_temp) for the given hour slot."""
    if not calendar_events:
        return None, None
    _priority = {"vacation_off": 0, "boost_max": 1, "boost_temp": 1, "vacation_min": 2}
    covering = [e for e in calendar_events if e.covers_hour(slot_dt)]
    if not covering:
        return None, None
    best = min(covering, key=lambda e: _priority.get(e.event_type, 99))
    return best.event_type, best.target_temp


def _slot_min_tmp(slot: HourSlot, normal_min_tmp: float, vacation_min_tmp: float) -> float:
    """Effective min_tmp constraint for a given slot, respecting calendar mode."""
    if slot.calendar_mode in ("vacation_min", "vacation_off"):
        return vacation_min_tmp
    return normal_min_tmp


class HeatingScheduler:
    """Plans heating windows for the next 24 hours using greedy cost minimization."""

    def __init__(self, boiler: BoilerParams):
        self.boiler = boiler
        self._real_wattage_kw = (boiler.wattage_w * boiler.efficiency) / 1000.0

    # ── Thermodynamic helpers ─────────────────────────────────────────────

    def _kwh_to_temp_rise(self, kwh: float) -> float:
        """Temperature rise in boiler from adding kwh of heat energy."""
        joules = kwh * 3_600_000
        mass_kg = self.boiler.capacity_l * WATER_DENSITY_KG_L
        return joules / (WATER_SPECIFIC_HEAT_J_KG_K * mass_kg)

    def _temp_drop_to_kwh(self, delta_c: float) -> float:
        """kWh stored in boiler temperature delta."""
        mass_kg = self.boiler.capacity_l * WATER_DENSITY_KG_L
        joules = WATER_SPECIFIC_HEAT_J_KG_K * mass_kg * delta_c
        return joules / 3_600_000

    def _standby_kwh_per_hour(self, tmp: float) -> float:
        """Standby heat loss per hour, proportional to temp delta vs ambient.

        By Newton's law, heat loss through the boiler jacket is proportional to
        (T_water - T_ambient).  The configured `standby_loss_w` is the rated
        loss at `set_tmp`; we scale it linearly for lower temperatures:

            loss_w(T) = standby_loss_w × (T - T_amb) / (T_set - T_amb)

        This means a boiler at 40 °C with T_amb=20 °C and T_set=60 °C loses
        half the standby energy compared to being at 60 °C.
        """
        delta = max(0.0, tmp - self.boiler.area_tmp)
        ref_delta = max(1.0, self.boiler.set_tmp - self.boiler.area_tmp)
        return (self.boiler.standby_loss_w * delta / ref_delta) / 1000.0

    # ── Battery / PV allocation ───────────────────────────────────────────

    def _compute_net_pv_for_boiler(self, pv_forecast: List[float]) -> List[float]:
        """Compute per-hour PV surplus available to the boiler after battery priority.

        battery_first: battery absorbs PV until full; boiler gets remainder.
        boiler_first:  boiler gets full PV surplus (battery charges from other sources).
        sell_first:    PV is sold to grid; boiler gets nothing for free.
        No battery:    all PV surplus goes directly to the boiler.
        """
        if self.boiler.battery_capacity_kwh <= 0:
            return list(pv_forecast)

        priority = self.boiler.battery_priority
        if priority == "sell_first":
            return [0.0] * len(pv_forecast)
        if priority == "boiler_first":
            return list(pv_forecast)

        # battery_first: simulate SoC evolution hour by hour
        soc = min(self.boiler.battery_soc_kwh, self.boiler.battery_capacity_kwh)
        capacity = self.boiler.battery_capacity_kwh
        max_charge = (
            self.boiler.battery_max_charge_kw
            if self.boiler.battery_max_charge_kw > 0
            else float("inf")
        )
        net = []
        for pv_kwh in pv_forecast:
            charge = min(pv_kwh, max_charge, capacity - soc)
            soc += charge
            net.append(max(0.0, pv_kwh - charge))
        return net

    # ── Core planner ──────────────────────────────────────────────────────

    def build_slots(
        self,
        consumption_forecast: List[float],
        pv_forecast: List[float],
        spot_prices: Dict[int, Optional[float]],
        hdo_blocked: List[bool],
        from_dt: Optional[datetime] = None,
        calendar_events: Optional[List] = None,
    ) -> List[HourSlot]:
        """Build a list of HourSlots for the next 24 hours.

        PV surplus is adjusted for battery priority before slot construction,
        so HourSlot.pv_surplus_kwh always reflects what the boiler can actually use.
        """
        if from_dt is None:
            from_dt = datetime.now().astimezone().replace(minute=0, second=0, microsecond=0)
        net_pv = self._compute_net_pv_for_boiler(pv_forecast)
        slots = []
        for i in range(24):
            dt_i = from_dt + timedelta(hours=i)
            cal_mode, cal_target = _calendar_mode_for_hour(dt_i, calendar_events)
            slots.append(
                HourSlot(
                    index=i,
                    dt=dt_i,
                    consumption_kwh=consumption_forecast[i] if i < len(consumption_forecast) else 0.0,
                    pv_surplus_kwh=net_pv[i] if i < len(net_pv) else 0.0,
                    spot_price=spot_prices.get(i),
                    hdo_blocked=hdo_blocked[i] if i < len(hdo_blocked) else False,
                    calendar_mode=cal_mode,
                    calendar_target_tmp=cal_target,
                )
            )
        return slots

    def plan(
        self,
        current_tmp: float,
        consumption_forecast: List[float],
        pv_forecast: List[float],
        spot_prices: Dict[int, Optional[float]],
        hdo_blocked: List[bool],
        from_dt: Optional[datetime] = None,
        calendar_events: Optional[List] = None,
        vacation_min_tmp: Optional[float] = None,
    ) -> Tuple[List[bool], List[HourSlot]]:
        """Compute optimal heating plan for next 24 hours.

        Candidate selection uses standby-loss-adjusted cost:
            adjusted(slot) = heating_cost_eur + standby_penalty_eur_per_hour * hours_early
        where hours_early = first_violation_index - slot.index.

        This means a cheap early slot is only preferred over a slightly pricier later
        slot when the price advantage exceeds the cost of the extra standby losses
        incurred by keeping the boiler hot for longer.  The `-slot.index` secondary key
        provides a final tiebreaker for slots with truly identical adjusted costs.

        Returns:
            (heating_plan, slots) — heating_plan[i] is True if hour i should heat.
        """
        _vacation_min = vacation_min_tmp if vacation_min_tmp is not None else self.boiler.min_tmp
        slots = self.build_slots(
            consumption_forecast, pv_forecast, spot_prices, hdo_blocked,
            from_dt, calendar_events,
        )
        n = len(slots)
        heating_plan = [False] * n

        # Step 1: always heat during PV surplus and boost events (free / forced energy)
        for slot in slots:
            if slot.effective_cost < 0:   # -2 (boost) or -1 (PV)
                heating_plan[slot.index] = True

        # Pre-compute standby penalty per hour of early heating.
        # Reference price = average of known spot prices (or DEFAULT_MEDIUM_PRICE).
        # Temperature proxy = midpoint between min_tmp and set_tmp.
        known_prices = [s.spot_price for s in slots if s.spot_price is not None]
        ref_price_eur_kwh = (
            sum(known_prices) / len(known_prices) / 1000.0
            if known_prices
            else DEFAULT_MEDIUM_PRICE / 1000.0
        )
        mid_tmp = (self.boiler.set_tmp + self.boiler.min_tmp) / 2.0
        standby_eur_per_hour = self._standby_kwh_per_hour(mid_tmp) * ref_price_eur_kwh

        # Step 2: greedy fill — add cheapest hours until per-slot min_tmp constraint satisfied
        max_iterations = n
        for _ in range(max_iterations):
            trajectory = self._simulate(current_tmp, slots, heating_plan)
            violations = [
                i for i, t in enumerate(trajectory)
                if t < _slot_min_tmp(slots[i], self.boiler.min_tmp, _vacation_min)
            ]
            if not violations:
                break

            first_viol = min(violations)
            candidates = [
                s
                for s in slots
                if s.index <= first_viol
                and not heating_plan[s.index]
                and s.effective_cost < float("inf")
            ]
            if not candidates:
                # No slots left — force-enable all non-blocked hours
                for s in slots:
                    if s.effective_cost < float("inf"):
                        heating_plan[s.index] = True
                break

            # Standby-loss-adjusted cost: prefer later slots when the price advantage
            # of an earlier slot does not cover the extra standby losses incurred.
            cheapest = min(
                candidates,
                key=lambda s: (
                    s.effective_cost / 1000.0 * self._real_wattage_kw
                    + standby_eur_per_hour * max(0, first_viol - s.index),
                    -s.index,   # tiebreaker: latest slot wins for truly equal costs
                ),
            )
            heating_plan[cheapest.index] = True

        return heating_plan, slots

    def _simulate(
        self,
        start_tmp: float,
        slots: List[HourSlot],
        heating_plan: List[bool],
    ) -> List[float]:
        """Simulate boiler water temperature trajectory given heating plan.

        Per-hour order of operations (matters for the temperature accounting):
          1. Standby cooling  — heat always leaks through the boiler jacket.
          2. Hot water draw   — modelled as dumping hot water and replacing with
                               cold; the energy carried away equals
                               consumed_kwh (measured at outlet vs inlet).
          3. Heating          — relay ON adds (wattage × efficiency) kWh;
                               clamped at set_tmp (thermostat cut-off).
          4. Physical floor   — temperature cannot fall below ambient.

        The trajectory[i] value represents the temperature at the END of hour i,
        which is also the start temperature for the min_tmp constraint check.
        """
        tmp = max(start_tmp, self.boiler.area_tmp)
        trajectory = []
        for slot in slots:
            # 1. Standby cooling — proportional to (T_water - T_ambient)
            tmp -= self._kwh_to_temp_rise(self._standby_kwh_per_hour(tmp))
            # 2. Hot water draw — energy removed when mixing hot water with cold
            tmp -= self._kwh_to_temp_rise(slot.consumption_kwh)
            # 3. Heating — element adds heat; internal thermostat caps at set_tmp
            if heating_plan[slot.index]:
                tmp += self._kwh_to_temp_rise(self._real_wattage_kw)
                tmp = min(tmp, self.boiler.set_tmp)
            # 4. Physical floor
            tmp = max(tmp, self.boiler.area_tmp)
            trajectory.append(round(tmp, 2))
        return trajectory

    def get_plan_summary(
        self,
        current_tmp: float,
        heating_plan: List[bool],
        slots: List[HourSlot],
    ) -> Dict:
        """Return human-readable plan summary for dashboard."""
        trajectory = self._simulate(current_tmp, slots, heating_plan)
        return {
            "heating_hours": [
                slots[i].dt.strftime("%H:%M") for i, h in enumerate(heating_plan) if h
            ],
            "temperature_trajectory": trajectory,
            "total_heating_hours": sum(heating_plan),
            "min_predicted_temp": min(trajectory) if trajectory else None,
            "slots": [
                {
                    "hour": s.dt.strftime("%H:%M"),
                    "heating": heating_plan[s.index],
                    "hdo_blocked": s.hdo_blocked,
                    "pv_free": s.pv_surplus_kwh > 0.1,
                    "consumption_kwh": round(s.consumption_kwh, 4),
                    "spot_price": s.spot_price,
                    "label": s.dt.strftime("%H:00"),
                    "calendar_mode": s.calendar_mode,
                    "calendar_target_tmp": s.calendar_target_tmp,
                }
                for s in slots
            ],
        }
