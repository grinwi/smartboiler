# Savings analysis: replay real history through different heating strategies.
#
# Given hourly hot-water use and PV production from the past, how much would
# smart heating have saved compared with a plain thermostat? Every strategy
# runs through the same boiler physics as the scheduler (standby loss, hot
# water draw, heater capped at set_tmp), hour by hour:
#
#   thermostat  — today's behaviour: heats whenever water drops 5 °C below
#                 set_tmp, whatever the price or the sun.
#   smart_ideal — the scheduler re-plans every hour knowing the real future
#                 consumption: the theoretical upper bound of savings.
#   smart_real  — the scheduler re-plans every hour with the rolling
#                 histogram predictor trained only on the past (retrained
#                 daily): what the add-on would realistically achieve.
#
# Both smart strategies see the real PV surplus of the next 24 h (a good
# forecast plus the real-time surplus dispatcher come close to this) and,
# like the controller, heat at once when water falls below min_tmp.
#
# Hot-water demand is counted as heat at a reference temperature, so the
# same showers need the same energy in every strategy:
#     kWh = litres × 4.186 × (hot_c − cold_c) / 3600
#
# Cost per hour = grid kWh × import price + PV kWh × export price (the
# feed-in income given up by using the PV at home). Prices come from the
# tariff (tariff.py) and, for spot tariffs, an optional hourly spot series.
#
# Run as:  python -m smartboiler.savings --help

import argparse
import json
import logging
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import pandas as pd

from smartboiler.predictor import RollingHistogramPredictor
from smartboiler.scheduler import BoilerParams, HeatingScheduler
from smartboiler.tariff import Tariff, parse_time_windows

logger = logging.getLogger(__name__)

WATER_KWH_PER_LITRE_K = 4.186 / 3600.0   # heat to warm 1 L of water by 1 K
THERMOSTAT_HYSTERESIS_C = 5.0             # typical boiler thermostat switching band
INPUT_KINDS = ("counter", "power_w", "per_minute", "per_hour")


# ── Loading data ─────────────────────────────────────────────────────────────

def load_csv(path: str, entity_id: Optional[str] = None) -> pd.Series:
    """Load a time series from CSV as a float Series with a UTC DatetimeIndex.

    Accepts either Home Assistant's history export (columns entity_id,
    state, last_changed — filtered by ``entity_id`` when the file holds
    several entities) or a plain two-column CSV ``time,value``.
    Non-numeric states (unavailable, unknown) are dropped.
    """
    df = pd.read_csv(path)
    if {"state", "last_changed"} <= set(df.columns):
        if entity_id:
            df = df[df["entity_id"] == entity_id]
        times, values = df["last_changed"], df["state"]
    else:
        times, values = df.iloc[:, 0], df.iloc[:, 1]
    s = pd.Series(
        pd.to_numeric(values, errors="coerce").to_numpy(),
        index=pd.to_datetime(times, utc=True, format="mixed"),
    )
    return s.dropna().sort_index()


def to_hourly(raw: pd.Series, kind: str) -> pd.Series:
    """Convert raw readings to an amount per clock hour.

    kind:
        counter    — cumulative total (L, kWh); hourly increase, resets ignored
        power_w    — power in W; time-weighted mean → kWh per hour
        per_minute — rate per minute (e.g. L/min); time-weighted mean × 60
        per_hour   — amounts already per interval; summed per hour
    """
    if kind not in INPUT_KINDS:
        raise ValueError(f"kind must be one of {INPUT_KINDS}")
    if raw.empty:
        return raw
    if kind == "per_hour":
        return raw.resample("1h").sum()
    if kind == "counter":
        last = raw.resample("1h").last().ffill()
        diff = last.diff()
        return diff.where(diff >= 0, 0.0).fillna(0.0)   # negative = counter reset
    # Rates: HA stores a value only when it changes, so weight each value by
    # how long it was valid (forward-fill to minutes, then average).
    per_min = raw.resample("1min").last().ffill()
    mean = per_min.resample("1h").mean()
    return mean / 1000.0 if kind == "power_w" else mean * 60.0


def water_to_heat_kwh(litres: pd.Series, hot_c: float, cold_c: float) -> pd.Series:
    """Heat needed to supply ``litres`` of hot water at ``hot_c`` from ``cold_c``."""
    return litres * WATER_KWH_PER_LITRE_K * max(0.0, hot_c - cold_c)


# ── Simulation ───────────────────────────────────────────────────────────────

@dataclass
class Inputs:
    """Aligned hourly inputs (same local-time DatetimeIndex)."""
    demand_kwh: pd.Series          # hot-water heat drawn per hour
    surplus_kwh: pd.Series         # PV surplus available to the boiler per hour
    spot_eur_mwh: pd.Series        # NaN where unknown
    hdo_blocked: pd.Series         # bool: heating impossible this hour


@dataclass
class StrategyResult:
    name: str
    electricity_kwh: float = 0.0
    pv_kwh: float = 0.0
    grid_kwh: float = 0.0
    cost: float = 0.0
    standby_kwh: float = 0.0
    hours_below_min: int = 0
    mean_tmp: float = 0.0
    final_tmp: float = 0.0
    monthly_cost: Dict[str, float] = field(default_factory=dict)


def _step(sched: HeatingScheduler, tmp: float, demand_kwh: float,
          heat_on: Callable[[float], bool], blocked: bool):
    """Advance the boiler by one hour; return (new_tmp, electricity_kwh, standby_kwh).

    Same order as HeatingScheduler._simulate: standby loss, draw, heating
    (capped at set_tmp), floor at ambient. ``heat_on`` is asked after the
    draw, so it can react like a thermostat.
    """
    b = sched.boiler
    standby = sched._standby_kwh_per_hour(tmp)
    tmp -= sched._kwh_to_temp_rise(standby)
    tmp -= sched._kwh_to_temp_rise(demand_kwh)
    tmp = max(tmp, b.area_tmp)
    electricity = 0.0
    if not blocked and heat_on(tmp):
        room_kwh = sched._temp_drop_to_kwh(max(0.0, b.set_tmp - tmp))
        heat = min(sched._real_wattage_kw, room_kwh)
        tmp += sched._kwh_to_temp_rise(heat)
        electricity = heat / b.efficiency
    return tmp, electricity, standby


def simulate(name: str, inputs: Inputs, sched: HeatingScheduler,
             vacation_min_tmp: Optional[float] = None) -> StrategyResult:
    """Run one strategy over the whole period."""
    b, tariff = sched.boiler, sched.tariff
    idx = inputs.demand_kwh.index
    demand = inputs.demand_kwh.to_numpy()
    surplus = inputs.surplus_kwh.to_numpy()
    spot = inputs.spot_eur_mwh.to_numpy()
    blocked = inputs.hdo_blocked.to_numpy()
    n = len(idx)
    low_hours = tariff.low_tariff_hours

    predictor = RollingHistogramPredictor() if name == "smart_real" else None
    history = pd.DataFrame({"consumed_kwh": inputs.demand_kwh})

    res = StrategyResult(name=name)
    tmp = b.set_tmp
    tmp_sum = 0.0
    monthly: Dict[str, float] = {}

    for i in range(n):
        ts = idx[i]
        spot_i = None if pd.isna(spot[i]) else float(spot[i])

        if name == "thermostat":
            def heat_on(t):
                return t < b.set_tmp - THERMOSTAT_HYSTERESIS_C
        else:
            j = min(n, i + 24)
            if predictor is not None:
                if i == 0 or ts.hour == 0:
                    predictor.update(history.iloc[:i], now=ts)
                forecast = predictor.predict_next_24h(ts.to_pydatetime())
            else:
                forecast = list(demand[i:j]) + [0.0] * (24 - (j - i))
            prices = {k: (None if pd.isna(spot[i + k]) else float(spot[i + k]))
                      for k in range(j - i)}
            plan, _ = sched.plan(
                current_tmp=tmp,
                consumption_forecast=forecast,
                pv_forecast=list(surplus[i:j]),
                spot_prices=prices,
                hdo_blocked=[bool(x) for x in blocked[i:j]],
                from_dt=ts.to_pydatetime(),
                vacation_min_tmp=vacation_min_tmp,
            )
            planned = plan[0]

            def heat_on(t, planned=planned):
                return planned or t < b.min_tmp

        tmp, elec, standby = _step(sched, tmp, demand[i], heat_on, bool(blocked[i]))

        low = (ts.hour in low_hours) if low_hours is not None else not blocked[i]
        import_price = tariff.import_price(spot_i, low_tariff=low)
        export_price = tariff.export_price_for(spot_i)
        pv_used = min(elec, max(0.0, surplus[i]))
        grid = elec - pv_used
        cost = grid * import_price + pv_used * export_price

        res.electricity_kwh += elec
        res.pv_kwh += pv_used
        res.grid_kwh += grid
        res.cost += cost
        res.standby_kwh += standby
        res.hours_below_min += tmp < b.min_tmp
        tmp_sum += tmp
        month = ts.strftime("%Y-%m")
        monthly[month] = monthly.get(month, 0.0) + cost

    res.mean_tmp = tmp_sum / n if n else 0.0
    res.final_tmp = tmp
    res.monthly_cost = monthly
    return res


STRATEGIES = ("thermostat", "smart_ideal", "smart_real")


def analyse(inputs: Inputs, boiler: BoilerParams, tariff: Tariff) -> List[StrategyResult]:
    """Run all strategies on the same inputs."""
    sched = HeatingScheduler(boiler, tariff=tariff)
    return [simulate(name, inputs, sched) for name in STRATEGIES]


# ── Report ───────────────────────────────────────────────────────────────────

LABELS = {
    "thermostat": "Thermostat (today)",
    "smart_ideal": "Smart, perfect forecast (upper bound)",
    "smart_real": "Smart, learned forecast (realistic)",
}


def format_report(results: List[StrategyResult], inputs: Inputs, currency: str = "") -> str:
    """Markdown report: totals, savings against the thermostat, monthly costs."""
    base = results[0]
    idx = inputs.demand_kwh.index
    days = len(idx) / 24.0
    cur = f" {currency}" if currency else ""
    lines = [
        "# SmartBoiler savings analysis",
        "",
        f"Period: {idx[0]:%Y-%m-%d} – {idx[-1]:%Y-%m-%d} ({days:.0f} days). "
        f"Hot-water heat used: {inputs.demand_kwh.sum():.0f} kWh "
        f"({inputs.demand_kwh.sum() / days:.1f} kWh/day). "
        f"PV surplus available: {inputs.surplus_kwh.sum():.0f} kWh.",
        "",
        "| Strategy | Electricity kWh | from PV | from grid | Standby loss kWh "
        "| Cost | Saving vs thermostat | Hours below min | Mean °C |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in results:
        saving = base.cost - r.cost
        pct = f" ({saving / base.cost * 100:.0f} %)" if base.cost > 0 and r is not base else ""
        lines.append(
            f"| {LABELS[r.name]} | {r.electricity_kwh:.0f} | {r.pv_kwh:.0f} | {r.grid_kwh:.0f} "
            f"| {r.standby_kwh:.0f} | {r.cost:.0f}{cur} "
            f"| {'—' if r is base else f'{saving:.0f}{cur}{pct}'} "
            f"| {r.hours_below_min} | {r.mean_tmp:.1f} |"
        )
    real = next(r for r in results if r.name == "smart_real")
    if days > 0:
        lines += ["", f"Realistic saving extrapolated to a year: "
                      f"**{(base.cost - real.cost) / days * 365:.0f}{cur}**."]
    lines += ["", "## Cost per month", "",
              "| Month | " + " | ".join(LABELS[r.name] for r in results) + " |",
              "|---|" + "---:|" * len(results)]
    for month in base.monthly_cost:
        lines.append(f"| {month} | " + " | ".join(
            f"{r.monthly_cost.get(month, 0.0):.0f}" for r in results) + " |")
    lines += [
        "",
        "## Assumptions",
        "",
        "- Same boiler physics for all strategies (config: volume, heater power, set/min "
        "temperature, standby loss, ambient temperature).",
        "- Hot water counted as heat at the reference temperature, so every strategy "
        "delivers the same showers.",
        "- PV surplus = production − constant house base load; smart strategies know the "
        "next 24 h of it, the thermostat uses it only by coincidence.",
        "- PV used at home is charged at the export price (income given up).",
        "- Hourly resolution: within-hour effects (thermostat timing, short clouds) are averaged.",
    ]
    return "\n".join(lines) + "\n"


# ── Command line ─────────────────────────────────────────────────────────────

def boiler_from_config(cfg: dict) -> BoilerParams:
    """BoilerParams from a smartboiler_setup.json dict (defaults if missing)."""
    return BoilerParams(
        capacity_l=float(cfg.get("boiler_volume", 120)),
        wattage_w=float(cfg.get("boiler_watt_power", 2000)),
        set_tmp=float(cfg.get("boiler_set_tmp", 60)),
        min_tmp=float(cfg.get("boiler_min_operation_tmp", 37)),
        area_tmp=float(cfg.get("average_boiler_surroundings_temp", 20)),
        standby_loss_w=float(cfg.get("boiler_standby_watts", 50)),
    )


def build_inputs(demand_kwh: pd.Series, pv_kwh: Optional[pd.Series], base_load_kw: float,
                 tz: str, spot: Optional[pd.Series] = None,
                 hdo_schedule: str = "") -> Inputs:
    """Align all series on one local-time hourly index covering the water data."""
    demand = demand_kwh.tz_convert(tz)
    idx = demand.index
    if pv_kwh is not None:
        pv = pv_kwh.tz_convert(tz).reindex(idx).fillna(0.0)
    else:
        pv = pd.Series(0.0, index=idx)
    surplus = (pv - base_load_kw).clip(lower=0.0)
    spot_s = (spot.tz_convert(tz).reindex(idx) if spot is not None
              else pd.Series(float("nan"), index=idx))
    blocked_hours = parse_time_windows(hdo_schedule)
    blocked = pd.Series([ts.hour in blocked_hours for ts in idx], index=idx)
    return Inputs(demand.fillna(0.0), surplus, spot_s, blocked)


def main(argv: Optional[List[str]] = None) -> None:
    p = argparse.ArgumentParser(
        prog="python -m smartboiler.savings",
        description="Estimate how much smart boiler heating would have saved on your history.",
    )
    p.add_argument("--config", help="smartboiler_setup.json (boiler, tariff, PV settings)")
    p.add_argument("--water", required=True, help="CSV with hot-water use")
    p.add_argument("--water-entity", help="entity_id inside a multi-entity HA export")
    p.add_argument("--water-kind", default="counter", choices=INPUT_KINDS)
    p.add_argument("--water-unit", default="litres", choices=("litres", "kwh"),
                   help="litres are converted to heat with --hot-temp/--cold-temp")
    p.add_argument("--hot-temp", type=float, default=50.0, help="°C of the hot water drawn")
    p.add_argument("--cold-temp", type=float, help="°C of the cold inlet (default: config or 10)")
    p.add_argument("--pv", help="CSV with PV production")
    p.add_argument("--pv-entity", help="entity_id inside a multi-entity HA export")
    p.add_argument("--pv-kind", default="counter", choices=INPUT_KINDS)
    p.add_argument("--spot", help="CSV with hourly spot prices in EUR/MWh (time,value)")
    p.add_argument("--timezone", default="Europe/Prague")
    p.add_argument("--currency", default="", help="label for money values, e.g. Kč")
    p.add_argument("--out", help="write the Markdown report here (default: print)")
    args = p.parse_args(argv)

    cfg: dict = {}
    if args.config:
        with open(args.config) as f:
            cfg = json.load(f)

    water = to_hourly(load_csv(args.water, args.water_entity), args.water_kind)
    if args.water_unit == "litres":
        cold = args.cold_temp if args.cold_temp is not None else float(cfg.get("cold_water_temp", 10))
        demand = water_to_heat_kwh(water, args.hot_temp, cold)
    else:
        demand = water
    pv = to_hourly(load_csv(args.pv, args.pv_entity), args.pv_kind) if args.pv else None
    spot = to_hourly(load_csv(args.spot), "per_hour") if args.spot else None

    inputs = build_inputs(
        demand, pv,
        base_load_kw=float(cfg.get("house_base_load_w", 300)) / 1000.0,
        tz=args.timezone, spot=spot,
        hdo_schedule=str(cfg.get("hdo_explicit_schedule", "") or ""),
    )
    report = format_report(analyse(inputs, boiler_from_config(cfg), Tariff.from_config(cfg)),
                           inputs, currency=args.currency)
    if args.out:
        with open(args.out, "w") as f:
            f.write(report)
        print(f"Report written to {args.out}")
    else:
        print(report)


if __name__ == "__main__":
    main()
