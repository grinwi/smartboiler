"""Tests for the savings analysis: loading, hourly conversion, physics and report."""
import json

import numpy as np
import pandas as pd
import pytest

from smartboiler import savings as sv
from smartboiler.scheduler import BoilerParams, HeatingScheduler
from smartboiler.tariff import Tariff

TZ = "Europe/Prague"


# ── Loading and conversion ───────────────────────────────────────────────────

def test_load_ha_export_filters_entity_and_drops_non_numeric(tmp_path):
    p = tmp_path / "h.csv"
    p.write_text(
        "entity_id,state,last_changed\n"
        "sensor.water,10,2025-01-01T00:00:00.000Z\n"
        "sensor.other,99,2025-01-01T00:10:00.000Z\n"
        "sensor.water,unavailable,2025-01-01T00:20:00.000Z\n"
        "sensor.water,12.5,2025-01-01T00:30:00Z\n"
    )
    s = sv.load_csv(str(p), "sensor.water")
    assert list(s) == [10.0, 12.5]
    assert str(s.index.tz) == "UTC"


def test_load_plain_two_column_csv(tmp_path):
    p = tmp_path / "p.csv"
    p.write_text("time,value\n2025-01-01 00:00:00+00:00,1.5\n2025-01-01 01:00:00+00:00,2\n")
    assert list(sv.load_csv(str(p))) == [1.5, 2.0]


def _series(values, start="2025-01-01 00:00", freq="1h"):
    return pd.Series(values, index=pd.date_range(start, periods=len(values), freq=freq, tz="UTC"),
                     dtype=float)


def test_counter_to_hourly_ignores_reset():
    hourly = sv.to_hourly(_series([100, 110, 125, 5, 8]), "counter")
    assert list(hourly) == [0.0, 10.0, 15.0, 0.0, 3.0]


def test_power_is_time_weighted():
    # 2000 W for 15 min, then 0 W for 45 min → 0.5 kWh in that hour
    raw = pd.Series([2000.0, 0.0, 0.0], index=pd.to_datetime(
        ["2025-01-01 00:00", "2025-01-01 00:15", "2025-01-01 00:59"], utc=True))
    assert sv.to_hourly(raw, "power_w").iloc[0] == pytest.approx(0.5)


def test_per_minute_rate_to_hourly_amount():
    # 6 L/min for 10 min of an hour → 60 L
    raw = pd.Series([6.0, 0.0, 0.0], index=pd.to_datetime(
        ["2025-01-01 00:00", "2025-01-01 00:10", "2025-01-01 00:59"], utc=True))
    assert sv.to_hourly(raw, "per_minute").iloc[0] == pytest.approx(60.0)


def test_unknown_kind_rejected():
    with pytest.raises(ValueError):
        sv.to_hourly(_series([1.0]), "weird")


def test_water_to_heat():
    # 100 L heated by 40 K = 100 × 4.186 × 40 / 3600 kWh
    assert sv.water_to_heat_kwh(pd.Series([100.0]), 50, 10).iloc[0] == pytest.approx(4.651, abs=1e-3)


# ── Simulation ───────────────────────────────────────────────────────────────

def _inputs(days=3, pv_peak=0.0, demand_scale=1.0):
    idx = pd.date_range("2025-06-02", periods=24 * days, freq="h", tz=TZ)
    h = idx.hour.to_numpy()
    demand = demand_scale * (0.8 * (h == 7) + 1.2 * (h == 20)).astype(float)
    pv = pv_peak * np.clip(np.cos((h - 12.5) / 6 * np.pi / 2), 0, None)
    return sv.Inputs(
        demand_kwh=pd.Series(demand, index=idx),
        surplus_kwh=pd.Series(pv, index=idx),
        spot_eur_mwh=pd.Series(np.nan, index=idx),
        hdo_blocked=pd.Series(False, index=idx),
    )


BOILER = BoilerParams(capacity_l=120, wattage_w=2000, set_tmp=60, min_tmp=40, area_tmp=20,
                      standby_loss_w=50)


def _sched(tariff=None):
    return HeatingScheduler(BOILER, tariff=tariff or Tariff(mode="fixed", fixed_price=4.0))


@pytest.mark.parametrize("name", sv.STRATEGIES)
def test_energy_balance(name):
    """Heat in = hot water out + standby loss + change of stored heat."""
    inp = _inputs()
    sched = _sched()
    r = sv.simulate(name, inp, sched)   # starts at set_tmp
    heat_in = r.electricity_kwh * BOILER.efficiency
    stored_change = sched._temp_drop_to_kwh(r.final_tmp - BOILER.set_tmp)
    assert heat_in == pytest.approx(inp.demand_kwh.sum() + r.standby_kwh + stored_change, abs=1e-6)


@pytest.mark.parametrize("name", sv.STRATEGIES)
def test_comfort_kept(name):
    assert sv.simulate(name, _inputs(), _sched()).hours_below_min == 0


def test_smart_keeps_water_cooler_and_loses_less():
    inp = _inputs()
    results = {r.name: r for r in sv.analyse(inp, BOILER, Tariff(mode="fixed", fixed_price=4.0))}
    assert results["smart_ideal"].mean_tmp < results["thermostat"].mean_tmp
    assert results["smart_ideal"].standby_kwh < results["thermostat"].standby_kwh
    assert results["smart_ideal"].cost <= results["thermostat"].cost


def test_smart_uses_more_free_solar():
    inp = _inputs(pv_peak=3.0)
    t = Tariff(mode="fixed", fixed_price=4.0, export_price=0.0)
    results = {r.name: r for r in sv.analyse(inp, BOILER, t)}
    assert results["smart_ideal"].pv_kwh > results["thermostat"].pv_kwh
    assert results["smart_ideal"].cost < results["thermostat"].cost


def test_cost_uses_import_and_export_prices():
    inp = _inputs(days=1)
    r = sv.simulate("thermostat", inp, _sched(Tariff(mode="fixed", fixed_price=2.0)))
    assert r.cost == pytest.approx(r.grid_kwh * 2.0)


def test_hdo_blocked_hours_never_heat():
    inp = _inputs(days=1)
    inp.hdo_blocked[:] = True
    r = sv.simulate("thermostat", inp, _sched())
    assert r.electricity_kwh == 0.0


def test_monthly_cost_sums_to_total():
    r = sv.simulate("smart_real", _inputs(days=5), _sched())
    assert sum(r.monthly_cost.values()) == pytest.approx(r.cost)


# ── Report and command line ──────────────────────────────────────────────────

def test_report_lists_all_strategies_and_saving():
    inp = _inputs()
    text = sv.format_report(sv.analyse(inp, BOILER, Tariff(mode="fixed", fixed_price=4.0)), inp, "Kč")
    for label in sv.LABELS.values():
        assert label in text
    assert "Saving vs thermostat" in text and "Kč" in text


def test_cli_end_to_end(tmp_path):
    idx = pd.date_range("2025-06-02", periods=48, freq="h", tz="UTC")
    litres = np.where(idx.hour.isin([5, 18]), 40.0, 0.0)
    pd.DataFrame({"entity_id": "sensor.water", "state": np.cumsum(litres),
                  "last_changed": idx.strftime("%Y-%m-%dT%H:%M:%SZ")}).to_csv(tmp_path / "w.csv", index=False)
    pd.DataFrame({"time": idx.astype(str), "value": np.cumsum(np.where(idx.hour.isin(range(8, 16)), 1.0, 0.0))}
                 ).to_csv(tmp_path / "pv.csv", index=False)
    (tmp_path / "cfg.json").write_text(json.dumps({"tariff_mode": "fixed", "fixed_price_per_kwh": 4.0}))
    out = tmp_path / "report.md"
    sv.main(["--config", str(tmp_path / "cfg.json"), "--water", str(tmp_path / "w.csv"),
             "--pv", str(tmp_path / "pv.csv"), "--out", str(out), "--currency", "Kč"])
    text = out.read_text()
    assert "# SmartBoiler savings analysis" in text and "2025-06" in text
