"""Tests for the tariff model and its config validation / scheduler wiring."""
from datetime import datetime

import pytest

from smartboiler.scheduler import BoilerParams, HeatingScheduler
from smartboiler.setup_config import DEFAULTS, validate_config
from smartboiler.tariff import DEFAULT_SPOT_EUR_MWH, Tariff, parse_time_windows

BASE_DT = datetime(2024, 3, 11, 0, 0)  # Monday 00:00


# ── parse_time_windows ───────────────────────────────────────────────────────

class TestParseTimeWindows:
    def test_empty(self):
        assert parse_time_windows("") == set()

    def test_simple_window(self):
        assert parse_time_windows("13:00-15:00") == {13, 14}

    def test_wraps_midnight(self):
        assert parse_time_windows("22:00-06:00") == {22, 23, 0, 1, 2, 3, 4, 5}

    def test_multiple_windows_and_spaces(self):
        assert parse_time_windows(" 01:00-02:00 , 13:00-14:00 ") == {1, 13}

    def test_hour_counts_by_midpoint(self):
        # 12:05-13:35 contains 12:30 and 13:30
        assert parse_time_windows("12:05-13:35") == {12, 13}

    def test_window_shorter_than_half_hour_covers_nothing(self):
        assert parse_time_windows("10:00-10:20") == set()

    @pytest.mark.parametrize("bad", ["22-06", "25:00-06:00", "abc", "10:00-11:00-12:00"])
    def test_malformed_raises(self, bad):
        with pytest.raises(ValueError):
            parse_time_windows(bad)


# ── Tariff prices ────────────────────────────────────────────────────────────

class TestTariff:
    def test_defaults_match_legacy_spot_in_eur(self):
        t = Tariff()
        assert t.import_price(42.0, low_tariff=False) == pytest.approx(0.042)
        assert t.import_price(None, low_tariff=False) == pytest.approx(DEFAULT_SPOT_EUR_MWH / 1000)
        assert t.export_price_for(42.0) == 0.0

    def test_spot_converted_to_local_currency_plus_fee(self):
        t = Tariff(mode="spot", currency_per_eur=25.0, grid_fee=2.0)
        # 100 EUR/MWh = 0.1 EUR/kWh = 2.5 CZK/kWh, + 2.0 fee
        assert t.import_price(100.0, low_tariff=False) == pytest.approx(4.5)

    def test_negative_spot_price_passes_through(self):
        t = Tariff(mode="spot", currency_per_eur=25.0, grid_fee=1.0)
        assert t.import_price(-40.0, low_tariff=False) == pytest.approx(0.0)

    def test_fixed_ignores_spot(self):
        t = Tariff(mode="fixed", fixed_price=3.0, grid_fee=1.5)
        assert t.import_price(500.0, low_tariff=False) == pytest.approx(4.5)
        assert t.import_price(None, low_tariff=True) == pytest.approx(4.5)

    def test_dual_uses_low_and_high(self):
        t = Tariff(mode="dual", low_price=2.0, high_price=4.0, grid_fee=1.0)
        assert t.import_price(None, low_tariff=True) == pytest.approx(3.0)
        assert t.import_price(None, low_tariff=False) == pytest.approx(5.0)

    def test_export_fixed_and_at_spot(self):
        assert Tariff(export_price=1.2).export_price_for(300.0) == pytest.approx(1.2)
        t = Tariff(currency_per_eur=25.0, export_at_spot=True)
        assert t.export_price_for(-20.0) == pytest.approx(-0.5)

    def test_from_config_reads_all_fields(self):
        cfg = dict(DEFAULTS, tariff_mode="dual", currency_per_eur=25, grid_fee_per_kwh=1,
                   low_price_per_kwh=2, high_price_per_kwh=4, low_tariff_schedule="22:00-06:00",
                   export_price_per_kwh=0.5)
        t = Tariff.from_config(cfg)
        assert t.mode == "dual"
        assert t.low_tariff_hours == {22, 23, 0, 1, 2, 3, 4, 5}
        assert t.export_price == pytest.approx(0.5)

    def test_from_config_empty_schedule_means_hdo_decides(self):
        assert Tariff.from_config(dict(DEFAULTS)).low_tariff_hours is None


# ── Config validation ────────────────────────────────────────────────────────

def _valid(**kw) -> dict:
    cfg = dict(DEFAULTS, boiler_switch_entity_id="switch.boiler")
    cfg.update(kw)
    return cfg


class TestTariffValidation:
    def test_defaults_valid(self):
        assert validate_config(_valid()) == []

    def test_unknown_mode(self):
        assert any("tariff_mode" in e for e in validate_config(_valid(tariff_mode="cheap")))

    def test_fixed_needs_price(self):
        assert any("fixed_price_per_kwh" in e for e in validate_config(_valid(tariff_mode="fixed")))
        assert validate_config(_valid(tariff_mode="fixed", fixed_price_per_kwh=4.2)) == []

    def test_dual_needs_both_prices(self):
        errs = validate_config(_valid(tariff_mode="dual", low_price_per_kwh=2.0))
        assert any("high_price_per_kwh" in e for e in errs)

    def test_dual_bad_schedule(self):
        errs = validate_config(_valid(tariff_mode="dual", low_price_per_kwh=2, high_price_per_kwh=4,
                                      low_tariff_schedule="nonsense"))
        assert any("low_tariff_schedule" in e for e in errs)

    def test_export_at_spot_needs_spot_prices(self):
        assert any("export_at_spot" in e for e in validate_config(_valid(export_at_spot=True)))
        assert validate_config(_valid(export_at_spot=True, has_spot_price=True)) == []


# ── Scheduler uses the tariff ────────────────────────────────────────────────

def _sched(tariff: Tariff) -> HeatingScheduler:
    return HeatingScheduler(BoilerParams(wattage_w=2000.0), tariff=tariff)


class TestSchedulerWithTariff:
    def test_dual_low_tariff_hours_are_preferred(self):
        t = Tariff(mode="dual", low_price=2.0, high_price=5.0, low_tariff_hours={2, 3})
        plan, slots = _sched(t).plan(36.0, [0.0] * 24, [0.0] * 24, {}, [False] * 24, BASE_DT)
        assert slots[2].import_price == pytest.approx(2.0)
        assert slots[10].import_price == pytest.approx(5.0)
        heated = [i for i, h in enumerate(plan) if h]
        assert heated and set(heated) <= {0, 1, 2, 3}

    def test_dual_without_schedule_treats_hdo_free_hours_as_low(self):
        t = Tariff(mode="dual", low_price=2.0, high_price=5.0)
        hdo = [i >= 6 for i in range(24)]   # available 00–06 only
        slots = _sched(t).build_slots([0.0] * 24, [0.0] * 24, {}, hdo, BASE_DT)
        assert slots[0].import_price == pytest.approx(2.0)
        assert slots[7].import_price == pytest.approx(5.0)

    def test_paid_feed_in_above_grid_price_does_not_force_pv(self):
        # Export earns more than the cheapest grid hour → keep exporting.
        t = Tariff(mode="fixed", fixed_price=1.0, export_price=2.0)
        pv = [0.0] * 24
        pv[12] = 3.0
        plan, _ = _sched(t).plan(58.0, [0.0] * 24, pv, {}, [False] * 24, BASE_DT)
        assert plan[12] is False

    def test_free_pv_is_forced_with_fixed_tariff(self):
        t = Tariff(mode="fixed", fixed_price=4.0, export_price=0.0)
        pv = [0.0] * 24
        pv[12] = 3.0
        plan, _ = _sched(t).plan(58.0, [0.0] * 24, pv, {}, [False] * 24, BASE_DT)
        assert plan[12] is True

    def test_negative_price_hour_is_forced(self):
        prices = {i: 50.0 for i in range(24)}
        prices[4] = -30.0
        plan, _ = _sched(Tariff()).plan(58.0, [0.0] * 24, [0.0] * 24, prices, [False] * 24, BASE_DT)
        assert plan[4] is True
