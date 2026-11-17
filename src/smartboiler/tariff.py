# Electricity tariff model: what one kWh costs to buy and earns when sold.
#
# The scheduler compares hours by the marginal cost of one kWh of
# electricity. This module turns the user's contract into that number:
#
#   import price  = energy price + grid fee        (what buying 1 kWh costs)
#   export price  = feed-in price                  (what exporting 1 kWh earns)
#
# Energy price depends on the contract type:
#   spot   — day-ahead spot price (EUR/MWh) converted to local currency per kWh
#   fixed  — one price for every hour
#   dual   — low / high price by time window (e.g. Czech NT / VT tariffs)
#
# All prices are in the user's local currency per kWh (e.g. CZK/kWh). Spot
# prices arrive in EUR/MWh and are converted with ``currency_per_eur``.
# Enter fees including VAT; the model does not add taxes on its own.

from dataclasses import dataclass
from typing import Optional, Set

# Used when the spot price for an hour is not (yet) known. Matches the
# historical scheduler default so behaviour without prices is unchanged.
DEFAULT_SPOT_EUR_MWH = 75.0

TARIFF_MODES = ("spot", "fixed", "dual")


def parse_time_windows(spec: str) -> Set[int]:
    """Return the hours of day (0–23) covered by ``spec``.

    ``spec`` is a comma-separated list of ``HH:MM-HH:MM`` windows, for
    example ``"22:00-06:00,13:00-15:00"``. Windows may wrap past midnight.
    An hour counts as covered when its midpoint (HH:30) lies inside a
    window, so ``12:05-13:35`` covers hours 12 and 13.

    Raises ValueError on malformed input; an empty string returns an
    empty set.
    """
    hours: Set[int] = set()
    for part in (p.strip() for p in (spec or "").split(",")):
        if not part:
            continue
        try:
            start_s, end_s = part.split("-")
            start = _minutes(start_s)
            end = _minutes(end_s)
        except ValueError as e:
            raise ValueError(f"Invalid time window '{part}' (expected HH:MM-HH:MM)") from e
        for h in range(24):
            mid = h * 60 + 30
            inside = start <= mid < end if start <= end else (mid >= start or mid < end)
            if inside:
                hours.add(h)
    return hours


def _minutes(hhmm: str) -> int:
    h, m = hhmm.strip().split(":")
    h, m = int(h), int(m)
    if not (0 <= h <= 24 and 0 <= m < 60):
        raise ValueError(hhmm)
    return h * 60 + m


@dataclass
class Tariff:
    """User's electricity contract. Defaults reproduce the pre-tariff
    behaviour: spot price in EUR, no fees, nothing paid for export."""

    mode: str = "spot"
    currency_per_eur: float = 1.0      # local currency per 1 EUR (1 = prices in EUR)
    grid_fee: float = 0.0              # local/kWh added to every imported kWh
    fixed_price: float = 0.0           # mode "fixed": energy price, local/kWh
    low_price: float = 0.0             # mode "dual": energy price in low-tariff hours
    high_price: float = 0.0            # mode "dual": energy price in other hours
    low_tariff_hours: Optional[Set[int]] = None  # mode "dual": hours 0–23; None → caller decides
    export_price: float = 0.0          # local/kWh earned for exported energy
    export_at_spot: bool = False       # True: export earns the spot price instead

    def spot_per_kwh(self, spot_eur_mwh: Optional[float]) -> float:
        """Spot price converted to local currency per kWh (default if unknown)."""
        eur_mwh = DEFAULT_SPOT_EUR_MWH if spot_eur_mwh is None else spot_eur_mwh
        return eur_mwh / 1000.0 * self.currency_per_eur

    def import_price(self, spot_eur_mwh: Optional[float], low_tariff: bool) -> float:
        """Cost of buying one kWh in an hour.

        Args:
            spot_eur_mwh: day-ahead price for the hour, or None if unknown.
            low_tariff: whether the hour is a low-tariff hour (mode "dual").
        """
        if self.mode == "fixed":
            energy = self.fixed_price
        elif self.mode == "dual":
            energy = self.low_price if low_tariff else self.high_price
        else:
            energy = self.spot_per_kwh(spot_eur_mwh)
        return energy + self.grid_fee

    def export_price_for(self, spot_eur_mwh: Optional[float]) -> float:
        """Income from exporting one kWh in an hour (can be negative at spot)."""
        if self.export_at_spot:
            return self.spot_per_kwh(spot_eur_mwh)
        return self.export_price

    @classmethod
    def from_config(cls, cfg: dict) -> "Tariff":
        """Build a Tariff from the setup config dict (see setup_config.DEFAULTS)."""
        low_spec = str(cfg.get("low_tariff_schedule", "") or "").strip()
        return cls(
            mode=cfg.get("tariff_mode", "spot"),
            currency_per_eur=float(cfg.get("currency_per_eur", 1.0)),
            grid_fee=float(cfg.get("grid_fee_per_kwh", 0.0)),
            fixed_price=float(cfg.get("fixed_price_per_kwh", 0.0)),
            low_price=float(cfg.get("low_price_per_kwh", 0.0)),
            high_price=float(cfg.get("high_price_per_kwh", 0.0)),
            low_tariff_hours=parse_time_windows(low_spec) if low_spec else None,
            export_price=float(cfg.get("export_price_per_kwh", 0.0)),
            export_at_spot=bool(cfg.get("export_at_spot", False)),
        )
