# Created as a part of Master's Thesis "Using machine learning methods to save energy in a smart home"
# Faculty of Information Technology, Brno University of Technology, 2024
# Author: Adam Grünwald
#
# Day-ahead electricity spot price fetcher using energy-charts.info free API.

import logging
from datetime import date, datetime, timedelta
from typing import Dict, Optional

import requests

logger = logging.getLogger(__name__)

BASE_URL = "https://api.energy-charts.info/price"

# energy-charts.info bidding zone codes
COUNTRY_TO_BZN = {
    "CZ": "cz",
    "SK": "sk",
    "AT": "at",
    "DE": "de",
    "PL": "pl",
    "HU": "hu",
    "FR": "fr",
    "IT": "it",
    "ES": "es",
}


class SpotPriceFetcher:
    """Fetches day-ahead electricity prices from energy-charts.info (free, no auth)."""

    def __init__(self, country: str = "CZ", timeout: int = 15):
        self.bzn = COUNTRY_TO_BZN.get(country.upper(), "cz")
        self.timeout = timeout

    def fetch_prices(self, for_date: Optional[date] = None) -> Dict[int, float]:
        """Fetch hourly spot prices for a given date.

        Returns:
            Dict mapping hour (0-23) → price in EUR/MWh.
            Empty dict on failure or unavailable data.
        """
        if for_date is None:
            for_date = date.today()

        start = datetime.combine(for_date, datetime.min.time()).strftime(
            "%Y-%m-%dT%H:%M+00:00"
        )
        end = datetime.combine(
            for_date + timedelta(days=1), datetime.min.time()
        ).strftime("%Y-%m-%dT%H:%M+00:00")

        try:
            resp = requests.get(
                BASE_URL,
                params={"bzn": self.bzn, "start": start, "end": end},
                timeout=self.timeout,
            )
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as e:
            logger.warning("Could not fetch spot prices for %s: %s", for_date, e)
            return {}
        except Exception as e:
            logger.warning("Unexpected error fetching spot prices: %s", e)
            return {}

        prices: Dict[int, float] = {}
        unix_seconds = data.get("unix_seconds", [])
        price_eur = data.get("price", [])
        for ts, price in zip(unix_seconds, price_eur):
            if price is None:
                continue
            # Use local time so the hour key matches local wall-clock hours used
            # by the scheduler (controller indexes prices by now_dt.hour which is
            # local). utcfromtimestamp would produce UTC hours and shift every
            # price slot by the UTC offset (e.g. -1/-2 h for CZ).
            hour = datetime.fromtimestamp(ts).hour
            prices[hour] = float(price)

        return prices

    def fetch_series(self, start: datetime, end: datetime, chunk_days: int = 30):
        """Hourly prices (EUR/MWh) between two aware datetimes as a pandas Series
        with a UTC index — for replaying history. 15-minute prices are averaged
        to hours. Chunks that fail are skipped (logged), so gaps stay NaN."""
        import pandas as pd

        times, values = [], []
        t = start
        while t < end:
            t_next = min(end, t + timedelta(days=chunk_days))
            try:
                resp = requests.get(
                    BASE_URL,
                    params={"bzn": self.bzn, "start": t.isoformat(), "end": t_next.isoformat()},
                    timeout=self.timeout,
                )
                resp.raise_for_status()
                data = resp.json()
                for ts, price in zip(data.get("unix_seconds", []), data.get("price", [])):
                    if price is not None:
                        times.append(ts)
                        values.append(float(price))
            except Exception as e:
                logger.warning("Could not fetch spot prices %s–%s: %s", t, t_next, e)
            t = t_next
        s = pd.Series(values, index=pd.to_datetime(times, unit="s", utc=True), dtype=float)
        return s.sort_index().resample("1h").mean() if len(s) else s

    def fetch_today_tomorrow(self) -> Dict[str, Dict[int, float]]:
        """Fetch prices for today and tomorrow (tomorrow may not be available yet)."""
        return {
            "today": self.fetch_prices(date.today()),
            "tomorrow": self.fetch_prices(date.today() + timedelta(days=1)),
        }

    def get_next_24h_prices(self, from_hour: int = 0) -> Dict[int, Optional[float]]:
        """Return prices for hours 0-23, spanning today+tomorrow if needed."""
        today = self.fetch_prices(date.today())
        tomorrow = self.fetch_prices(date.today() + timedelta(days=1))
        result: Dict[int, Optional[float]] = {}
        for i in range(24):
            hour = (from_hour + i) % 24
            day_offset = (from_hour + i) // 24
            if day_offset == 0:
                result[i] = today.get(hour)
            else:
                result[i] = tomorrow.get(hour)
        return result

    @staticmethod
    def categorize(price_eur_mwh: float) -> str:
        """Categorize a price as cheap/medium/expensive."""
        if price_eur_mwh < 50:
            return "cheap"
        if price_eur_mwh < 100:
            return "medium"
        return "expensive"
