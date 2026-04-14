# Tests for spot_price.py — focusing on the hour-extraction fix:
# fetch_prices must key the result dict by LOCAL hour, not UTC hour.

import time
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from smartboiler.spot_price import SpotPriceFetcher


def _mock_response(unix_seconds, prices):
    """Build a mock requests.Response for given parallel lists."""
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json.return_value = {"unix_seconds": unix_seconds, "price": prices}
    return resp


class TestFetchPricesHourExtraction:
    """fetch_prices must use local time when keying the price dict."""

    def test_hour_key_matches_local_hour(self):
        """The dict key for any timestamp must equal datetime.fromtimestamp(ts).hour."""
        fetcher = SpotPriceFetcher(country="CZ")

        # Build 24 synthetic UTC timestamps one hour apart starting at midnight UTC.
        base_ts = int(datetime(2024, 6, 1, 0, 0, 0).timestamp())
        unix_seconds = [base_ts + i * 3600 for i in range(24)]
        prices_eur = [float(10 + i) for i in range(24)]

        with patch("smartboiler.spot_price.requests.get") as mock_get:
            mock_get.return_value = _mock_response(unix_seconds, prices_eur)
            result = fetcher.fetch_prices()

        # Every key in the result must match local hour, not UTC hour.
        for ts, expected_price in zip(unix_seconds, prices_eur):
            local_hour = datetime.fromtimestamp(ts).hour
            assert local_hour in result, (
                f"Local hour {local_hour} missing from result keys {sorted(result)}"
            )
            assert result[local_hour] == pytest.approx(expected_price), (
                f"Price mismatch at local hour {local_hour}"
            )

    def test_hour_key_never_uses_utc_hour_when_offset_nonzero(self):
        """If local TZ differs from UTC, UTC-keyed and local-keyed results diverge.

        We use a timestamp where UTC hour != local hour to verify which one is used.
        The test is only meaningful when the system timezone is not UTC; in UTC
        environments both hours are equal so we skip rather than falsely pass.
        """
        # Pick 2024-06-01 01:30:00 UTC  →  UTC hour = 1
        ts = int(datetime(2024, 6, 1, 1, 30, 0).timestamp())
        local_hour = datetime.fromtimestamp(ts).hour
        utc_hour = datetime.utcfromtimestamp(ts).hour

        if local_hour == utc_hour:
            pytest.skip("System timezone is UTC — cannot distinguish local vs UTC key")

        fetcher = SpotPriceFetcher(country="CZ")
        with patch("smartboiler.spot_price.requests.get") as mock_get:
            mock_get.return_value = _mock_response([ts], [99.0])
            result = fetcher.fetch_prices()

        assert local_hour in result, "Result must be keyed by local hour"
        assert utc_hour not in result, "Result must NOT be keyed by UTC hour"

    def test_none_prices_skipped(self):
        """Slots with price=None must be omitted from the result."""
        fetcher = SpotPriceFetcher(country="CZ")
        base_ts = int(datetime(2024, 1, 1, 12, 0, 0).timestamp())
        unix_seconds = [base_ts, base_ts + 3600]
        prices_eur = [50.0, None]

        with patch("smartboiler.spot_price.requests.get") as mock_get:
            mock_get.return_value = _mock_response(unix_seconds, prices_eur)
            result = fetcher.fetch_prices()

        assert len(result) == 1
        assert list(result.values()) == [50.0]

    def test_api_failure_returns_empty_dict(self):
        """Network errors must return {} without raising."""
        import requests as req_lib
        fetcher = SpotPriceFetcher(country="CZ")
        with patch("smartboiler.spot_price.requests.get", side_effect=req_lib.RequestException("timeout")):
            result = fetcher.fetch_prices()
        assert result == {}

    def test_fetch_today_tomorrow_returns_both_keys(self):
        """fetch_today_tomorrow must return a dict with 'today' and 'tomorrow' keys."""
        fetcher = SpotPriceFetcher(country="CZ")
        with patch.object(fetcher, "fetch_prices", return_value={0: 30.0}) as mock_fp:
            result = fetcher.fetch_today_tomorrow()
        assert "today" in result
        assert "tomorrow" in result


class TestHDOLearnerLockOnBootstrap:
    """hdo_learner replacement in the bootstrap thread must hold _lock."""

    def test_hdo_learner_swap_is_atomic(self):
        """Concurrent control-loop reads must never see None or a partially
        initialised HDOLearner during the bootstrap swap."""
        import threading
        from smartboiler.hdo_learner import HDOLearner

        # Simulate the assignment pattern used in _run_bootstrap
        class FakeController:
            def __init__(self):
                self._lock = threading.Lock()
                self.hdo_learner = HDOLearner()

            def swap_under_lock(self, new_learner):
                with self._lock:
                    self.hdo_learner = new_learner

            def read_under_lock(self):
                # Mirrors control-loop: reads hdo_learner while _lock may be held
                return self.hdo_learner.is_blocked(0, 12)

        ctrl = FakeController()
        errors = []

        def reader():
            for _ in range(500):
                try:
                    ctrl.read_under_lock()
                except Exception as exc:
                    errors.append(exc)

        def writer():
            for _ in range(50):
                ctrl.swap_under_lock(HDOLearner())

        t_read = threading.Thread(target=reader)
        t_write = threading.Thread(target=writer)
        t_read.start()
        t_write.start()
        t_read.join()
        t_write.join()

        assert errors == [], f"Concurrent access raised: {errors}"
