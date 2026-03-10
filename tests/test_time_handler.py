"""Unit tests for TimeHandler (pure datetime logic, no network required)."""
import pytest
from datetime import datetime, timedelta, timezone

from smartboiler.time_handler import TimeHandler


@pytest.fixture
def th() -> TimeHandler:
    return TimeHandler()


# ---------------------------------------------------------------------------
# is_date_between
# ---------------------------------------------------------------------------

class TestIsDateBetween:
    def test_returns_true_when_now_is_between(self, th):
        begin = datetime.now(timezone.utc) - timedelta(hours=1)
        end = datetime.now(timezone.utc) + timedelta(hours=2)
        assert th.is_date_between(begin, end) is True

    def test_returns_false_when_now_is_before_begin(self, th):
        begin = datetime.now(timezone.utc) + timedelta(hours=1)
        end = datetime.now(timezone.utc) + timedelta(hours=3)
        assert th.is_date_between(begin, end) is False

    def test_returns_false_when_now_is_after_end_minus_1h(self, th):
        # The function subtracts 1 hour from end_date before comparing
        begin = datetime.now(timezone.utc) - timedelta(hours=3)
        end = datetime.now(timezone.utc) + timedelta(minutes=30)
        # now > end - 1h  =>  False
        assert th.is_date_between(begin, end) is False

    def test_returns_true_just_before_cutoff(self, th):
        begin = datetime.now(timezone.utc) - timedelta(hours=1)
        end = datetime.now(timezone.utc) + timedelta(hours=1, minutes=1)
        # now < end - 1h  =>  True
        assert th.is_date_between(begin, end) is True


# ---------------------------------------------------------------------------
# date_to_datetime
# ---------------------------------------------------------------------------

class TestDateToDatetime:
    def test_parses_iso_string(self, th):
        result = th.date_to_datetime("2024-06-15T10:30:00")
        assert result.year == 2024
        assert result.month == 6
        assert result.day == 15
        assert result.hour == 10
        assert result.minute == 30

    def test_parses_date_only_string(self, th):
        result = th.date_to_datetime("2023-01-01")
        assert result.year == 2023
        assert result.month == 1
        assert result.day == 1

    def test_parses_string_with_timezone(self, th):
        result = th.date_to_datetime("2024-06-15T10:30:00+02:00")
        assert result.year == 2024
        assert result.month == 6
        assert result.day == 15

    def test_returns_datetime_object(self, th):
        result = th.date_to_datetime("2024-01-01")
        assert isinstance(result, datetime)
