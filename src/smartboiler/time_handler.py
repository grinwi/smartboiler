# Created as a part of Master's Thesis "Using machine learning methods to save energy in a smart home"
# Faculty of Information Technology, Brno University of Technology, 2024
# Author: Adam Grünwald
#
# This module is used for manipulating with time and date formats.

from datetime import datetime, timedelta, timezone
from dateutil import parser


class TimeHandler:
    """Class for manipulating with time and date formats."""

    def __init__(self):
        """Initialize the class with the days of the week."""
        self.daysofweek = [
            "Monday",
            "Tuesday",
            "Wednesday",
            "Thursday",
            "Friday",
            "Saturday",
            "Sunday",
        ]

    def is_date_between(self, begin_date: datetime, end_date: datetime) -> bool:
        """Method for checking if the actual date is between the begin and end date.

        Args:
            begin_date (datetime): First date.
            end_date (datetime): Second date

        Returns:
            bool: If the actual date is between the begin and end date returns True, otherwise False.
        """
        check_date = datetime.now(timezone.utc)

        # one hour before the end of vacation should be water preheated
        return begin_date <= check_date <= (end_date - timedelta(hours=1))

    def date_to_datetime(self, date: str) -> datetime:
        """Method for converting date to datetime format.

        Args:
            date (str): Date in string format.

        Returns:
            datetime: Datetime object from string date representation.
        """
        return parser.parse(date)
