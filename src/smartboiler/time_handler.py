# Created as a part of Master's Thesis "Using machine learning methods to save energy in a smart home"
# Faculty of Information Technology, Brno University of Technology, 2024
# Author: Adam Grünwald
#
# This module is used for manipulating with time and date formats.

from datetime import datetime, timedelta
from typing import Optional


class TimeHandler:
    """Class for manipulating with time and date formats.
    """
    def __init__(self):
        """Initialize the class with the days of the week.
        """
        self.daysofweek = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

    def float_to_time(self, float_number: float)-> datetime:
        """Method for converting float number to time format.

        Args:
            float_number (float): Number to convert.

        Returns:
            datetime: DateTime object.
        """
        return datetime.strptime('{0:02.0f}:{1:02.0f}'.format(*divmod(float_number * 60, 60)), "%H:%M%f")

    def hour_minutes_now(self, date_time: Optional[datetime] = None) -> datetime:
        """Method for getting actual hour and minute.

        Args:
            date_time ([DateTime], optional): Datetime object. Defaults to None.

        Returns:
            [type]: [Hour and minute]
        """
        if date_time is None:
            date_time = datetime.now()
        return date_time.strptime(date_time.strftime("%H:%M"), "%H:%M")


    def is_date_between(self, begin_date:datetime, end_date:datetime) -> bool:
        """Method for checking if the actual date is between the begin and end date.

        Args:
            begin_date (datetime): First date.
            end_date (datetime): Second date

        Returns:
            bool: If the actual date is between the begin and end date returns True, otherwise False.
        """
        # addition because google calendar api returns utc
        check_date = datetime.now() + timedelta(hours = 1)

        # one hour before the ond of vacation should be water preheated
        return begin_date <= check_date <= (end_date - timedelta(hours = 1))

    def date_from_influxdb_to_datetime(self, date_from_db: str) -> datetime:
        """Method for converting date from InfluxDB to datetime format.

        Args:
            date_from_db (str): String with date from InfluxDB.

        Returns:
            datetime: Datetime object of the date.
        """
        return datetime.strptime(date_from_db, "%Y-%m-%dT%H:%M:%SZ")
    def date_to_datetime(self, date: str) -> datetime:
        """Method for converting date to datetime format.

        Args:
            date (str): Date in string format.

        Returns:
            datetime: Datetime object from string date representation.
        """
        return datetime.strptime(date, "%Y-%m-%dT%H:%M:%S+01:00")

