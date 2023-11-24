#################################################################
# Bachelor's thesis                                             #
# From a dumb boiler to a smart one using a smart socket        #
# Author: Adam Grünwald                                         #
# BUT FIT BRNO, Faculty of Information Technology               #
# 26/6/2021                                                     #
#                                                               #
# Module with class for maniulating with time and date  formats #
#################################################################
from datetime import datetime, timedelta
import calendar 



class TimeHandler:
    def __init__(self):
        self.daysofweek = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        print("------------------------------------------------------\n")
        print('initializing of TimeHandler\n')
        print("------------------------------------------------------\n")


    def float_to_time(self, float_number):
        return datetime.strptime('{0:02.0f}:{1:02.0f}'.format(*divmod(float_number * 60, 60)), "%H:%M%f")


    def hour_minutes_now(self, dtime = None):
        """Returns actual hour and minute

        Args:
            dtime ([DateTime], optional): [Actual DateTime]. Defaults to None.

        Returns:
            [type]: [Hour and minute]
        """
        if dtime is None:
            dtime = datetime.now()
        return dtime.strptime(dtime.strftime("%H:%M"), "%H:%M")


    def is_date_between(self, begin_date, end_date):
        """ Returns boolean value describing presence actual time between two times.

        Args:
            begin_date ([type]): [time of beginning]
            end_date ([type]): [time of end]

        Returns:
            [boolean]: [if is date between - True]
        """
        # addition because google calendar api returns utc
        check_date = datetime.now() + timedelta(hours = 1)

        # one hour before the ond of vacation should be water preheated
        return begin_date <= check_date <= (end_date - timedelta(hours = 1))

    def date_from_influxdb_to_datetime(self, date_from_db):
        return datetime.strptime(date_from_db, "%Y-%m-%dT%H:%M:%SZ")
    def date_to_datetime(self, date):
        return datetime.strptime(date, "%Y-%m-%dT%H:%M:%S+01:00")

