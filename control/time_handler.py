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

    def this_day_string(self, days_delta=None):
        date = datetime.now()
        
        if days_delta is not None:
            date += timedelta(days=days_delta)
            
        return (calendar.day_name[date.weekday()]) 

    def hour_minutes_now(self, dtime = None):
        if dtime is None:
            dtime = datetime.now()
        return dtime.strptime(dtime.strftime("%H:%M"), "%H:%M")


    #funkce pro kalendar Google, jez ma data v SEC, nutno pricist hodinu k datetime.now()
    def is_date_between(self, begin_date, end_date):
        check_date = datetime.now() + timedelta(hours = 1)

        #hodinu pred koncem udalosti uz ridit normalne, aby se voda stihla pripadne nahrat
        return begin_date <= check_date <= (end_date - timedelta(hours = 1))

    def date_from_influxdb_to_datetime(self, date_from_db):
        return datetime.strptime(date_from_db, "%Y-%m-%dT%H:%M:%SZ")
    def date_to_datetime(self, date):
        return datetime.strptime(date, "%Y-%m-%dT%H:%M:%S+01:00")

