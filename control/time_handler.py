from datetime import datetime, timedelta
import calendar 



class TimeHandler:
    def __init__(self):
        self.daysofweek = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        print("------------------------------------------------------\n")
        print('initializing of TimeHandler\n')
        print("------------------------------------------------------\n")


    def float_to_time(self, float_number):
        return datetime.strptime('{0:02.0f}:{1:02.0f}'.format(*divmod(float_number * 60, 60)), "%H:%M")

    def this_day_string(self, days_delta=None):
        date = datetime.now()
        
        if days_delta is not None:
            date += timedelta(days=days_delta)
            
        return (calendar.day_name[date.weekday()]) 

    def hour_minutes_now(self):
        return datetime.strptime(datetime.now().strftime("%H:%M"), "%H:%M")

    def is_date_between(self, begin_date, end_date):
        check_date = datetime.datetime.now()   

        return begin_date <= check_date <= (end_date - timedelta(hours = 1))
    
    def date_to_datetime(self, date):
        return datetime.strptime(date, "%Y-%m-%dT%H:%M:%S+01:00")

