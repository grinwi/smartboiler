from scipy.signal import argrelextrema, find_peaks, peak_widths
from datetime import datetime, timedelta, date
from time_handler import TimeHandler

import pandas as pd
import numpy as np
class WeekPlanner:

    def __init__(self, data):
        print("------------------------------------------------------\n")
        print('initializing of WeekPlanner\n')
        print("------------------------------------------------------\n")        
        self.TimeHandler = TimeHandler()
        self.week_days_consumptions = self._create_days_average(data)
        self.week_days_high_tarifs_intervals = self._find_empty_intervals(data)

    def _find_empty_intervals(self, data):
            
        time_interval_int = 5
        time_interval = str(time_interval_int) + 'min'
        #grouping data from db by time interval in minutes
        #data = data.to_frame()

        
        data = data[data.index > (data.last_valid_index() - timedelta(days=10))]

        data = data.groupby(pd.Grouper(freq=time_interval)).aggregate(np.sum)

        #grouping data grouped by time interval by dayofweek, hour and minut
        df_grouped = data.groupby([data.index.dayofweek, data.index.hour, data.index.minute], dropna=True).sum()


        #start and of measuring for creating an emptu dataframe with rows for all times by 5 minuts
        start = data.first_valid_index()
        end = data.last_valid_index()
        #new empty dataframe
        df = pd.DataFrame({'tmp2': -1} ,index=pd.date_range(start, 
                            end, freq='1min'))
        #grouped by same time interval as dataframe from db and then by day of week, hour and minute
        df = df.groupby(pd.Grouper(freq=time_interval)).aggregate(np.sum)
        df = df.groupby([df.index.dayofweek, df.index.hour, df.index.minute], dropna = False).sum()

        #adding of values from db into an empty dataframe
        df.tmp2 = df_grouped.tmp2


        week_high_tarifs =  {}

        #for each of day of week finds intervals of high tarif
        for idx in range(7):    

            i = 0   #index of interval of the day

            day_high_tarifs = {}
            first_none = False
            for index, row in df.iloc[df.index.get_level_values(level=0) ==  idx].iterrows():


                day_of_week = index[0]
                hour = index[1]
                minute = index[2]

                tmp1 = row[0]
                    
                if(np.isnan(tmp1)):
                    if (first_none == False):
                        first_none = True
                        start_of_none = timedelta(hours=(hour + (minute/60))) 

                        minute -= time_interval_int

                        if minute < 0:
                            minute += time_interval_int
                            hour -= 1 
                            if hour < 0:
                                hour += 1
                                day_of_week = (day_of_week+1) % 7
                        previous_index = (day_of_week,hour,minute)

                        tmp_before_start = df.loc[previous_index][0]
                    
                else:
                    if first_none:
                        end_of_none = timedelta(hours=(hour + (minute/60)))
                        duration = (end_of_none - start_of_none)

                        end_of_none = (datetime.min + end_of_none).time()
                        start_of_none = (datetime.min + start_of_none).time()

                        minute += time_interval_int

                        if minute >= 60:
                            minute -= time_interval_int
                            hour += 1 
                            if hour > 23:
                                hour -= 1
                                day_of_week = (day_of_week-1) % 7

                        next_index = (day_of_week,hour,minute)
                        tmp_after_end = df.loc[next_index][0]

                        tmp_delta = tmp_before_start - tmp_after_end

                


                        day_high_tarifs.update({i:{"start": start_of_none, "end":end_of_none, "duration":duration, "tmp_delta":tmp_delta}})
                        i += 1
                        first_none = False

            #print(day_high_tarifs)
            week_high_tarifs.update({idx:day_high_tarifs})
        print(week_high_tarifs)
        return week_high_tarifs


    def _create_days_average(self, data):


        new_week_days_consumptions = dict.fromkeys(self.TimeHandler.daysofweek, 0)
        
        prepared_data = self._prepare_data(data)
        #specialni height pro kazdy den tydne zvlast

        for idx in range(7):

            x = prepared_data[prepared_data.day_of_week == idx].tmp1.reset_index(drop=True)

            peaks, _ = self._find_ideal_peaks(x)
        
            results_half = peak_widths(x, peaks, rel_height=0.7)

            day_aconsumptions = {}
            for i in range(len(results_half[0])):
                day_aconsumptions.update({i:{   "start":self.TimeHandler.float_to_time(results_half[2][i]), 
                                                "end":self.TimeHandler.float_to_time(results_half[3][i]), 
                                                "duration" : results_half[0][i], 
                                                "peak":round(_['peak_heights'][i],2)}})

            new_week_days_consumptions.update({idx:day_aconsumptions})

        print("created new days average")

        return new_week_days_consumptions

    def _find_ideal_peaks(self, x):
        height = 20
        distance = 3
        peaks, _ = find_peaks(x, height=height, distance = distance)

        while(len(peaks) > 3):
            height += 1
            distance += 0.2
            peaks, _ = find_peaks(x, height=height, distance=distance)
          
        return(peaks, _)

    def _prepare_data(self, data):
        df_grouped = data.groupby([data.index.dayofweek, data.index.hour]).mean().reset_index()
        return df_grouped.rename(columns={'level_0': 'day_of_week', 'level_1': 'hour'})

    def week_plan(self, data_from_db = None):

        """Accepts data from DB and returns plan for week based on days of week average.
        If data wasn't specified, returns last created week plan.

        Args:
            data ([result of query from dB]): data from DB grouped by minute
        """

        if ((data_from_db is not None)):
            self.week_days_consumptions = self._create_days_average(data_from_db)
            self.week_days_high_tarifs_intervals = self._find_empty_intervals(data_from_db)


        print("created new week plan!")
        
        return self.week_days_consumptions

    def is_in_DTO(self):

        time_to_next_DTO_start = self.next_high_tarif_interval('start')['next_high_tarif_in']
        time_to_next_DTO_end = self.next_high_tarif_interval('end')['next_high_tarif_in']

        return time_to_next_DTO_start > time_to_next_DTO_end
    def next_high_tarif_interval(self, event):

        actual_time = datetime.now().time()


        day_of_week = datetime.now().weekday()
        days_plus = 0

        while(days_plus < 7):
        
            day_high_tarifs_intervals = self.week_days_high_tarifs_intervals[day_of_week]

            for  key, item in   day_high_tarifs_intervals.items():
                next_time = item[event]

                next_actual_time_delta = datetime.combine(date.min, next_time)- datetime.combine(date.min, actual_time) 
                

                if (next_actual_time_delta >= timedelta(minutes=0)):
                    time_to_next_heating_event = (next_actual_time_delta + timedelta(days = days_plus))  / timedelta(hours=1)

                    return {"next_high_tarif_in" : time_to_next_heating_event, "tmp_delta" : item['tmp_delta']}

            actual_time = actual_time.replace(hour=0, minute=0)
            days_plus += 1
            day_of_week = (day_of_week + 1) % 7
        return None

    def duration_of_low_tarif_to_next_heating(self, hours_to_next_heating):
    #time in hours float

        datetime_now = datetime.now()
        time_now = datetime_now.time()
        day_time_start = time_now

        hours_to_next_heating=timedelta(hours=hours_to_next_heating)

        day_of_week = datetime_now.weekday()

        week_high_tarifs_intervals = self.week_days_high_tarifs_intervals
        
        time_added = timedelta(hours=0)
        time_of_high_tarif = timedelta(hours=0)


        while((time_added < hours_to_next_heating) and day_of_week < 7):
            day_high_tarifs_intervals = week_high_tarifs_intervals[day_of_week]
            if(day_high_tarifs_intervals is not None):

                added_in_day = timedelta(hours=0)

                for j in range(len(day_high_tarifs_intervals)):
                    actual_interval = day_high_tarifs_intervals[j]

                    if time_now > actual_interval['end']:
                        continue
                    
                    added_in_day += datetime.combine(date.min, actual_interval['start']) - datetime.combine(date.min, time_now) 

                
                    if ((time_now < actual_interval['end']) and (time_now > actual_interval['start'])):
                        time_of_high_tarif += datetime.combine(date.min, actual_interval['end']) - datetime.combine(date.min, time_now)

                    elif added_in_day + time_added < hours_to_next_heating:
                            time_of_high_tarif += actual_interval['duration']


                    time_now = actual_interval['end']
                        
            time_added += datetime.combine(date.min, time_now.replace(hour = 23, minute=59)) - datetime.combine(date.min, day_time_start)
            day_of_week = (day_of_week + 1) % 7
            time_now = time_now.replace(hour = 0, minute=0)
            day_time_start = time_now


        return (hours_to_next_heating - time_of_high_tarif) / timedelta(hours=1)

    def _empty_intervals(self, data_from_db = None):
        if (data_from_db is not None):
            self.empty_intervals = self._find_empty_intervals(data_from_db)

        print('empty intervals found')

        return self.empty_intervals


