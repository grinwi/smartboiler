from scipy.signal import argrelextrema, find_peaks, peak_widths
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
        self.empty_intervals = self._find_empty_intervals(data)

    def _find_empty_intervals(self, data):
        df_grouped = data.groupby([data.index.hour, data.index.minute]).mean()
        df_grouped = df_grouped.to_frame()

        df_mask = pd.DataFrame({'tmp1': 1000} ,index=pd.date_range('2018-03-01 00:00:00', 
                            periods=1440, freq='1min'))
        df_mask = df_mask.groupby([df_mask.index.hour,df_mask.index.minute], dropna=False).mean()

        df_mask.tmp1 = df_grouped.tmp1
        df_mapped = df_mask
        first_none = False

        day_out_of_service_dict = {}
        i = 0

        for index, row in df_mapped.iterrows():
            if(np.isnan( row['tmp1'])):
                if (first_none == False):
                    first_none = True
                    start_of_none = self.TimeHandler.float_to_time(index[0] + index[1]/60)
                else:
                    continue
            else:
                if first_none:
                    end_of_none = self.TimeHandler.float_to_time(index[0] + index[1]/60)
                    duration = end_of_none - start_of_none
                    day_out_of_service_dict.update({i:{"start": start_of_none, "end":end_of_none, "duration":duration}})
                    i += 1
                    first_none = False

        return day_out_of_service_dict
    def _create_days_average(self, data):


        new_week_days_consumptions = dict.fromkeys(self.TimeHandler.daysofweek, 0)
        
        prepared_data = self._prepare_data(data)
        #specialni height pro kazdy den tydne zvlast

        for idx, value in enumerate(self.TimeHandler.daysofweek):

            x = prepared_data[prepared_data.day_of_week == idx].tmp1.reset_index(drop=True)

            peaks, _ = self._find_ideal_peaks(x)
        
            results_half = peak_widths(x, peaks, rel_height=0.7)

            day_aconsumptions = {}
            for i in range(len(results_half[0])):
                day_aconsumptions.update({i:{   "start":self.TimeHandler.float_to_time(results_half[2][i]), 
                                                "end":self.TimeHandler.float_to_time(results_half[3][i]), 
                                                "duration" : results_half[0][i], 
                                                "peak":round(_['peak_heights'][i],2)}})

            new_week_days_consumptions.update({value:day_aconsumptions})

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

        print("created new week plan!")
        
        return self.week_days_consumptions

    def _empty_intervals(self, data_from_db = None):
        if (data_from_db is not None):
            self.empty_intervals = self._find_empty_intervals(data_from_db)

        print('empty intervals found')

        return self.empty_intervals


