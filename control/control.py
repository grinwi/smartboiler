"""
Module that controls a heating of smart boiler. 

Uses module Bojler for computing time needed to heating of water, 
module TimeHandler for basic time and date operations, 
WeekPlanner for plan week heating, 
SettingsLoader to load setting from settings file and EventChecker which checks events in calendar, 
when the water shouldnt be heated.

"""

from datetime import datetime, timedelta
import pandas as pd
import calendar 
from influxdb import DataFrameClient
from influxdb import InfluxDBClient
import os.path
import signal
import sys

import time
import json
import requests



from scipy.misc import electrocardiogram
import numpy as np


from bojler import Bojler
from time_handler import TimeHandler
from week_planner import WeekPlanner
from event_checker import EventChecker
#################################################
#######################TODO######################
#################################################
#jednou za 14 dni ohrat pred odberem na max

class Controller:
    """Main and only class which decides about heating
    """


    def __init__(self, settings_file = 'settings.json'):
        """Initializes controller with settings from SettingsLoader

        Args:


            db_name ([type]): [description]
            socket_url ([type]): [description]
            tmp_output (str, optional): [description]. Defaults to 'tmp1'.
            tmp_bojler (str, optional): [description]. Defaults to 'tmp2'.
            host (str, optional): [description]. Defaults to 'influxdb'.
            port (int, optional): [description]. Defaults to 8086.
            bojler_capacity (int, optional): [description]. Defaults to 100.
            bojler_wattage (int, optional): [description]. Defaults to 2000.
        """
        from settings_loader import SettingsLoader
        SettingsLoader = SettingsLoader(settings_file)
        settings = SettingsLoader.load_settings()

        self.host = settings['db_host']
        self.port = settings['db_port']        
        self.db_name = settings['db_name']
        self.measurement = settings['measurement']
        
        self.socket_url = settings['socket_url']
        

        self.tmp_output = settings['tmp_output']
        self.tmp_bojler = settings['tmp_bojler']
        self.tmp_min = settings['tmp_min']
        self.consumption_tmp_min = settings['consumption_tmp_min']
        self.heating_coef = settings['heating_coef']

        bojler_wattage = settings['bojler_wattage']
        bojler_capacity = settings['bojler_capacity']
        bojler_set_tmp = settings['bojler_set_tmp']

        print("------------------------------------------------------\n")
        print('initializing of Control...\n\tdb_name = {}\n\tsocker_url = {}\n\ttmp_output = {}\n\ttmp_bojler = {}\n\thost name = {}\n\tport = {}\n\tbojler capacity = {}\n\tbojler woltage = {}\n'.format(self.db_name, self.socket_url, self.tmp_output, self.tmp_bojler, self.host, self.port, bojler_capacity, bojler_wattage))
        print("------------------------------------------------------\n")        



        self.InfluxDBClient = InfluxDBClient(self.host, self.port, retries=5, timeout=1)
        self.DataFrameClient = DataFrameClient(host=self.host, database=self.db_name)

        self.EventChecker = EventChecker()
        self.TimeHandler = TimeHandler()
        self.Bojler = Bojler(bojler_capacity, bojler_wattage, bojler_set_tmp)
        
        self.data_db = self._actualize_data()
        self.last_data_update = datetime.now() 
        self.WeekPlanner = WeekPlanner(self.data_db)
        self.coef_up_in_current_heating_cycle_changed = False
        self.coef_down_in_current_heating_cycle_changed = False



    def _last_entry(self):
        #zde jsou casy v poradku
        """Loads last entru from DB - actual.

        Args:
            measurement (str, optional): [description]. Defaults to 'senzory_bojler'.

        Returns:
            [type]: [last entry values]
        """
        try:
            result = self.InfluxDBClient.query('SELECT * FROM "' + self.db_name + '"."autogen"."' + self.measurement + '" ORDER BY DESC LIMIT 1')
    
            result_list = list(result.get_points(measurement=self.measurement))[0]
            
            time_of_last_entry = self.TimeHandler.date_from_influxdb_to_datetime(result_list["time"])
            tmp1 = result_list["tmp1"]
            tmp2 = result_list["tmp2"]
            socket_turned_on = result_list["turned"]

            return {"tmp1":tmp1, "tmp2":tmp2, "socket_turned_on":socket_turned_on, "time_of_last_entry":time_of_last_entry}
         
        except:
            print("unable to read from influxDBclient")
            return None




    def _actualize_data(self):
        """

        Args:
            measurement (str, optional): [description]. Defaults to 'senzory_bojler'.
            host (str, optional): [description]. Defaults to 'influx_db'.

        Returns:
            [type]: Returns data from DB
        """
        print("trying to get new datasets...")
        

        try:
    
            datasets = self.DataFrameClient.query('SELECT * FROM "' + self.db_name + '"."autogen"."' + self.measurement + '" ORDER BY DESC')[self.measurement]
            
            print("got new datasets")
            df = pd.DataFrame(datasets)
            df = df[df.in_event != True]
            self.Bojler.set_measured_tmp(df)
    
            return df

        except:
            print("it wasnt posiible to get new datasets")
            return None

    def _next_heating_event(self, event):

        """Finds how long it takes to next heating.

        Returns:
            [type]: [description]
        """
        actual_time = self.TimeHandler.hour_minutes_now() 

        day_of_week = datetime.now().weekday()

        days_plus = 0

        while(days_plus < 7):
        
            day_plan = self.WeekPlanner.week_days_consumptions[day_of_week]

            for  key, item in   day_plan.items():
                next_time = item[event]

                if (next_time >= actual_time):
                    time_to_next_heating_event = (next_time - actual_time + timedelta(days = days_plus))  / timedelta(hours=1)

                    return{"will_occur_in" : time_to_next_heating_event, "duration": item['duration'], "peak": item["peak"], "time" : next_time}
                    #return [(next_time - actual_time + timedelta(days = days_plus)) / timedelta(hours=1), item['duration'], item['peak']]


            actual_time = self.TimeHandler.hour_minutes_now().replace(hour=0, minute=0)
            days_plus += 1
            day_of_week = (day_of_week + 1) % 7
        return None

                
    def _is_in_heating(self):

        hours_to_end = self._next_heating_event('end')["will_occur_in"]
        hours_to_start = self._next_heating_event('start')["will_occur_in"]

        return hours_to_start > hours_to_end
      

    def _check_data(self):
        if self.last_data_update - datetime.now()  > timedelta(days = 7):
            print(datetime.now() )
            print("actualizing data")

            actualized_data = self._actualize_data() 
            if actualized_data is not None:
                self.data_db = actualized_data
                self.WeekPlanner.week_plan(self.data_db)

    def control(self):


        self._check_data()

        last_entry = self._last_entry()

        
        #print("unable to find next heatup event")

        if self.EventChecker.check_off_event():
            print("naplanovana udalost")
            self._turn_socket_off()
            time.sleep(600)
            return
        next_heat_up_event = self.EventChecker.next_heat_up_event()

        if last_entry is None:
            self._turn_socket_on()
            return

        
        time_now = datetime.now()
        tmp_out = last_entry['tmp1']
        tmp_act = last_entry['tmp2']
        is_on = last_entry['socket_turned_on']
        time_of_last_entry = last_entry['time_of_last_entry']

        if (time_now - time_of_last_entry > timedelta(minutes = 10)):
            if not self.WeekPlanner.is_in_DTO():
                print("too old last entry ({}), need to heat".format(time_of_last_entry))
                if not is_on:
                        self._turn_socket_on()
            return
                
        
        if next_heat_up_event['hours_to_event'] is not None:
            time_to_without_DTO = self.WeekPlanner.duration_of_low_tarif_to_next_heating(next_heat_up_event['hours_to_event'])
            tmp_goal = next_heat_up_event['degree_target']
            print("time to next heating without DTO: ", time_to_without_DTO)
            print("tmp goal : ", tmp_goal)
            if( self.Bojler.is_needed_to_heat(tmp_act, tmp_goal=tmp_goal, time_to_consumption = time_to_without_DTO)):
                print("planned event to heat up with target {} Celsius occurs in {} hours".format(next_heat_up_event['degree_target'], next_heat_up_event['hours_to_event']))
                if not is_on:
                    self._turn_socket_on()
                return
                



        
        if(self._is_in_heating()):
            self.coef_down_in_current_heating_cycle_changed = False
        else:
            self.coef_up_in_current_heating_cycle_changed = False

        if (tmp_act < self.tmp_min):
            if not is_on:
                self._turn_socket_on()
            return
        ##############################################################################
        


        #upravit, zjednodusit
        if (self._is_in_heating()):

            #for current heating   
            current_heating = self._next_heating_event('end')
            current_heating_half_duration = current_heating['duration'] / 2
            how_long_to_current_heating_end = current_heating['will_occur_in'] 
            

            if(tmp_act < self.consumption_tmp_min):
            
                print("in heating, needed to increase tmp({}°C) above tmp min({}°C)".format(tmp_act, self.consumption_tmp_min))
                if not is_on:
                    self._turn_socket_on()

                if( (how_long_to_current_heating_end > current_heating_half_duration)  and not self.coef_up_in_current_heating_cycle_changed):
                    self.coef_up_in_current_heating_cycle_changed = True
                    self.heating_coef *= 1.015
                    print("changing coef to {}".format(self.heating_coef))
                    #zmena aby v cyklu nedochazelo stale ke zvysovani, nebot zmena se promitne az po nejake dobe

            #pokud neni treba doohrivat behem heatingu, vypinam
            else:
                if is_on:
                    print("turning off in heating, actual_tmp = {}".format(tmp_act))

                    self._turn_socket_off()

            return
        #not in heating
        else:
            #reseni ohrevu pro dalsi spotrebu
            next_heating =  self._next_heating_event('start')


            time_to_next_heating = self.WeekPlanner.duration_of_low_tarif_to_next_heating(next_heating['will_occur_in']) 


            next_heating_goal_temperature = next_heating['peak'] * self.heating_coef
            #print("{}   next heating at {} starts in: {}".format(datetime.now(), next_heating['time'], time_to_next_heating))

            #v tomto pripade je v momente neodberu potreba ohrivat + v pripadech, ze je teplota pod min viz vyse
            if( self.Bojler.is_needed_to_heat(tmp_act, tmp_goal=next_heating_goal_temperature, time_to_consumption = time_to_next_heating)):
                print("need to heat up before consumption, time to coms:{} , time without DTO: {}".format(next_heating['will_occur_in'], time_to_next_heating))
                if not is_on:
                    print("bojler is needed to heat up from {} to {}. turning socket on".format(tmp_act, next_heating_goal_temperature))
                    self._turn_socket_on()

                return

      

            if ( (tmp_act > (self.consumption_tmp_min + 3)) and not self.coef_down_in_current_heating_cycle_changed):
                self.coef_down_in_current_heating_cycle_changed = True
                #rozlisovat kolik casu zbyva do zacatku dalsiho ohrivani a podle toho uspat
                print("actual tmp is greater than consumption tmp min")
                self.heating_coef *= 0.985            
                print("changing coef to {}".format(self.heating_coef))

            #if boiler need to heat tmp act, tmp act + delta, time to next high tarif

            next_high_tarif_interval = self.WeekPlanner.next_high_tarif_interval('start')
            if next_high_tarif_interval is not None:
                tmp_delta = next_high_tarif_interval['tmp_delta']
                if tmp_delta > 0:
                    time_to_next_high_tarif_interval = next_high_tarif_interval['next_high_tarif_in']
                    if (self.Bojler.is_needed_to_heat(tmp_act, tmp_goal=tmp_act + tmp_delta, time_to_consumption = time_to_next_high_tarif_interval)):
                        if not is_on:
                            print("heating up before in high tarif consumption from {} to {}".format(tmp_act, tmp_act + tmp_delta))
                        return
            if is_on:
                    print("turning off outside of heating, actual_tmp = {}".format(tmp_act))
                    self._turn_socket_off()

        


    def _turn_socket_on(self):
        try:
            requests.get("http://" + self.socket_url +"/relay/0?turn=on")   
            print("socket turned on")
        except:
            print(datetime.now())
            print("it was unable to turn on socket")       

    def _turn_socket_off(self):
        try:
            requests.get("http://" + self.socket_url +"/relay/0?turn=off")
            print("socket turned off")

        except:
            print(datetime.now())
            print("it was unable to turn off socket")        


 
  
  


if __name__ == '__main__':

    import sys
    
    from optparse import OptionParser
    parser = OptionParser('%prog [OPTIONS] <host> <port>')
    
    parser.add_option(
        '-f', '--settings_file', dest='settings_file',
        type='string', 
        default=None
        )
    options, args = parser.parse_args()

    settings_file = options.settings_file


    #data = pd.read_pickle('data.pkl')




    #predat cele nastavei
    c = Controller(settings_file)

    while (1):
        #try:
        c.control()
        #except:
        #    print(datetime.now())
        #    print("unable to control in this cycle")
        
        #c.control()
        time.sleep(60)
    




