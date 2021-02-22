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

        print("------------------------------------------------------\n")
        print('initializing of Control...\n\tdb_name = {}\n\tsocker_url = {}\n\ttmp_output = {}\n\ttmp_bojler = {}\n\thost name = {}\n\tport = {}\n\tbojler capacity = {}\n\tbojler woltage = {}\n'.format(self.db_name, self.socket_url, self.tmp_output, self.tmp_bojler, self.host, self.port, bojler_capacity, bojler_wattage))
        print("------------------------------------------------------\n")        



        self.InfluxDBClient = InfluxDBClient(self.host, self.port, retries=5, timeout=1)
        self.DataFrameClient = DataFrameClient(host=self.host, database=self.db_name)

        self.EventChecker = EventChecker()
        self.TimeHandler = TimeHandler()
        self.Bojler = Bojler(bojler_capacity, bojler_wattage)
        
        self.data_db = self._actualize_data()
        self.last_data_update = datetime.now() 
        self.WeekPlanner = WeekPlanner(self.data_db)



    def _last_entry(self):
        """Loads last entru from DB - actual.

        Args:
            measurement (str, optional): [description]. Defaults to 'senzory_bojler'.

        Returns:
            [type]: [last entry values]
        """
        #dodelat kontrolu stari zaznamu. pri prilis starem ohriva
        try:
            result = self.InfluxDBClient.query('SELECT * FROM "' + self.db_name + '"."autogen"."' + self.measurement + '" ORDER BY DESC LIMIT 1')
    
            result_list = list(result.get_points(measurement=self.measurement))[0]

            
        
            tmp1 = result_list["tmp1"]
            tmp2 = result_list["tmp2"]
            socket_turned_on = result_list["turned"]
        
            #print(tmp1, tmp2, socket_turned_on)


            return (tmp1, tmp2, socket_turned_on)

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
            df = df[df.in_event != True].tmp1
    
            return df

        except:
            print("it wasnt posiible to get new datasets")
            return None

    def _next_heating_time(self, event):

        """Finds how long it takes to next heating.

        Returns:
            [type]: [description]
        """
        actual_time = self.TimeHandler.hour_minutes_now()


        days_plus = 0

        while(1):
        
            day_plan = self.WeekPlanner.week_days_consumptions[self.TimeHandler.this_day_string(days_plus)]

            for  key, item in   day_plan.items():
                next_time = item[event]

                if (next_time >= actual_time):

                    return [(next_time - actual_time + timedelta(days = days_plus)) / timedelta(hours=1), item['duration'], item['peak']]


            actual_time = self.TimeHandler.hour_minutes_now().replace(hour=0, minute=0)
            days_plus += 1
                
    def _is_in_heating(self):

        hours_to_end = self._next_heating_time('end')[0]
        hours_to_start = self._next_heating_time('start')[0]

        if(hours_to_start > hours_to_end):
            print("in heating period")
            return True
        False

    def _check_data(self):
        if self.last_data_update - datetime.now() > timedelta(days = 7):
            print(datetime.now())
            print("actualizing data")

            actualized_data = self._actualize_data() 
            if actualized_data is not None:
                self.data_db = actualized_data
                self.WeekPlanner.week_plan(self.data_db)

    def control(self):

        self._check_data()

        tmp_out, tmp_act, is_on = self._last_entry()


        #or is in consumption and temperature is below smthing

        #if(self.is_in_heating and tmp_act < self.consumption_tmp_min):
            #pokud dojdu sem, je potreba zvetsit teplotu ohrevu. pokud to jiz nejde, zvetsit min (pouze jen do nejake hodnoty)

        time_to_consumption = self._next_heating_time('start')[0]
        tmp_goal = self._heating_temperature()

        if self.EventChecker.check_event():
            if is_on:
                self._turn_socket_off()
            return
        else:
            print("no scheduled event")

        if (tmp_act < self.tmp_min):
            if not is_on:
                self._turn_socket_on()
            return
        if( self.Bojler.is_needed_to_heat(tmp_act, tmp_goal=tmp_goal, time_to_consumption = time_to_consumption)):
            if not is_on:
                self._turn_socket_on()
            return


        #rozlisovat cas do dalsiho topeni. pokud prave skoncilo, tolik nevadi  


        #for current heating   
        current_heating = self._next_heating_time('end')
        current_heating_half_duration = current_heating[1] / 2
        how_long_to_current_heating_end = current_heating[0]

        #upravit, zjednodusit
        if (self._is_in_heating() and tmp_act < self.consumption_tmp_min and  how_long_to_current_heating_end > current_heating_half_duration ):

            print("changing coef to")
            self.heating_coef *= 1.025
            print(self.heating_coef)

            if not is_on:
                self._turn_socket_on()
            return    

        if((not self._is_in_heating()) and (tmp_act > (self.consumption_tmp_min + 2) and (how_long_to_current_heating_end < current_heating_half_duration))):


            print("changing coef to")
            self.heating_coef *= 0.975            
            print(self.heating_coef)
       
        if is_on:
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


    def _heating_temperature(self):

        if(self._is_in_heating()):
            heating = self._next_heating_time('end')
        else:
            heating = self._next_heating_time('start')

        peak = heating[2]

        heat_temp = self.heating_coef * peak
        print("heattemp:")
        print(heat_temp)

        return heat_temp
  
  


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
        try:
            c.control()
        except:
            print(datetime.now())
            print("unable to control in this cycle")
        time.sleep(120)
    




