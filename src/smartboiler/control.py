###########################################################
# Bachelor's thesis                                       #
# From a dumb boiler to a smart one using a smart socket  #
# Author: Adam Grünwald                                   #
# BUT FIT BRNO, Faculty of Information Technology         #
# 26/6/2021                                               #
#                                                         #
# Module that controls a heating of smart boiler.         #
# Uses module Boiler for computing time needed to heating #
# of water, module TimeHandler for basic time and date    #
# operations, WeekPlanner for plan week heating,          # 
# SettingsLoader to load setting from settings file and   # 
# EventChecker which checks events in calendar,           # 
# when the water shouldn't be heated.                      #
###########################################################


from datetime import datetime, timedelta
import pandas as pd
import calendar
from influxdb import DataFrameClient
from influxdb import InfluxDBClient
import os.path
import signal
import sys
import argparse

import time

import json
import requests



from distutils.util import strtobool
from scipy.misc import electrocardiogram
import numpy as np


from smartboiler.boiler import Boiler
from smartboiler.time_handler import TimeHandler
from smartboiler.week_planner import WeekPlanner
from smartboiler.event_checker import EventChecker


class Controller:
    """Main class which makes decisions about about heating
    """

    def __init__(self, settings_file='settings.json'):
        """Inits class of Controller. Loads settings from a settings file

        Args:
            settings_file (str, optional): [name of json file with settings]. Defaults to 'settings.json'.
        """

        # from settings_loader import SettingsLoader
        # SettingsLoader = SettingsLoader(settings_file)
        # settings = SettingsLoader.load_settings()

        # self.host = settings['db_host']
        # self.port = settings['db_port']
        # self.db_name = settings['db_name']
        # self.measurement = settings['measurement']
        # self.db_user = settings['db_user']
        # self.db_pass = settings['db_pass']
        # self.socket_url = settings['socket_url']

        # self.tmp_output = settings['tmp_output']
        # self.tmp_boiler = settings['tmp_boiler']

        # self.tmp_min = settings['tmp_min']
        # self.consumption_tmp_min = settings['consumption_tmp_min']

        # self.start_date = datetime.now()
        
        # self.Hass = remote.API('localhost', 'smart_boiler01')
        self.shelly_entity_id = 'switch.shelly1pm_34945475a969'

        
        # boiler_wattage = settings['boiler_wattage']
        # boiler_capacity = settings['boiler_capacity']
        # boiler_set_tmp = settings['boiler_set_tmp']

        # print("------------------------------------------------------\n")
        # print('initializing of Control...\n\tdb_name = {}\n\tsocker_url = {}\n\ttmp_output = {}\n\ttmp_boiler = {}\n\thost name = {}\n\tport = {}\n\tboiler capacity = {}\n\tboiler woltage = {}\n'.format(
        #     self.db_name, self.socket_url, self.tmp_output, self.tmp_boiler, self.host, self.port, boiler_capacity, boiler_wattage))
        # print("------------------------------------------------------\n")

        # self.InfluxDBClient = InfluxDBClient(
        #     host=self.host, 
        #     port=self.port, 
        #     username=self.db_user, 
        #     password=self.db_pass, 
        #     retries=5, 
        #     timeout=1)
        # self.DataFrameClient = DataFrameClient(
        #     username=self.db_user, 
        #     password=self.db_pass,
        #     host=self.host, 
        #     database=self.db_name)

        # #self.EventChecker = EventChecker()
        # #self.TimeHandler = TimeHandler()
        # #self.Boiler = Boiler(capacity=boiler_capacity,
        # #                    wattage=boiler_wattage, set_tmp=boiler_set_tmp)

        # #self.data_db = self._actualize_data()
        # self.last_data_update = datetime.now()
        # self.last_legionella_heating = datetime.now()

        # #self.WeekPlanner = WeekPlanner(self.data_db)
        # self.coef_up_in_current_heating_cycle_changed = False
        # self.coef_down_in_current_heating_cycle_changed = False

    def _last_entry(self):
        """Loads last entru from DB - actual.

        Args:
            measurement (str, optional): [description]. Defaults to 'senzory_boiler'.

        Returns:
            [type]: [last entry values]
        """
        try:
            result = self.InfluxDBClient.query(
                'SELECT * FROM "' + self.db_name + '"."autogen"."' + self.measurement + '" ORDER BY DESC LIMIT 1')

            result_list = list(result.get_points(
                measurement=self.measurement))[0]

            time_of_last_entry = self.TimeHandler.date_from_influxdb_to_datetime(
                result_list["time"])
            tmp1 = result_list["tmp1"]
            tmp2 = result_list["tmp2"]
            socket_turned_on = result_list["turned"]

            return {"tmp1": tmp1, "tmp2": tmp2, "socket_turned_on": socket_turned_on, "time_of_last_entry": time_of_last_entry}

        except:
            print("unable to read from influxDBclient")
            return None

    def _actualize_data(self):
        """
        Actualizes data form DB

        Args:
            measurement (str, optional): [description]. Defaults to 'senzory_boiler'.
            host (str, optional): [description]. Defaults to 'influx_db'.

        Returns:
            [type]: Returns data from DB
        """
        print("trying to get new datasets...")

        try:

            datasets = self.DataFrameClient.query(
                # 'SELECT * FROM "' + self.db_name + '"."autogen"."' + self.measurement + '" ORDER BY DESC')[self.measurement]
                'SELECT * FROM "' + self.db_name + '"."autogen"."°C" ORDER BY DESC')['°C']

            print("got new datasets")
            df = pd.DataFrame(datasets)
            # df = df[df.in_event != True]
            self.Boiler.set_measured_tmp(df)

            return df

        except:
            print("it wasnt possible to get new datasets")
            return None


    def _check_data(self):
        """ Refreshs data every day
        """
        if self.last_data_update - datetime.now() > timedelta(days=1):
            print(datetime.now())
            print("actualizing data")

            actualized_data = self._actualize_data()
            if actualized_data is not None:
                self.data_db = actualized_data
                self.WeekPlanner.week_plan(self.data_db)

    def _learning(self):
        """ After one week of only measuring the data starts heating based on historical data.

        Returns:
            [boolean]: [True if in learing]
        """

        return ( ( datetime.now() - self.start_date) < timedelta(days=7) )

    def control(self):
        """ Method which decides about turning on or off the heating of a boiler.
        """

        self._check_data()

        last_entry = self._last_entry()

        # checks whether the water in boiler should be even ready 
        if self.EventChecker.check_off_event():
            print("naplanovana udalost")
            self.toggle_shelly_relay('off')
            time.sleep(600)
            return

        # last measured entry in DB
        if last_entry is None:
            self.toggle_shelly_relay('on')
            return


        # looks for the next heat up event from a calendar    
        next_calendar_heat_up_event = self.EventChecker.next_calendar_heat_up_event(
            self.Boiler)

        time_now = datetime.now()
        day_of_week = time_now.weekday()

        # actual tmp of water in boiler
        tmp_act = self.Boiler.real_tmp(last_entry['tmp2'])

        # state of smart socket
        is_on = last_entry['socket_turned_on']
        # in first week is water in boiler hold around 60 degrees

        #protection from freezing
        if tmp_act < 5:
            if not is_on:
                self.toggle_shelly_relay('on')

        if self._learning():
            if tmp_act > 60:
                if is_on:
                    self.toggle_shelly_relay('off')
            else:
                if tmp_act < 57:
                    if not is_on:
                        self.toggle_shelly_relay('on')

            return


        # if last entry is older than 10 minutes and not because of high tarif, water in a boiler is heated for sure
        time_of_last_entry = last_entry['time_of_last_entry']
        if (time_now - time_of_last_entry > timedelta(minutes=10)):
            if not self.WeekPlanner.is_in_DTO():
                print("too old last entry ({}), need to heat".format(
                    time_of_last_entry))
                if not is_on:
                    self.toggle_shelly_relay('on')
            return


        # helping variables for changing day coefs
        if(self.WeekPlanner.is_in_heating()):
            self.coef_down_in_current_heating_cycle_changed = False
        else:
            self.coef_up_in_current_heating_cycle_changed = False


        # once a three weeks is water in boiler heated on max for ellimination of Legionella
        if self.last_legionella_heating - datetime.now() > timedelta(days=21):
            self.coef_down_in_current_heating_cycle_changed = True
            if not is_on:
                print("starting heating for reduce legionella, this occurs every 3 weeks")
                self.toggle_shelly_relay('on')

            if tmp_act >= (65):
                time.sleep(1200)
                self.last_legionella_heating = datetime.now()
                print("legionella was eliminated, see you in 3 weeks")

        # if is scheduled event for heating, there is evaluated if it is needed to heat
        if next_calendar_heat_up_event['hours_to_event'] is not None:
            time_to_without_DTO = self.WeekPlanner.duration_of_low_tarif_to_next_heating(
                next_calendar_heat_up_event['hours_to_event'])
            tmp_goal = next_calendar_heat_up_event['degree_target']
            print("time to next heating without DTO: ", time_to_without_DTO)
            print("tmp goal : ", tmp_goal)
            if(self.Boiler.is_needed_to_heat(tmp_act, tmp_goal=tmp_goal, time_to_consumption=time_to_without_DTO)):
                print("planned event to heat up with target {} Celsius occurs in {} hours".format(
                    next_calendar_heat_up_event['degree_target'], next_calendar_heat_up_event['hours_to_event']))
                if not is_on:
                    self.toggle_shelly_relay('on')
                return

        # if is actual tmp lower than tmp min, there is need to heat
        if (tmp_act < self.tmp_min):
            if not is_on:
                self.toggle_shelly_relay('on')
            return


        if (self.WeekPlanner.is_in_heating()):

            current_heating = self.WeekPlanner.next_heating_event('end')
            if (current_heating is None):
                return
            current_heating_half_duration = current_heating['duration'] / 2
            how_long_to_current_heating_end = current_heating['will_occur_in']
            
            if(tmp_act < self.consumption_tmp_min):

                print("in heating, needed to increase tmp({}°C) above tmp min({}°C)".format(
                    tmp_act, self.consumption_tmp_min))
                if not is_on:
                    self.toggle_shelly_relay('on')

                if((how_long_to_current_heating_end > current_heating_half_duration) and not self.coef_up_in_current_heating_cycle_changed):
                    #changing the coefs if the temperature during predicted consumption is too low
                    self.coef_up_in_current_heating_cycle_changed = True
                    self.WeekPlanner.week_days_coefs[day_of_week] *= 1.015
                    print("changing day ({}) coef to {}".format(
                        (day_of_week + 1), self.WeekPlanner.week_days_coefs[day_of_week]))

            else:
                if is_on:
                    print("turning off in heating, actual_tmp = {}".format(tmp_act))

                    self.to

            return
        else:
            # checking whether it is needed to heat before the next predicted consumption
            next_heating = self.WeekPlanner. next_heating_event('start')
            if (next_heating is None):
                return
            time_to_next_heating = self.WeekPlanner.duration_of_low_tarif_to_next_heating(
                next_heating['will_occur_in'])

            next_heating_goal_temperature = next_heating['peak'] * \
                self.WeekPlanner.week_days_coefs[day_of_week]

            if(self.Boiler.is_needed_to_heat(tmp_act, tmp_goal=next_heating_goal_temperature, time_to_consumption=time_to_next_heating)):
                print("need to heat up before consumption, time to coms:{} , time without DTO: {}".format(
                    next_heating['will_occur_in'], time_to_next_heating))
                if not is_on:
                    print("boiler is needed to heat up from {} to {}. turning socket on".format(
                        tmp_act, next_heating_goal_temperature))
                    self.toggle_shelly_relay('on')

                return
            # te day coef is changed whether the temperature is too high outside of the predicted consumption
            if ((tmp_act > (self.consumption_tmp_min + 3)) and not self.coef_down_in_current_heating_cycle_changed):
                self.coef_down_in_current_heating_cycle_changed = True
                print("actual tmp is greater than consumption tmp min")
                self.WeekPlanner.week_days_coefs[day_of_week] *= 0.985
                print("changing day ({}) coef to {}".format(
                    (day_of_week + 1), self.WeekPlanner.week_days_coefs[day_of_week]))

            # if boiler need to heat tmp act, tmp act + delta, time to next high tarif
            next_high_tarif_interval = self.WeekPlanner.next_high_tarif_interval(
                'start')
            if next_high_tarif_interval is not None:
                tmp_delta = next_high_tarif_interval['tmp_delta']
                if tmp_delta > 0:
                    time_to_next_high_tarif_interval = next_high_tarif_interval['next_high_tarif_in']
                    if (self.Boiler.is_needed_to_heat(tmp_act, tmp_goal=tmp_act + tmp_delta, time_to_consumption=time_to_next_high_tarif_interval)):
                        if not is_on:
                            print("heating up before in high tarif consumption from {} to {}".format(
                                tmp_act, tmp_act + tmp_delta))
                        return
            if is_on:
                print("turning off outside of heating, actual_tmp = {}".format(tmp_act))
                self.to

    
    def toggle_shelly_relay(self, action, headers, base_url):
        service = 'switch.turn_' + action
        data = {'entity_id': self.shelly_entity_id}
        print("Setting shelly relay to {}".format(action))
        response = requests.post(
            f"{base_url}services/{service}", headers=headers, json=data
        )
        if response.status_code == 200:
            print(f"Shelly turned {action} successfully")
        else:
            print("Failed to toggle Shelly")

if __name__ == '__main__':

    import sys
    parser = argparse.ArgumentParser()
    parser.add_argument('--url', type=str, help='The URL to your Home Assistant instance, ex the external_url in your hass configuration')
    parser.add_argument('--key', type=str, help='Your access key. If using EMHASS in standalone this should be a Long-Lived Access Token')
    parser.add_argument('--addon', type=strtobool, default='False', help='Define if we are usinng EMHASS with the add-on or in standalone mode')
    args = parser.parse_args()
    
    base_url = args.url
    url = base_url + '/config'
    key = args.key
    web_ui = "0.0.0.0"
    
    headers = {
            "Authorization": "Bearer " + key,
            "content-type": "application/json"
        }
    # response = requests.get(url, headers=headers)
    # config_hass = response.json()
    # params_secrets = {
    #     'hass_url': base_url,
    #     'long_lived_token': key,
    #     'time_zone': config_hass['time_zone'],
    #     'lat': config_hass['latitude'],
    #     'lon': config_hass['longitude'],
    #     'alt': config_hass['elevation']
    #     }

    # parser.add_option(
    #     '-f', '--settings_file', dest='settings_file',
    #     type='string',
    #     default=None
    # )
    # options, args = parser.parse_args()

    # settings_file = options.settings_file
    setting_file = 'settings.json'
    c = Controller(setting_file)
    while (1):
        # c.control()
        c.toggle_shelly_relay('on', headers, base_url)
        
        time.sleep(60)
        c.toggle_shelly_relay('off', headers, base_url)
        
        time.sleep(60)

