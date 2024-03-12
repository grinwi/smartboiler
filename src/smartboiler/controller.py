from pathlib import Path
from pyexpat import model
import re, pytz

print('Running' if __name__ == '__main__' else 'Importing', Path(__file__).resolve())

###########################################################
# Masters's thesis                                       #
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
import logging
import time
import os

import json
import requests



from distutils.util import strtobool
from scipy.misc import electrocardiogram
import numpy as np


from smartboiler.data_handler import DataHandler
from smartboiler.forecast import Forecast
from smartboiler.boiler import Boiler

class Controller:
    """Main class which makes decisions about about heating
    """

    def __init__(self, dataHandler : DataHandler, boiler : Boiler, forecast : Forecast, load_model = False):
        """Inits class of Controller. Loads settings from a settings file

        Args:
            settings_file (str, optional): [name of json file with settings]. Defaults to 'settings.json'.
        """
        # TODO - load settings from config file or home assistant


        
        # self.socket_url = settings['socket_url']

        # self.tmp_output = settings['tmp_output']        
        #self.tmp_boiler = settings['tmp_boiler']
        #self.tmp_boiler_room = settings['tmp_boiler_room']

        # self.how_water_flow = settings['how_water_flow']
        # self.tmp_water_flow = settings['tmp_water_flow']
        
        
        self.tmp_min = 5
        self.consumption_tmp_min = 40

        self.start_date = datetime.now()
        
        # self.Hass = remote.API('localhost', 'smart_boiler01')
        self.dataHandler = dataHandler
        self.boiler = boiler
        self.forecast = forecast
        if (load_model):
            print("loading model")
            self.forecast.load_model() 
        else:
            print("training model")
            forecast.train_model()
            self.forecast.build_model()
            self.forecast.fit_model()    
            
        self.last_model_training = datetime.now()   


        # print("------------------------------------------------------\n")
        # print('initializing of Control...\n\tdb_name = {}\n\tsocker_url = {}\n\ttmp_output = {}\n\ttmp_boiler = {}\n\thost name = {}\n\tport = {}\n\tboiler capacity = {}\n\tboiler woltage = {}\n'.format(
        #     self.db_name, self.socket_url, self.tmp_output, self.tmp_boiler, self.host, self.port, boiler_capacity, boiler_wattage))
        # print("------------------------------------------------------\n")



        # #self.EventChecker = EventChecker()
        # #self.TimeHandler = TimeHandler()
        # #self.Boiler = Boiler(capacity=boiler_capacity,
        # #                    wattage=boiler_wattage, set_tmp=boiler_set_tmp)

        # #self.data_db = self._actualize_data()
        self.last_legionella_heating = datetime.now()

        # #self.WeekPlanner = WeekPlanner(self.data_db)
        self.coef_up_in_current_heating_cycle_changed = False
        self.coef_down_in_current_heating_cycle_changed = False

    def _last_entry(self):
        print("getting last entry")
        last_entry = self.dataHandler.get_actual_boiler_stats()
        print ("last entry: {}".format(last_entry))
        return last_entry


    def _check_data(self):
        """ Retrain model if the last data update is older than 7 days.
        """
        if self.last_model_training - datetime.now() > timedelta(days=7):
            print(datetime.now())
            print("actualizing data")

            self.forecast.train_model()
            self.forecast.fit_model()
            self.last_model_training = datetime.now()

    def _learning(self):
        """ After one week of only measuring the data starts heating based on historical data.

        Returns:
            [boolean]: [True if in learing]
        """

        return ( ( datetime.now() - self.start_date) < timedelta(days=7) )

    def control(self):
        """ Method which decides about turning on or off the heating of a boiler.
        """
        
        time_now = datetime.now().astimezone(pytz.timezone('Europe/Prague'))
        print("controling boiler")
        print(time_now)
        time_now_plus_12_hours = time_now + timedelta(hours=12)
        # self._check_data()

        last_entry = self._last_entry()

        # # checks whether the water in boiler should be even ready 
        # if self.eventChecker.check_off_event():
        #     print("naplanovana udalost")
        #     self.boiler.turn_off()
        #     time.sleep(600)
        #     return


        tmp_measured = last_entry['boiler_case_tmp']
        is_on = last_entry['is_boiler_on']
        
        print("tmp_measured: {}".format(tmp_measured))
        print("is_on: {}".format(is_on))
        
        # # in case of too old data, the boiler is turned on
        # if ( ( time_now.microsecond - (last_entry['boiler_case_last_time_entry']).microsecond)/1000000 > timedelta(minutes=10)):
        #     print("too old data, turning on")
        #     boiler.turn_on()
        #     return
        
        # TODO - heatup events from calendar
        # # looks for the next heat up event from a calendar    
        # next_calendar_heat_up_event = self.eventChecker.next_calendar_heat_up_event(
        #     self.Boiler)

   

        print("measured tmp: {}".format(tmp_measured))
        # actual tmp of water in boiler
        tmp_act = self.boiler.real_tmp(tmp_measured)
        print("actual tmp: {}".format(tmp_act))
        
        if is_on is None:
            print("boiler state is unknown")
            is_on = False
        # state of smart socket
        print("is_on: {}".format(bool(is_on)))
        
        #protection from freezing
        if tmp_act < 5:
            if not is_on:
                self.boiler.turn_on()

        # if self._learning():
        #     if tmp_act > 60:
        #         if is_on:
        #             self.boiler.turn_off()
        #     else:
        #         if tmp_act < 57:
        #             if not is_on:
        #                 self.boiler.turn_on()

        #     return


        # # if last entry is older than 10 minutes and not because of high tarif, water in a boiler is heated for sure
        # time_of_last_entry = last_entry['time_of_last_entry']
        # if (time_now - time_of_last_entry > timedelta(minutes=10)):
        #     if not self.weekPlanner.is_in_DTO():
        #         print("too old last entry ({}), need to heat".format(
        #             time_of_last_entry))
        #         if not is_on:
        #             self.boiler.turn_on()
        #     return

        print("getting forecast")
        consumption_forecast = self.forecast.get_forecast_next_steps() # step has 30 minutes, so 24 steps is 12 hours
        print("forecast: {}".format(consumption_forecast))
        # if the boiler is needed to heat up before the next predicted consumption
        print("checking if boiler is needed to heat")
        need_to_heat = self.boiler.is_needed_to_heat(tmp_act, consumption_forecast)
        print("need to heat: {}".format(need_to_heat))
        if need_to_heat:
            print("need to heat, turning on")
            if not is_on:
                self.boiler.turn_on()
        else:
            print("no need to heat, turning off")
            if is_on:
                self.boiler.turn_off()
            
        
        
        # # helping variables for changing day coefs
        # if(self.weekPlanner.is_in_heating()):
        #     self.coef_down_in_current_heating_cycle_changed = False
        # else:
        #     self.coef_up_in_current_heating_cycle_changed = False


        # once a three weeks is water in boiler heated on max for ellimination of Legionella
        if self.last_legionella_heating - datetime.now() > timedelta(days=21):
            self.coef_down_in_current_heating_cycle_changed = True
            if not is_on:
                print("starting heating for reduce legionella, this occurs every 3 weeks")
                self.boiler.turn_on()

            if tmp_act >= (65):
                time.sleep(1200)
                self.last_legionella_heating = datetime.now()
                print("legionella was eliminated, see you in 3 weeks")






        # if (self.WeekPlanner.is_in_heating()):

        #     current_heating = self.WeekPlanner.next_heating_event('end')
        #     if (current_heating is None):
        #         return
        #     current_heating_half_duration = current_heating['duration'] / 2
        #     how_long_to_current_heating_end = current_heating['will_occur_in']
            
        #     if(tmp_act < self.consumption_tmp_min):

        #         print("in heating, needed to increase tmp({}°C) above tmp min({}°C)".format(
        #             tmp_act, self.consumption_tmp_min))
        #         if not is_on:
        #             self.boiler.turn_on()

        #         if((how_long_to_current_heating_end > current_heating_half_duration) and not self.coef_up_in_current_heating_cycle_changed):
        #             #changing the coefs if the temperature during predicted consumption is too low
        #             self.coef_up_in_current_heating_cycle_changed = True
        #             self.WeekPlanner.week_days_coefs[day_of_week] *= 1.015
        #             print("changing day ({}) coef to {}".format(
        #                 (day_of_week + 1), self.WeekPlanner.week_days_coefs[day_of_week]))

        #     else:
        #         if is_on:
        #             print("turning off in heating, actual_tmp = {}".format(tmp_act))

        #             self.to

        #     return
        # else:
        #     # checking whether it is needed to heat before the next predicted consumption
        #     next_heating = self.WeekPlanner. next_heating_event('start')
        #     if (next_heating is None):
        #         return
        #     time_to_next_heating = self.WeekPlanner.duration_of_low_tarif_to_next_heating(
        #         next_heating['will_occur_in'])

        #     next_heating_goal_temperature = next_heating['peak'] * \
        #         self.WeekPlanner.week_days_coefs[day_of_week]

        #     if(self.Boiler.is_needed_to_heat(tmp_act, tmp_goal=next_heating_goal_temperature, time_to_consumption=time_to_next_heating)):
        #         print("need to heat up before consumption, time to coms:{} , time without DTO: {}".format(
        #             next_heating['will_occur_in'], time_to_next_heating))
        #         if not is_on:
        #             print("boiler is needed to heat up from {} to {}. turning socket on".format(
        #                 tmp_act, next_heating_goal_temperature))
        #             self.boiler.turn_on()

        #         return
        #     # te day coef is changed whether the temperature is too high outside of the predicted consumption
        #     if ((tmp_act > (self.consumption_tmp_min + 3)) and not self.coef_down_in_current_heating_cycle_changed):
        #         self.coef_down_in_current_heating_cycle_changed = True
        #         print("actual tmp is greater than consumption tmp min")
        #         self.WeekPlanner.week_days_coefs[day_of_week] *= 0.985
        #         print("changing day ({}) coef to {}".format(
        #             (day_of_week + 1), self.WeekPlanner.week_days_coefs[day_of_week]))

        #     # if boiler need to heat tmp act, tmp act + delta, time to next high tarif
        #     next_high_tarif_interval = self.WeekPlanner.next_high_tarif_interval(
        #         'start')
        #     if next_high_tarif_interval is not None:
        #         tmp_delta = next_high_tarif_interval['tmp_delta']
        #         if tmp_delta > 0:
        #             time_to_next_high_tarif_interval = next_high_tarif_interval['next_high_tarif_in']
        #             if (self.Boiler.is_needed_to_heat(tmp_act, tmp_goal=tmp_act + tmp_delta, time_to_consumption=time_to_next_high_tarif_interval)):
        #                 if not is_on:
        #                     print("heating up before in high tarif consumption from {} to {}".format(
        #                         tmp_act, tmp_act + tmp_delta))
        #                 return
        #     if is_on:
        #         print("turning off outside of heating, actual_tmp = {}".format(tmp_act))
        #         self.to

    


if __name__ == '__main__':
    logging.info("Starting SmartBoiler Controller")

    parser = argparse.ArgumentParser()
    parser.add_argument('--url', type=str, help='The URL to your Home Assistant instance, ex the external_url in your hass configuration')
    parser.add_argument('--key', type=str, help='Your access key. If using EMHASS in standalone this should be a Long-Lived Access Token')
    parser.add_argument('--addon', type=strtobool, default='False', help='Define if we are usinng EMHASS with the add-on or in standalone mode')
    args = parser.parse_args()

    OPTIONS_PATH = os.getenv('OPTIONS_PATH', default="/app/options.json")
    options_json = Path(OPTIONS_PATH)

    # Read options info
    if options_json.exists():
        with options_json.open('r') as data:
            options = json.load(data)

    DATA_PATH = os.getenv("DATA_PATH", default="/app/data/")
    MODEL_PATH = os.getenv("MODEL_PATH", default="/app/lstm_model_zukalovi.h5")
    
    data_path = Path(DATA_PATH)
    model_path = Path(MODEL_PATH)
    
    start_of_data_measurement = datetime(2023,10,1,0,0,0,0)

    hass_url = options['hass_url']
    long_lived_token = options['long_lived_token']
    influxdb_host = options['influxdb_host']
    influxdb_port: 8086
    influxdb_user = options['influxdb_user']
    influxdb_pass = options['influxdb_pass']
    influxdb_name = options['influxdb_name']
    boiler_socket_switch_id = options['boiler_entidy_id']
    boiler_socket_id = options['boiler_socket_id']
    boiler_socket_power_id = options['boiler_socket_power_id']
    boiler_case_tmp_entity_id = options['boiler_case_tmp_entity_id']
    boiler_case_tmp_measurement = options['boiler_case_tmp_measurement']
    boiler_water_flow_entity_id = options['boiler_water_flow_entity_id']
    boiler_water_flow_measurement = options['boiler_water_flow_measurement']
    boiler_water_temp_entity_id = options['boiler_water_temp_entity_id']
    boiler_water_temp_measurement = options['boiler_water_temp_measurement']
    boiler_volume = options['boiler_volume']
    boiler_set_tmp = options['boiler_set_tmp']
    boiler_min_operation_tmp = options['boiler_min_operation_tmp']
    one_shower_volume = options['one_shower_volume']
    boiler_watt_power = options['boiler_watt_power']
    household_floor_size = options['household_floor_size']
    household_members = options['household_members']
    thermostat_entity_id = options['thermostat_entity_id']
    logging_level = options['logging_level']
    load_model = options['load_model']
    
    print(type(load_model))
    print(load_model)
    
    for value, key in options.items():
        print(f'{value}: {key}')
    

    print(f'Starting SmartBoiler Controller with the following settings: {options}')
    base_url = hass_url
    url = base_url + '/config'
    web_ui = "0.0.0.0"

    headers = {
            "Authorization": f'Bearer {long_lived_token}',
            "content-type": "application/json"
        }


    dataHandler = DataHandler(influx_id=influxdb_host, db_name=influxdb_name, db_username=influxdb_user, db_password=influxdb_pass, relay_entity_id=boiler_socket_id, relay_power_entity_id=boiler_socket_power_id, tmp_boiler_case_entity_id=boiler_case_tmp_entity_id, tmp_output_water_entity_id=boiler_water_temp_entity_id, start_of_data=start_of_data_measurement)
    boiler_switch_entity_id = 'switch.' + boiler_socket_id
    print("inicializing boiler from controller __main__")
    boiler = Boiler(base_url, long_lived_token, headers, boiler_switch_entity_id=boiler_socket_switch_id, dataHandler=dataHandler, capacity=boiler_volume, wattage=boiler_watt_power, set_tmp=boiler_set_tmp)
    print("inicializing forecast from controller __main__   ")
    forecast = Forecast(dataHandler=dataHandler, start_of_data=start_of_data_measurement,model_path=model_path)
    print("inicializing controller from controller __main__")
    controller = Controller(dataHandler=dataHandler, boiler=boiler, forecast=forecast, load_model=load_model)

    while (1):
        
        controller.control()
        # c.toggle_shelly_relay('on', headers, base_url)
        
        # time.sleep(60)
        # c.toggle_shelly_relay('off', headers, base_url)
        
        time.sleep(60)

