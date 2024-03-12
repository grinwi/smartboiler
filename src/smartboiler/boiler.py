from calendar import week
from math import inf
from pathlib import Path

import influxdb

from smartboiler.data_handler import DataHandler
print('Running' if __name__ == '__main__' else 'Importing', Path(__file__).resolve())

##########################################################
# Bachelor's thesis                                      #
# From a dumb boiler to a smart one using a smart socket #
# Author: Adam Grünwald                                  #
# BUT FIT BRNO, Faculty of Information Technology        #
# 26/6/2021                                              #
#                                                        #
# Module with class representing boiler itself.          #
# Serves for computing time needed to heat to certain    #
# temperature or recalculates number of showers          #
# to needed temperature to heat to.                      #
##########################################################

from datetime import datetime, timedelta
import time

from numpy import ndarray
from smartboiler.switch import Switch
import pandas as pd


class Boiler(Switch):
    def __init__(self,  base_url, token, headers, boiler_switch_entity_id, dataHandler : DataHandler, capacity=100, wattage=2000, set_tmp=60, one_shower_volume=40, shower_temperature=40, min_tmp=37, heater_efficiency=0.98):

        print("------------------------------------------------------\n")
        print('initializing of control...\n\tCapacity of Boiler = {}\n\t Wattage of boiler = {}\n'.format(
            capacity, wattage))
        print("------------------------------------------------------\n")
        #    def __init__(self, entity_id, base_url, token, headers):

        Switch.__init__(self, entity_id=boiler_switch_entity_id, base_url=base_url, token=token, headers=headers)
        self.boiler_heat_cap = capacity * 1.163
        self.real_wattage = wattage * heater_efficiency
        self.high_tarif_schedule = dataHandler.get_high_tarif_schedule()
        self.set_tmp = set_tmp
        self.capacity = capacity
        self.one_shower_volume = one_shower_volume
        self.shower_temperature = shower_temperature
        self.min_tmp = min_tmp
        self.area_tmp = 13
        self.boiler_measured_max = 35.6

    def time_needed_to_heat_up_minutes(self, consumption_kWh):
        """
        Calculates the time needed for heating water in boiler by temperature delta.
            time = (m * c) * d'(tmp) / P * effectivity_coef

            https://vytapeni.tzb-info.cz/tabulky-a-vypocty/97-vypocet-doby-ohrevu-teple-vody
        """

        return (consumption_kWh) / (self.real_wattage)

    def is_needed_to_heat(self, tmp_act:int, prediction_of_consumption):
        """Conciders if it is needed to heat.

        Args:
            tmp_act ([type]): [actual temperature of water in boiler]
            tmp_goal ([type]): [temperature of water to heat before consumption]
            time_to_consumption ([type]): [time to consumption]

        Returns:
            [boolean]: [boolean value of needed to heat]
        """
        if(tmp_act < self.min_tmp):
            print(f'tmp_act ({tmp_act}) < self.min_tmp ({self.min_tmp})')
            return True
        
        # get actual kWh in boiler from volume and tmp
        boiler_kWh_above_set = (self.capacity * 1.163 * (self.set_tmp - tmp_act)) / 3600
        print(f'boiler_kWh_above_set: {boiler_kWh_above_set}')
        
        datetime_now = datetime.now()
        actual_time = datetime_now.time()
        actual_schedule = self.high_tarif_schedule[(self.high_tarif_schedule['time'] > actual_time) & (self.high_tarif_schedule['weekday'] >= datetime_now.weekday())]
        # get first 6*12 rows
        actual_schedule = actual_schedule.head(2*12)

        if (len(actual_schedule) < 2*12):
            #concat actual schedule with beggining of df_reset
            actual_schedule = pd.concat([actual_schedule, self.high_tarif_schedule.head(2*12 - len(actual_schedule))])
            
        len_of_df = len(prediction_of_consumption)
        print(f'len_of_df: {len_of_df}')
        for i in range(len_of_df, 0):
            sum_of_consumption = prediction_of_consumption.iloc[:i].sum() - boiler_kWh_above_set # todo add computation of water coldering = time * coef of coldering
            time_to_consumption_minutes = i*30
            
            unavailible_minutes = actual_schedule.iloc[:i]['unavailible_minutes'].sum()
            print(f'unavailible_minutes: {unavailible_minutes}')
            time_needed_to_heat = self.time_needed_to_heat_up_minutes(consumption_kWh=sum_of_consumption) + unavailible_minutes
            print(f'time_needed_to_heat: {time_needed_to_heat}')
            if(time_to_consumption_minutes < time_needed_to_heat):
                print(f'time_to_consumption_minutes ({time_to_consumption_minutes}) < time_needed_to_heat ({time_needed_to_heat})')
                return True
        
        print('no need to heat, returning false')
        return False
            

    def showers_degrees(self, number_of_showers):
        """Recalculates number of showers to temperature 
        on which is needed to heat up water in boiler.

        Args:
            number_of_showers ([int]): [number of showers to prepare]

        Returns:
            [float]: [tempreture to heat on]
        """
        showers_volume = number_of_showers * self.one_shower_volume

        cold_water_tmp = 10

        needed_temperature = ((self.min_tmp * self.capacity) + (self.shower_temperature *
                              showers_volume) - (showers_volume * cold_water_tmp)) / self.capacity

        if needed_temperature > self.set_tmp:
            return self.set_tmp

        return needed_temperature

    def real_tmp(self, tmp_act):
        """Calculates the real temperature of water in boiler

        Args:
            tmp_act ([float]): [actual temperature measured in boiler]

        Returns:
            [float]: [the real temperature of water in boiler]
        """
        print("measured_max: ", self.boiler_measured_max,)
        print("area_tmp: ", self.area_tmp)
        print("set_tmp: ", self.set_tmp)
        print("tmp_act: ", tmp_act)
        
        if((tmp_act is None) or (self.area_tmp is None) or (self.set_tmp is None)):
            return 50

        if(tmp_act < self.area_tmp or tmp_act > self.set_tmp):
            return tmp_act

        tmp_act_and_area_delta = tmp_act - self.area_tmp
        tmp_max_and_area_delta = self.boiler_measured_max - self.area_tmp

        p1 = tmp_act_and_area_delta / tmp_max_and_area_delta

        tmp = p1 * (self.set_tmp - self.area_tmp) + self.area_tmp
        
        return tmp

    def set_measured_tmp(self, df):
        """Initializes measured temperatures for calculation of real temperature of water in boiler.

        Args:
            df ([DataFrame]): [dataframe with measured data]
        """
        df_of_last_week = df[df.index > (
            df.last_valid_index() - timedelta(days=21))]

        self.area_tmp = df_of_last_week['tmp1'].nsmallest(100).mean()
        self.boiler_measured_max = df_of_last_week['tmp2'].nlargest(100).mean()

        print("area_tmp: ", self.area_tmp,
              "\nboiler_max: ", self.boiler_measured_max)


if __name__ == '__main__':
    tmp_act = 30
    tmp_goal = 52
    time_to_consumption = 1
    influxdb_host = 'localhost'
    influxdb_name = 'smart_home_zukalovi'
    influxdb_user = 'root'
    influxdb_pass = 'root'
    boiler_socket_id = 'shelly1pm_34945475a969'
    boiler_socket_power_id = 'esphome_web_c771e8_power'
    boiler_case_tmp_entity_id = 'esphome_web_c771e8_tmp3'
    boiler_water_temp_entity_id = 'esphome_web_c771e8_tmp2'
    start_of_data_measurement = datetime(2023, 11, 1)
    dataHandler = DataHandler(influx_id=influxdb_host, db_name=influxdb_name, db_username=influxdb_user, db_password=influxdb_pass, relay_entity_id=boiler_socket_id, relay_power_entity_id=boiler_socket_power_id, tmp_boiler_case_entity_id=boiler_case_tmp_entity_id, tmp_output_water_entity_id=boiler_water_temp_entity_id, start_of_data=start_of_data_measurement)
    boiler = Boiler('test', 'test2', 'dem', boiler_switch_entity_id='nkew', dataHandler=dataHandler)
