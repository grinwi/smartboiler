from calendar import week
from math import inf
from pathlib import Path

import influxdb

from smartboiler.data_handler import DataHandler

print("Running" if __name__ == "__main__" else "Importing", Path(__file__).resolve())

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

from datetime import date, datetime, timedelta
import time

from numpy import ndarray
from smartboiler.switch import Switch
import pandas as pd


class Boiler(Switch):
    def __init__(
        self,
        base_url,
        token,
        headers,
        boiler_switch_entity_id,
        dataHandler: DataHandler,
        capacity=100,
        wattage=2000,
        set_tmp=60,
        one_shower_volume=40,
        shower_temperature=40,
        min_tmp=40,
        heater_efficiency=0.88,
        average_boiler_surroundings_temp=15,
        boiler_case_max_tmp=40,
        hdo=False,
        cooldown_coef_B = 1.12
    ):

        print("------------------------------------------------------\n")
        print(
            "initializing of control...\n\tCapacity of Boiler = {}\n\t Wattage of boiler = {}\n".format(
                capacity, wattage
            )
        )
        print("------------------------------------------------------\n")
        #    def __init__(self, entity_id, base_url, token, headers):

        Switch.__init__(
            self,
            entity_id=boiler_switch_entity_id,
            base_url=base_url,
            token=token,
            headers=headers,
        )
        self.dataHandler = dataHandler
        self.boiler_heat_cap = capacity * 1.163
        self.real_wattage = wattage * heater_efficiency
        self.hdo = hdo
        self.high_tarif_schedule = dataHandler.get_high_tarif_schedule()
        self.set_tmp = set_tmp
        self.capacity = capacity
        self.one_shower_volume = one_shower_volume
        self.shower_temperature = shower_temperature
        self.min_tmp = min_tmp
        self.area_tmp = average_boiler_surroundings_temp
        self.boiler_case_max_tmp = boiler_case_max_tmp
        self.cooldown_coef_B = cooldown_coef_B

        # if (average_boiler_surroundings_temp is None or boiler_case_max_tmp is None):
        #     boiler_data_stats = dataHandler.get_boiler_data_stats(left_time_interval=datetime.now() - timedelta(days=21))
        #     self.set_measured_tmp(boiler_data_stats)
    def get_kWh_loss_in_time(self, time_minutes, tmp_act=60):
        """Calculates the kWh loss in time.

        Args:
            time_minutes ([int]): [time in minutes]

        Returns:
            [float]: [kWh loss in time]
        """
        tmp_delta = tmp_act - self.area_tmp
        kWh_loss = (time_minutes * self.cooldown_coef_B * tmp_delta / 1000) / 60
        # print(f"kWh_loss: {kWh_loss}, time_minutes: {time_minutes}, tmp_act: {tmp_act}, tmp_delta: {tmp_delta}")
        return kWh_loss
        
        
    def time_needed_to_heat_up_minutes(self, consumption_kWh):
        """
        Calculates the time needed for heating water in boiler by temperature delta.
            time = (m * c) * d'(tmp) / P * effectivity_coef

            https://vytapeni.tzb-info.cz/tabulky-a-vypocty/97-vypocet-doby-ohrevu-teple-vody
        """

        return (( consumption_kWh / (self.real_wattage / 1000)) * 60)
    
    def get_kWh_delta_from_temperatures(self, tmp_act: int, tmp_goal: int):
        return ( 4.186 * self.capacity * (tmp_goal - tmp_act) / 3600 )

    def is_needed_to_heat(self, tmp_act: int, prediction_of_consumption)->tuple[bool, int]:
        """Conciders if it is needed to heat.

        Args:
            tmp_act ([type]): [actual temperature of water in boiler]
            tmp_goal ([type]): [temperature of water to heat before consumption]
            time_to_consumption ([type]): [time to consumption]

        Returns:
            [boolean]: [boolean value of needed to heat]
        """

        if tmp_act < self.min_tmp:
            kWh_needed_to_heat = self.get_kWh_delta_from_temperatures(tmp_act, (self.min_tmp+3)) # add 3 degrees above min
            
            return (True, self.time_needed_to_heat_up_minutes(consumption_kWh=kWh_needed_to_heat))

        # get actual kWh in boiler from volume and tmp
        boiler_kWh_above_set = self.get_kWh_delta_from_temperatures(self.min_tmp, tmp_act)#(self.capacity * 4.186 * (self.min_tmp - tmp_act)) / 3600

        datetime_now = datetime.now()
        actual_time = datetime_now.time()
        actual_schedule = self.high_tarif_schedule[
            (self.high_tarif_schedule["time"] > actual_time)
            & (self.high_tarif_schedule["weekday"] >= datetime_now.weekday())
        ]
        # get first 6*12 rows
        actual_schedule = actual_schedule.head(2 * 12)

        if len(actual_schedule) < 2 * 12:
            # concat actual schedule with beggining of df_reset
            actual_schedule = pd.concat(
                [
                    actual_schedule,
                    self.high_tarif_schedule.head(2 * 12 - len(actual_schedule)),
                ]
            )
        if not self.hdo:
            actual_schedule["unavailable_minutes"] = 0

        len_of_df = len(prediction_of_consumption)

        for i in range(len_of_df, 0, -1):
            time_to_consumption_minutes = (i * 60) - 30

            sum_of_consumption = (
                sum(prediction_of_consumption[:i])
                - boiler_kWh_above_set
            ) + self.get_kWh_loss_in_time(time_minutes=time_to_consumption_minutes, tmp_act=tmp_act) 

            unavailible_minutes = actual_schedule.iloc[:i]["unavailable_minutes"].sum()
            time_needed_to_heat = (
                self.time_needed_to_heat_up_minutes(consumption_kWh=sum_of_consumption)
            )
            if time_needed_to_heat > 0:
                time_needed_to_heat += unavailible_minutes

            if time_to_consumption_minutes < time_needed_to_heat:
                print(
                    f"time_needed_to_heat: {time_needed_to_heat}, time_to_consumption_minutes: {time_to_consumption_minutes}, sum_of_consumption: {sum_of_consumption}, unavailible_minutes: {unavailible_minutes}"
                )
                return (True, time_needed_to_heat)  

        print(f'no need to heat, returning false, time_needed_to_heat: {time_needed_to_heat}, above_set: {boiler_kWh_above_set}')
        return (False, 0)

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

        needed_temperature = (
            (self.min_tmp * self.capacity)
            + (self.shower_temperature * showers_volume)
            - (showers_volume * cold_water_tmp)
        ) / self.capacity

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

        if (tmp_act is None) or (self.area_tmp is None) or (self.set_tmp is None):
            return 50

        if tmp_act < self.area_tmp or tmp_act > self.set_tmp:
            return tmp_act

        tmp_act_and_area_delta = tmp_act - self.area_tmp
        tmp_max_and_area_delta = self.boiler_case_max_tmp - self.area_tmp

        p1 = tmp_act_and_area_delta / tmp_max_and_area_delta

        tmp = p1 * (self.set_tmp - self.area_tmp) + self.area_tmp

        return tmp

    def set_measured_tmp(self, df):
        """Initializes measured temperatures for calculation of real temperature of water in boiler.

        Args:
            df ([DataFrame]): [dataframe with measured data]
        """
        df_of_last_week = df[df.index > (df.last_valid_index() - timedelta(days=21))]

        self.area_tmp = df_of_last_week["tmp1"].nsmallest(100).mean()
        self.boiler_case_max_tmp = df_of_last_week["tmp2"].nlargest(100).mean()


if __name__ == "__main__":
    tmp_act = 30
    tmp_goal = 52
    time_to_consumption = 1
    influxdb_host = "localhost"
    influxdb_name = "smart_home_zukalovi"
    influxdb_user = "root"
    influxdb_pass = "root"
    boiler_socket_id = "shelly1pm_34945475a969"
    boiler_socket_power_id = "esphome_web_c771e8_power"
    boiler_case_tmp_entity_id = "esphome_web_c771e8_tmp3"
    boiler_water_temp_entity_id = "esphome_web_c771e8_tmp2"
    start_of_data_measurement = datetime(2023, 11, 1)
    dataHandler = DataHandler(
        influx_id=influxdb_host,
        db_name=influxdb_name,
        db_username=influxdb_user,
        db_password=influxdb_pass,
        relay_entity_id=boiler_socket_id,
        relay_power_entity_id=boiler_socket_power_id,
        tmp_boiler_case_entity_id=boiler_case_tmp_entity_id,
        tmp_output_water_entity_id=boiler_water_temp_entity_id,
        start_of_data=start_of_data_measurement,
    )
    boiler = Boiler(
        "test", "test2", "dem", boiler_switch_entity_id="nkew", dataHandler=dataHandler
    )
