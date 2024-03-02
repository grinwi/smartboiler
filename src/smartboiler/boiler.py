from pathlib import Path
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
    def __init__(self,  base_url, token, headers, switch_entity_id, capacity=100, wattage=2000, set_tmp=60, one_shower_volume=40, shower_temperature=40, min_tmp=37, heater_efficiency=0.98):

        print("------------------------------------------------------\n")
        print('initializing of control...\n\tCapacity of Boiler = {}\n\t Wattage of boiler = {}\n'.format(
            capacity, wattage))
        print("------------------------------------------------------\n")
        super.__init__(self, ntity_id=switch_entity_id, url=base_url, token=token, headers=headers)
        self.boiler_heat_cap = capacity * 1.163
        self.real_wattage = wattage * heater_efficiency
        
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

    def is_needed_to_heat(self, tmp_act:int, prediction_of_consumption:ndarray):
        """Conciders if it is needed to heat.

        Args:
            tmp_act ([type]): [actual temperature of water in boiler]
            tmp_goal ([type]): [temperature of water to heat before consumption]
            time_to_consumption ([type]): [time to consumption]

        Returns:
            [boolean]: [boolean value of needed to heat]
        """
        if(tmp_act < self.min_tmp):
            return True
        
        # get whole dataframe and check if it is needed to heat
        # if it is, return True
        # if not, pop last row and continue with reduced df
        # if df is empty, return False
        len_of_df = len(prediction_of_consumption)
        for i in range(len_of_df, 0):
            sum_of_consumption = prediction_of_consumption.iloc[:i].sum()
            time_to_consumption_minutes = i*30
            
            time_needed_to_heat = self.time_needed_to_heat_up_minutes(consumption_kWh=sum_of_consumption)
            
            if(time_to_consumption_minutes < time_needed_to_heat):
                return True
            
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
    b = Boiler(capacity=80, wattage=2000)
    print(b.is_needed_to_heat(tmp_act, tmp_goal, time_to_consumption))
