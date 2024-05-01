# Created as a part of Master's Thesis "Using machine learning methods to save energy in a smart home"
# Faculty of Information Technology, Brno University of Technology, 2024
# Author: Adam Grünwald
#
# This module is used for calculations of the heating time and controlling the boiler itself

from datetime import datetime, timedelta
from smartboiler.switch import Switch
import pandas as pd
from typing import Optional


from smartboiler.event_checker import EventChecker
from smartboiler.data_handler import DataHandler
from smartboiler.fotovoltaics import Fotovoltaics


class Boiler(Switch):
    """Class representing boiler itself. Used for the calculation of time needed to heat to certain temperature or recalculates number of showers to needed temperature to heat to.
        Also used for controlling the boiler itself via the relay.
    Args:
        Switch (_type_): Parent is a Switch class for the relay controlling the heating of the boiler.
    """

    def __init__(
        self,
        base_url: str,
        token: str,
        headers: dict,
        boiler_switch_entity_id: str,
        dataHandler: DataHandler,
        eventChecker: EventChecker,
        fotovoltaics: Optional[Fotovoltaics] = None,
        capacity: Optional[int] = 100,
        wattage: Optional[int] = 2000,
        set_tmp: Optional[int] = 60,
        one_shower_volume: Optional[int] = 40,
        shower_temperature: Optional[int] = 40,
        min_tmp: Optional[int] = 40,
        heater_efficiency: Optional[float] = 0.88,
        average_boiler_surroundings_temp: Optional[int] = 15,
        boiler_case_max_tmp: Optional[int] = 40,
        hdo: Optional[bool] = False,
        cooldown_coef_B: Optional[float] = 1.12,
    ):
        """Init method for the Boiler class.

        Args:
            base_url (str): Url of the Shelly device.
            token (str): Token for the Shelly device.
            headers (dict): Headers for the Shelly device.
            boiler_switch_entity_id (str): Entity ID of the boiler switch.
            dataHandler (DataHandler): Instance of the DataHandler class.
            eventChecker (EventChecker): Instance of the EventChecker class.
            fotovoltaics (Optional[Fotovoltaics], optional): Instance of the Fotovoltaics slass. Defaults to None.
            capacity (Optional[int], optional): Capacity of the boiler represented in liters. Defaults to 100.
            wattage (Optional[int], optional): Wattage of the boiler. Defaults to 2000.
            set_tmp (Optional[int], optional): Set temperature on the boiler. Defaults to 60.
            one_shower_volume (Optional[int], optional): Volume used for one shower in liters. Defaults to 40.
            shower_temperature (Optional[int], optional): Ideal temperature of one shower. Defaults to 40.
            min_tmp (Optional[int], optional): Minimal temperature of the water in the boiler. Defaults to 40.
            heater_efficiency (Optional[float], optional): Efficiency of the heater itself. Defaults to 0.88.
            average_boiler_surroundings_temp (Optional[int], optional): Temperature of the surroundings of the boiler. Defaults to 15.
            boiler_case_max_tmp (Optional[int], optional): Maximum temperature measured by sensor in the boiler case. Defaults to 40.
            hdo (Optional[bool], optional): Is the boiler connected to HDO. Defaults to False.
            cooldown_coef_B (Optional[float], optional): Coeficient of the cooldown of the water in boiler. Defaults to 1.12.
        """

        # Call the parent class constructor
        Switch.__init__(
            self,
            entity_id=boiler_switch_entity_id,
            base_url=base_url,
            token=token,
            headers=headers,
        )

        self.dataHandler = dataHandler
        self.fotovoltaics = fotovoltaics
        self.eventChecker = eventChecker
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

    def get_kWh_loss_in_time(
        self, time_minutes: float, tmp_act: Optional[float] = 60
    ) -> float:
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

    def time_needed_to_heat_up_minutes(self, consumption_kWh: float) -> float:
        """
        Calculates the time needed for heating water in boiler by temperature delta.
            time = (m * c) * d'(tmp) / P * effectivity_coef

            https://vytapeni.tzb-info.cz/tabulky-a-vypocty/97-vypocet-doby-ohrevu-teple-vody
        """

        return (consumption_kWh / (self.real_wattage / 1000)) * 60

    def get_kWh_delta_from_temperatures(self, tmp_act: int, tmp_goal: int) -> float:
        """Calculates delta of kWh from temperatures of the water in boiler

        Args:
            tmp_act (int): Actual temperature of tha water in boiler.
            tmp_goal (int): Temperature of the boiler to heat to.

        Returns:
            float: kWh delta from temperatures
        """
        return 4.186 * self.capacity * (tmp_goal - tmp_act) / 3600

    def is_needed_to_heat(
        self, tmp_act: int, prediction_of_consumption: pd.DataFrame
    ) -> tuple[bool, int]:
        """Method to check if it is needed to heat the water in boiler.

        Args:
            tmp_act (int): Actual temperature of the water in boiler.
            prediction_of_consumption (pd.DataFrame): Dataframe with prediction of consumption.

        Returns:
            tuple[bool, int]: Tuple with boolean if it is needed to heat and time needed to heat in minutes.
        """

        # if the actual temperature is lower than minimal temperature, heat
        if tmp_act < self.min_tmp:
            kWh_needed_to_heat = self.get_kWh_delta_from_temperatures(
                tmp_act, (self.min_tmp + 3)
            )  # add 3 degrees above min

            return (
                True,
                self.time_needed_to_heat_up_minutes(consumption_kWh=kWh_needed_to_heat),
            )
        # if fotovoltaics generates more than is consumed and battery is not charging, heat
        if (
            self.fotovoltaics is not None
            and (self.fotovoltaics.is_consumption_lower_than_production())
            and (not self.fotovoltaics.is_battery_charging())
        ):
            time_needed_to_heat_to_full = self.time_needed_to_heat_up_minutes(
                consumption_kWh=self.get_kWh_delta_from_temperatures(
                    tmp_act, self.set_tmp
                )
            )
            print("fve heating to full, time_needed: ", time_needed_to_heat_to_full)
            return (True, time_needed_to_heat_to_full)

        # get actual kWh in boiler from volume and tmp
        boiler_kWh_above_set = self.get_kWh_delta_from_temperatures(
            self.min_tmp, tmp_act
        )

        # get schedule of high tarif
        datetime_now = datetime.now()
        actual_time = datetime_now.time()
        actual_hdo_schedule = self.high_tarif_schedule[
            (self.high_tarif_schedule["time"] > actual_time)
            & (self.high_tarif_schedule["weekday"] >= datetime_now.weekday())
        ]
        # get first 6*12 rows
        actual_hdo_schedule = actual_hdo_schedule.head(2 * 12)
        # if the schedule is shorter than 12 hours, concat with beggining of the week
        if len(actual_hdo_schedule) < 2 * 12:
            # concat actual schedule with beggining of the week
            actual_hdo_schedule = pd.concat(
                [
                    actual_hdo_schedule,
                    self.high_tarif_schedule.head(2 * 12 - len(actual_hdo_schedule)),
                ]
            )

        # if the boiler is not connected to HDO, set unavailable minutes to 0
        if not self.hdo:
            actual_hdo_schedule["unavailable_minutes"] = 0
        # check if there is a need to heat up for the next event
        next_heat_event = self.eventChecker.next_calendar_heat_up_event()

        # if there is an event to heat up in the calendar
        if next_heat_event["minutes_to_event"] is not None:

            minutes_to_event = next_heat_event["minutes_to_event"]
            hours_to_event = minutes_to_event // 60
            degree_target = next_heat_event["degree_target"]
            minutes_needed_to_heat = self.time_needed_to_heat_up_minutes(
                consumption_kWh=self.get_kWh_delta_from_temperatures(
                    tmp_act, degree_target
                )
            )
            minutes_unavailable = actual_hdo_schedule.iloc[:hours_to_event][
                "unavailable_minutes"
            ].sum()
            minutes_needed_to_heat += minutes_unavailable

            if minutes_to_event < minutes_needed_to_heat:
                return (True, minutes_needed_to_heat)

        len_of_df = len(prediction_of_consumption)

        # check if there is a need to heat up for the next predicted consumption
        for i in range(len_of_df, 0, -1):
            time_to_consumption_minutes = (i * 60) - 30

            sum_of_consumption = (
                prediction_of_consumption.iloc[:i].sum().values[0]
                - boiler_kWh_above_set
            ) + self.get_kWh_loss_in_time(
                time_minutes=time_to_consumption_minutes, tmp_act=tmp_act
            )

            unavailible_minutes = actual_hdo_schedule.iloc[:i][
                "unavailable_minutes"
            ].sum()
            time_needed_to_heat = self.time_needed_to_heat_up_minutes(
                consumption_kWh=sum_of_consumption
            )
            if time_needed_to_heat > 0:
                time_needed_to_heat += unavailible_minutes

            if time_to_consumption_minutes < time_needed_to_heat:
                print(
                    f"time_needed_to_heat: {time_needed_to_heat}, time_to_consumption_minutes: {time_to_consumption_minutes}, sum_of_consumption: {sum_of_consumption}, unavailible_minutes: {unavailible_minutes}"
                )
                return (True, time_needed_to_heat)

        print(
            f"no need to heat, returning false, time_needed_to_heat: {time_needed_to_heat}, above_set: {boiler_kWh_above_set}"
        )
        return (False, 0)

    def showers_degrees(self, number_of_showers: int) -> float:
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

    def real_tmp(self, tmp_act: float) -> float:
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

    def set_measured_tmp(self, df: pd.DataFrame) -> None:
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
