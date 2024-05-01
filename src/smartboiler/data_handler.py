# Created as a part of Master's Thesis "Using machine learning methods to save energy in a smart home"
# Faculty of Information Technology, Brno University of Technology, 2024
# Author: Adam Grünwald
#
# This module is used for handling the data from the database.

from typing import Optional

import pandas as pd
from influxdb import DataFrameClient, InfluxDBClient

from datetime import datetime, timedelta
import numpy as np

import numpy as np
from haversine import haversine


class DataHandler:
    """Class for handling the data from the database."""

    def __init__(
        self,
        influx_id: str,
        db_name: str,
        db_username: str,
        db_password: str,
        relay_entity_id: str,
        relay_power_entity_id: str,
        tmp_boiler_case_entity_id: str,
        tmp_output_water_entity_id: str,
        tmp_output_water_entity_id_2: str,
        device_tracker_entity_id: str,
        device_tracker_entity_id_2: str,
        home_longitude: float,
        home_latitude: float,
        start_of_data: Optional[datetime] = datetime(2023, 1, 1, 0, 0, 0, 0),
    ):
        """Init function for the DataHandler class.

        Args:
            influx_id (str): Influxdb server name.
            db_name (str): Name of the influxdb database with data for processing.
            db_username (str): Username for influxdb.
            db_password (str): Password for influxdb.
            relay_entity_id (str): Entity ID of shelly relay in home assistant.
            relay_power_entity_id (str): Entity ID of the shelly relay power in home assistant.
            tmp_boiler_case_entity_id (str): Entity ID of the boiler case temperature sensor.
            tmp_output_water_entity_id (str): Entity ID of the output water temperature sensor.
            tmp_output_water_entity_id_2 (str): Entity ID for the case of two output water temperature sensors.
            device_tracker_entity_id (str): Entity ID of the first device tracker.
            device_tracker_entity_id_2 (str): Entity ID of the second device tracker.
            home_longitude (float):l Longitude of the home.
            home_latitude (float): atitude of the home.
            start_of_data (Optional[datetime], optional): Start from when should be the data from db taken. Defaults to datetime(2023, 1, 1, 0, 0, 0, 0).
        """

        self.influx_id = influx_id
        self.db_name = db_name
        self.db_username = db_username
        self.db_password = db_password
        self.group_by_time_interval = "30min"
        self.relay_entity_id = relay_entity_id
        self.relay_power_entity_id = relay_power_entity_id
        self.tmp_boiler_case_entity_id = tmp_boiler_case_entity_id
        self.tmp_output_water_entity_id = tmp_output_water_entity_id
        self.tmp_output_water_entity_id_2 = tmp_output_water_entity_id_2

        self.home_longitude = home_longitude
        self.home_latitude = home_latitude

        self.device_tracker_entity_id = device_tracker_entity_id
        self.device_tracker_entity_id_2 = device_tracker_entity_id_2

        self.start_of_data = start_of_data
        self.last_time_data_update = datetime.now()

        # initialize clients
        self.dataframe_client = DataFrameClient(
            host=self.influx_id,
            port=8086,
            username=self.db_username,
            password=self.db_password,
            database=self.db_name,
        )
        self.influxdb_client = InfluxDBClient(
            host=self.influx_id,
            port=8086,
            username=self.db_username,
            password=self.db_password,
            database=self.db_name,
            retries=5,
            timeout=1,
        )

    def get_actual_boiler_stats(
        self,
        group_by_time_interval: Optional[str] = "10m",
        limit: Optional[int] = 300,
        left_time_interval: Optional[datetime] = datetime.now() - timedelta(hours=6),
        right_time_interval: Optional[datetime] = datetime.now() - timedelta(hours=1),
    ) -> dict:
        # TODO handle correct the timezones
        """Method to retrieve the actual stats for the boilers. Serves for algorithm of controling the boiler.

        Args:
            group_by_time_interval (Optional[str], optional): Time interval by which should be the data grouped. Defaults to "10m".
            limit (Optional[int], optional): Limit of the last N values. Defaults to 300.
            left_time_interval (Optional[datetime], optional): Left interval in which we are interested. Defaults to datetime.now()-timedelta(hours=6).
            right_time_interval (Optional[datetime], optional): Right time interval. the -1h is because of the correction of Europe/Prague timezone. Defaults to datetime.now()-timedelta(hours=1).

        Returns:
            dict: Dictionary of the actual stats for boiler_case_tmp, is_boiler_on, boiler_case_last_time_entry, is_boiler_on_last_time_entry.
        """
        left_time_interval = f"'{left_time_interval.strftime('%Y-%m-%dT%H:%M:%SZ')}'"
        right_time_interval = f"'{right_time_interval.strftime('%Y-%m-%dT%H:%M:%SZ')}'"
        actual_boiler_stats = {
            "boiler_temperature": {
                "sql_query": f'SELECT last("value") AS "boiler_case_tmp" FROM "{self.db_name}"."autogen"."°C" WHERE  "entity_id"=\'{self.tmp_boiler_case_entity_id}\' ',
                "measurement": "°C",
            },
            "is_boiler_on": {
                "sql_query": f'SELECT last("value") AS "is_boiler_on" FROM "{self.db_name}"."autogen"."state" WHERE "entity_id"=\'{self.relay_entity_id}\' ',
                "measurement": "state",
            },
        }
        data = self.get_df_from_queries(actual_boiler_stats)

        boiler_case_tmp = data["boiler_case_tmp"].dropna().reset_index()
        is_boiler_on = data["is_boiler_on"].dropna().reset_index()

        boiler_case_tmp["index"] = pd.to_datetime(boiler_case_tmp["index"])
        is_boiler_on["index"] = pd.to_datetime(is_boiler_on["index"])

        # last values
        boiler_case_last_time_entry = boiler_case_tmp["index"].iloc[-1]
        boiler_case_tmp = boiler_case_tmp["boiler_case_tmp"].iloc[-1]
        is_boiler_on_last_time_entry = is_boiler_on["index"].iloc[-1]
        is_boiler_on = bool(is_boiler_on["is_boiler_on"].iloc[-1])

        return {
            "boiler_case_tmp": boiler_case_tmp,
            "is_boiler_on": is_boiler_on,
            "boiler_case_last_time_entry": boiler_case_last_time_entry,
            "is_boiler_on_last_time_entry": is_boiler_on_last_time_entry,
        }

    def get_database_queries(
        self,
        left_time_interval: datetime,
        right_time_interval: datetime,
    ) -> dict:
        """Method to create dictionary of queries for the database.

        Args:
            left_time_interval (datetime): Left time interval.
            right_time_interval (datetime): Right time interval.

        Returns:
            dict: Dictionary of queries for the database.
        """

        group_by_time_interval = "1m"

        # format datetime to YYYY-MM-DDTHH:MM:SSZ
        left_time_interval = f"'{left_time_interval.strftime('%Y-%m-%dT%H:%M:%SZ')}'"
        right_time_interval = f"'{right_time_interval.strftime('%Y-%m-%dT%H:%M:%SZ')}'"

        return {
            "water_flow": {
                "sql_query": f'SELECT mean("value") AS "water_flow_L_per_minute_mean" FROM "{self.db_name}"."autogen"."L/min" WHERE time > {left_time_interval} AND time <= {right_time_interval} GROUP BY time({group_by_time_interval}) FILL(0)',
                "measurement": "L/min",
            },
            "water_temperature": {
                "sql_query": f'SELECT mean("value") AS "water_temperature_mean" FROM "{self.db_name}"."autogen"."°C" WHERE time > {left_time_interval} AND time <= {right_time_interval} AND ("entity_id"=\'{self.tmp_output_water_entity_id}\' OR "entity_id"=\'{self.tmp_output_water_entity_id_2}\') GROUP BY time({group_by_time_interval}) FILL(0)',
                "measurement": "°C",
            },
            "temperature": {
                "sql_query": f'SELECT mean("temperature") AS "outside_temperature_mean" FROM "{self.db_name}"."autogen"."state" WHERE time > {left_time_interval} AND time <= {right_time_interval} AND "domain"=\'weather\' AND "entity_id"=\'domov\' GROUP BY time({group_by_time_interval}) FILL(null)',
                "measurement": "state",
            },
            "humidity": {
                "sql_query": f'SELECT mean("humidity") AS "outside_humidity_mean" FROM "{self.db_name}"."autogen"."state" WHERE time > {left_time_interval} AND time <= {right_time_interval} AND "domain"=\'weather\' AND "entity_id"=\'domov\' GROUP BY time({group_by_time_interval}) FILL(null)',
                "measurement": "state",
            },
            "wind_speed": {
                "sql_query": f'SELECT mean("wind_speed") AS "outside_wind_speed_mean" FROM "{self.db_name}"."autogen"."state" WHERE time > {left_time_interval} AND time <= {right_time_interval} AND "entity_id"=\'domov\' GROUP BY time({group_by_time_interval}) FILL(null)',
                "measurement": "state",
            },
            "presence": {
                "sql_query": f'SELECT count(distinct("friendly_name_str")) AS "device_presence_distinct_count" FROM "{self.db_name}"."autogen"."state" WHERE time > {left_time_interval} AND time <= {right_time_interval} AND "domain"=\'device_tracker\' AND "state"=\'home\' GROUP BY time({group_by_time_interval}) FILL(0)',
                "measurement": "state",
            },
            "boiler_water_temperature": {
                "sql_query": f'SELECT mean("value") AS "boiler_water_temperature_mean" FROM "{self.db_name}"."autogen"."°C" WHERE time > {left_time_interval} AND time <= {right_time_interval} AND "entity_id"=\'{self.tmp_boiler_case_entity_id}\' GROUP BY time({group_by_time_interval}) FILL(null)',
                "measurement": "°C",
            },
            "boiler_relay_status": {
                "sql_query": f'SELECT last("value") AS "boiler_relay_status" FROM "{self.db_name}"."autogen"."state" WHERE time > {left_time_interval} AND time <= {right_time_interval} AND "entity_id"=\'{self.relay_entity_id}\' GROUP BY time({group_by_time_interval}) FILL(null)',
                "measurement": "state",
            },
            "device_longitude": {
                "sql_query": f'SELECT mean("longitude") AS "mean_longitude" FROM "{self.db_name}"."autogen"."state" WHERE time > {left_time_interval} AND time <= {right_time_interval} AND "domain"=\'device_tracker\' AND "entity_id"=\'{self.device_tracker_entity_id}\' GROUP BY time({group_by_time_interval}) FILL(previous)',
                "measurement": "state",
            },
            "device_latitude": {
                "sql_query": f'SELECT mean("latitude") AS "mean_latitude" FROM "{self.db_name}"."autogen"."state" WHERE time > {left_time_interval} AND time <= {right_time_interval} AND "domain"=\'device_tracker\' AND "entity_id"=\'{self.device_tracker_entity_id}\' GROUP BY time({group_by_time_interval}) FILL(previous)',
                "measurement": "state",
            },
            "device_longitude_2": {
                "sql_query": f'SELECT mean("longitude") AS "mean_longitude_2" FROM "{self.db_name}"."autogen"."state" WHERE time > {left_time_interval} AND time <= {right_time_interval} AND "domain"=\'device_tracker\' AND "entity_id"=\'{self.device_tracker_entity_id_2}\' GROUP BY time({group_by_time_interval}) FILL(previous)',
                "measurement": "state",
            },
            "device_latitude_2": {
                "sql_query": f'SELECT mean("latitude") AS "mean_latitude_2" FROM "{self.db_name}"."autogen"."state" WHERE time > {left_time_interval} AND time <= {right_time_interval} AND "domain"=\'device_tracker\' AND "entity_id"=\'{self.device_tracker_entity_id_2}\' GROUP BY time({group_by_time_interval}) FILL(previous)',
                "measurement": "state",
            },
        }

    def haversine_dist(self, x1: float, x2: float, y1: float, y2: float) -> float:
        """Calculates the distance between two coordinates

        Args:
            x1 (float): X of the first coordinate.
            x2 (float): X of the second coordinate.
            y1 (float): Y of the first coordinate.
            y2 (float): Y of the second coordinate.

        Returns:
            float: Distance between two coordinates.
        """

        return haversine((x1, x2), (y1, y2), unit="km")

    # Data Processing

    def extract_features_from_longitude_latitude(
        self, df_old: pd.DataFrame
    ) -> pd.DataFrame:
        """Extracts the features from longitude and latitude as speed_towards_home, distance_from_home, heading_to_home_sin, heading_to_home_cos.

        Args:
            df_old (pd.DataFrame): Incoming dataframe.

        Returns:
            pd.DataFrame: Dataframe with extracted features.
        """

        # drop na in columns mean_latitude and mean_longitude
        df = df_old.copy()

        df = df.dropna(subset=["mean_latitude", "mean_longitude"])
        df.loc[:, "distance_from_home"] = np.vectorize(self.haversine_dist)(
            df["mean_latitude"],
            df["mean_longitude"],
            self.home_latitude,
            self.home_longitude,
        )

        df.loc[:, "heading_to_home"] = np.arctan2(
            df["mean_latitude"] - self.home_latitude,
            df["mean_longitude"] - self.home_longitude,
        )
        df.loc[:, "heading_to_home_sin"] = np.sin(df["heading_to_home"])
        df.loc[:, "heading_to_home_cos"] = np.cos(df["heading_to_home"])
        # resample by 10m mean
        df.loc[:, "time_stamp"] = df.index
        # calculate the speed of device
        df.loc[:, "time_diff"] = (
            df["time_stamp"].diff().dt.total_seconds() / 3600
        )  # Convert seconds to hours
        df.loc[:, "distance"] = np.vectorize(self.haversine_dist)(
            df["mean_latitude"],
            df["mean_longitude"],
            df["mean_latitude"].shift(1),
            df["mean_longitude"].shift(1),
        )  # calculate haversine distance

        df.loc[:, "speed"] = df["distance"] / df["time_diff"]  # cal speed
        df.loc[df["speed"] > 200, "speed"] = 0
        df.loc[:, "speed_towards_home"] = df["speed"] * df["heading_to_home_cos"]

        #######

        df = df.dropna(subset=["mean_latitude_2", "mean_longitude_2"])
        df.loc[:, "distance_from_home_2"] = np.vectorize(self.haversine_dist)(
            df["mean_latitude_2"],
            df["mean_longitude_2"],
            self.home_latitude,
            self.home_longitude,
        )

        df.loc[:, "heading_to_home_2"] = np.arctan2(
            df["mean_latitude_2"] - self.home_latitude,
            df["mean_longitude_2"] - self.home_longitude,
        )
        df.loc[:, "heading_to_home_sin_2"] = np.sin(df["heading_to_home_2"])
        df.loc[:, "heading_to_home_cos_2"] = np.cos(df["heading_to_home_2"])
        # resample by 10m mean
        df.loc[:, "time_stamp_2"] = df.index
        # calculate the speed of device
        df.loc[:, "time_diff_2"] = (
            df["time_stamp_2"].diff().dt.total_seconds() / 3600
        )  # Convert seconds to hours
        df.loc[:, "distance_2"] = np.vectorize(self.haversine_dist)(
            df["mean_latitude_2"],
            df["mean_longitude_2"],
            df["mean_latitude_2"].shift(1),
            df["mean_longitude_2"].shift(1),
        )  # calculate haversine distance

        df.loc[:, "speed_2"] = df["distance_2"] / df["time_diff_2"]  # cal speed
        df.loc[df["speed_2"] > 200, "speed_2"] = 0
        df.loc[:, "speed_towards_home_2"] = df["speed_2"] * df["heading_to_home_cos_2"]

        return df

    def get_df_from_queries(self, queries: dict) -> pd.DataFrame:
        """Get the data from the database based on the queries.

        Args:
            queries (dict): Dictionary of queries.

        Returns:
            pd.DataFrame: Result dataframe.
        """

        df_all_list = []
        # iterate over key an value in data
        for _, value in queries.items():

            # print(value["sql_query"])
            result = self.dataframe_client.query(value["sql_query"])[
                value["measurement"]
            ]
            df = pd.DataFrame(result)
            df_all_list.append(df)

        df = pd.concat(df_all_list, axis=1)

        return df

    def process_kWh_water_consumption(self, df: pd.DataFrame) -> pd.DataFrame:
        """Method to process the data from the database and calculate the consumed heat in kWh

        Args:
            df (pd.DataFrame): Dataframe with statistics from db.

        Returns:
            pd.DataFrame: Dataframe with processed data.
        """

        divide_heat_coefficient = 0.6
        # resample the data by 1 min as the flow is represended as L/min
        df = df.resample("1min").mean()

        # calculation of consumed kJ
        df["consumed_heat_kJ"] = (
            df["water_flow_L_per_minute_mean"]
            * (df["water_temperature_mean"] - 10)
            * 4.186
            * divide_heat_coefficient
        )
        # extraction of features from device position
        df = self.extract_features_from_longitude_latitude(df)

        # cleaning the outliners from speed feature
        df.loc[df["speed"] > 200, "speed"] = 0

        # aggregate by 60 minutes
        df = df.groupby(pd.Grouper(freq="60T"))
        df = df.agg(
            {
                "consumed_heat_kJ": "sum",
                "water_flow_L_per_minute_mean": "mean",
                "water_temperature_mean": "mean",
                "outside_temperature_mean": "mean",
                "outside_humidity_mean": "mean",
                "outside_wind_speed_mean": "mean",
                "device_presence_distinct_count": "max",
                "mean_longitude": "mean",
                "mean_latitude": "mean",
                "speed": "mean",
                "speed_towards_home": "mean",
                "distance_from_home": "mean",
                "heading_to_home_sin": "mean",
                "heading_to_home_cos": "mean",
                "mean_longitude_2": "mean",
                "mean_latitude_2": "mean",
                "speed_2": "mean",
                "speed_towards_home_2": "mean",
                "distance_from_home_2": "mean",
                "heading_to_home_sin_2": "mean",
                "heading_to_home_cos_2": "mean",
            }
        )

        # transform kj to kWh
        df["consumed_heat_kWh"] = df["consumed_heat_kJ"] / 3600
        df = df.drop(columns=["consumed_heat_kJ"])

        return df

    def get_data_for_training_model(
        self,
        left_time_interval: Optional[datetime] = None,
        right_time_interval: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """Method to retrieve data for training.

        Args:
            left_time_interval (Optional[datetime], optional): Left time datetime. Defaults to self.start_of_data.
            right_time_interval (Optional[datetime], optional): Right time datetime. Defaults to datetime.now().

        Returns:
            pd.DataFrame: Resulting dataframe.
        """

        # default the intervals if None
        if left_time_interval is None:
            left_time_interval = self.start_of_data

        if right_time_interval is None:
            right_time_interval = datetime.now()

        left_time_interval = left_time_interval.replace(
            minute=0, second=0, microsecond=0
        )
        right_time_interval = right_time_interval.replace(
            minute=0, second=0, microsecond=0
        )

        # retrieve the queries based on the intervals
        queries = self.get_database_queries(
            left_time_interval=left_time_interval,
            right_time_interval=right_time_interval,
        )
        # get the data from the database
        df_all = self.get_df_from_queries(queries)
        # process the data
        df_all = self.process_kWh_water_consumption(df_all)
        # transform for ML
        return self.transform_data_for_ml(df_all)

    def get_data_for_prediction(
        self,
        left_time_interval: Optional[datetime] = None,
        right_time_interval: Optional[datetime] = None,
    ) -> tuple[pd.DataFrame, pd.DatetimeIndex]:
        """Method to get data for the prediction.

        Args:
            left_time_interval (Optional[datetime], optional): Left time datetime. Defaults to datetime.now()-timedelta(days=5).
            right_time_interval (Optional[datetime], optional): Right time datetime. Defaults to datetime.now().

        Returns:
            tuple[pd.DataFrame, pd.DatetimeIndex]: Tuple with dataframe and datetime index.
        """
        # defaulting the intervals if None
        if left_time_interval is None:
            left_time_interval = datetime.now() - timedelta(days=5)
        if right_time_interval is None:
            right_time_interval = datetime.now()

        # replace the minutes, seconds and microseconds with 0 to get the full hour
        left_time_interval = left_time_interval.replace(
            minute=0, second=0, microsecond=0
        )
        right_time_interval = right_time_interval.replace(
            minute=0, second=0, microsecond=0
        )
        # get the queries for the database
        queries = self.get_database_queries(
            left_time_interval=left_time_interval,
            right_time_interval=right_time_interval,
        )
        # get the data from the database
        df_all = self.get_df_from_queries(queries)

        # process the data
        df_all = self.process_kWh_water_consumption(df_all)

        # transform the data for ML
        df_all.index = df_all.index.tz_localize(None)

        # return the dataframe and the datetimes
        df_all, datetimes = self.transform_data_for_ml(df_all)

        return df_all, datetimes

    def write_forecast_to_influxdb(self, df: pd.DataFrame) -> None:
        """Method which writes the current forecast to the influxdb.

        Args:
            df (pd.DataFrame): Dataframe with current prediction.
        """
        current_time = datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ")

        df.index = df.index.astype(str)

        # Create dictionary
        result_dict = df.squeeze().to_dict()
        print(f"Writing forecast to influxdb: {result_dict}")
        # Create a dictionary
        measurement_dict = {
            "measurement": "prediction",
            "time": current_time,
            "fields": result_dict,
        }

        self.influxdb_client.write_points([measurement_dict])

    def get_high_tarif_schedule(self) -> pd.DataFrame:
        """Method to retrieve the high tarif schedule from data for the last 14 days.

        Returns:
            pd.DataFrame: Dataframe with high tarif schedule by hour in each weekday.
        """

        print("Getting high tarif schedule")
        left_time_interval = datetime.now() - timedelta(days=14)
        queries = self.get_database_queries(
            left_time_interval=left_time_interval, right_time_interval=datetime.now()
        )
        df_all = self.get_df_from_queries(queries)
        df_all.index = df_all.index.tz_localize(None)
        data_resampled = df_all.resample("10T").max()
        data_resampled = data_resampled["boiler_relay_status"]
        data_resampled = data_resampled.notna().astype(int)
        data_resampled = data_resampled.resample("30T").sum() * 10
        # group by weekday and hour and minute and calculate max
        grouped = data_resampled.groupby(
            [
                data_resampled.index.weekday,
                data_resampled.index.hour,
                data_resampled.index.minute,
            ]
        ).max()

        # not null values to 1, null as 0
        df_reset = grouped.reset_index()
        df_reset["time"] = (
            df_reset["level_1"].astype(str) + ":" + df_reset["level_2"].astype(str)
        )
        df_reset["time"] = pd.to_datetime(df_reset["time"], format="%H:%M").dt.time
        df_reset["weekday"] = df_reset["level_0"]
        df_reset["unavailable_minutes"] = abs(30 - df_reset["boiler_relay_status"])
        df_reset = df_reset.drop(
            columns=["level_0", "level_1", "level_2", "boiler_relay_status"]
        )

        return df_reset

    def transform_data_for_ml(
        self, df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DatetimeIndex]:
        """Method to transform the data for ML model.

        Args:
            df (pd.DataFrame): Dataframe to transform.

        Returns:
            tuple[pd.DataFrame, pd.DatetimeIndex]: Transformed dataframe and datetime.
        """

        # frequency i hours
        freq = 1
        df.index = pd.to_datetime(df.index)

        df.loc[:, "weekday"] = df.index.weekday
        df.loc[:, "hour"] = df.index.hour
        df.loc[:, "minute"] = 0

        # fill consumed_heat_kWh with 0 if nan
        df["consumed_heat_kWh"] = df["consumed_heat_kWh"].fillna(0)

        # fill negative values with 0
        df["consumed_heat_kWh"] = df["consumed_heat_kWh"].clip(lower=0)

        # ffill na in df based on column
        df["temperature"] = df[f"outside_temperature_mean"].fillna(method="ffill")
        df["humidity"] = df[f"outside_humidity_mean"].fillna(method="ffill")
        df["wind_speed"] = df[f"outside_wind_speed_mean"].fillna(method="ffill")
        df["count"] = df["device_presence_distinct_count"].fillna(method="ffill")

        df["mean_longitude"] = df["mean_longitude"].fillna(method="ffill")
        df["mean_latitude"] = df["mean_latitude"].fillna(method="ffill")
        df["speed"] = df["speed"].fillna(0)
        df["speed_towards_home"] = df["speed_towards_home"].fillna(0)
        df["distance_from_home"] = df["distance_from_home"].fillna(method="ffill")
        df["heading_to_home_sin"] = df["heading_to_home_sin"].fillna(method="ffill")
        df["heading_to_home_cos"] = df["heading_to_home_cos"].fillna(method="ffill")

        df["mean_longitude_2"] = df["mean_longitude_2"].fillna(method="ffill")
        df["mean_latitude_2"] = df["mean_latitude_2"].fillna(method="ffill")
        df["speed_2"] = df["speed_2"].fillna(0)
        df["speed_towards_home_2"] = df["speed_towards_home_2"].fillna(0)
        df["distance_from_home_2"] = df["distance_from_home_2"].fillna(method="ffill")
        df["heading_to_home_sin_2"] = df["heading_to_home_sin_2"].fillna(method="ffill")
        df["heading_to_home_cos_2"] = df["heading_to_home_cos_2"].fillna(method="ffill")

        # adding cooling down to the consumed heat
        df["consumed_heat_kWh"] += 1.25 / (24 // freq)

        window = 3
        # creating a 3 hour rolling window for the mean of the consumed heat for better prediction
        df["longtime_mean"] = (
            df["consumed_heat_kWh"]
            .rolling(window=window, min_periods=1, center=True)
            .mean()
        )
        # extrahing next features
        df["last_3_week_mean"] = (
            df.groupby([df.index.weekday, df.index.hour])["consumed_heat_kWh"]
            .rolling(window=3, min_periods=1)
            .mean()
            .reset_index(level=[0, 1], drop=True)
        )
        df["last_3_week_mean"] = df["last_3_week_mean"].fillna(method="ffill")

        df["last_3_week_std"] = (
            df.groupby([df.index.weekday, df.index.hour])["consumed_heat_kWh"]
            .rolling(window=3, min_periods=1)
            .std()
            .reset_index(level=[0, 1], drop=True)
        )
        df["last_3_week_std"] = df["last_3_week_std"].fillna(method="ffill")

        df["last_3_week_max"] = (
            df.groupby([df.index.weekday, df.index.hour])["consumed_heat_kWh"]
            .rolling(window=3, min_periods=1)
            .max()
            .reset_index(level=[0, 1], drop=True)
        )
        df["last_3_week_max"] = df["last_3_week_max"].fillna(method="ffill")

        df["last_3_week_min"] = (
            df.groupby([df.index.weekday, df.index.hour])["consumed_heat_kWh"]
            .rolling(window=3, min_periods=1)
            .min()
            .reset_index(level=[0, 1], drop=True)
        )
        df["last_3_week_min"] = df["last_3_week_min"].fillna(method="ffill")

        df["last_3_week_skew"] = (
            df.groupby([df.index.weekday, df.index.hour])["consumed_heat_kWh"]
            .rolling(window=3, min_periods=1)
            .skew()
            .reset_index(level=[0, 1], drop=True)
        )
        df["last_3_week_skew"] = df["last_3_week_skew"].fillna(method="ffill")
        # fill

        df["last_3_week_median"] = (
            df.groupby([df.index.weekday, df.index.hour])["consumed_heat_kWh"]
            .rolling(window=3, min_periods=1)
            .median()
            .reset_index(level=[0, 1], drop=True)
        )
        df["last_3_week_median"] = df["last_3_week_median"].fillna(method="ffill")

        df_reverse = df.iloc[::-1]

        # getting nan indices and filling them with rolling mean from the end of the dataframe
        nan_indices = df_reverse[df_reverse["last_3_week_skew"].isna()].index
        for index in nan_indices:

            rolling_skew = (
                df_reverse.loc[index:, "consumed_heat_kWh"]
                .rolling(window=3, min_periods=1)
                .skew()
            )
            rolling_std = (
                df_reverse.loc[index:, "consumed_heat_kWh"]
                .rolling(window=3, min_periods=1)
                .std()
            )
            df_reverse.loc[index:, "last_3_week_skew"] = np.where(
                df_reverse.loc[index:, "last_3_week_skew"].isna(),
                rolling_skew,
                df_reverse.loc[index:, "last_3_week_skew"],
            )
            df_reverse.loc[index:, "last_3_week_std"] = np.where(
                df_reverse.loc[index:, "last_3_week_std"].isna(),
                rolling_std,
                df_reverse.loc[index:, "last_3_week_std"],
            )

        df["last_3_week_skew"] = df_reverse["last_3_week_skew"].iloc[::-1]
        df["last_3_week_std"] = df_reverse["last_3_week_std"].iloc[::-1]

        df = df.drop(columns=["consumed_heat_kWh"])

        # transform weekday, minute, hour to sin cos
        df["weekday_sin"] = np.sin(2 * df["weekday"] * np.pi / 7)
        df["weekday_cos"] = np.cos(2 * df["weekday"] * np.pi / 7)

        df["hour_sin"] = np.sin(2 * df["hour"] * np.pi / 24)
        df["hour_cos"] = np.cos(2 * df["hour"] * np.pi / 24)

        df["minute_sin"] = np.sin(2 * df["minute"] * np.pi / 60)
        df["minute_cos"] = np.cos(2 * df["minute"] * np.pi / 60)

        # result df with features and target variable as a first
        df = df[
            [
                "longtime_mean",
                "last_3_week_skew",
                "last_3_week_std",
                "distance_from_home",
                "speed_towards_home",
                # "distance_from_home_2",
                # "speed_towards_home_2",
                "count",
                "heading_to_home_sin",
                "heading_to_home_cos",
                # "heading_to_home_sin_2",
                # "heading_to_home_cos_2",
                "temperature",
                "humidity",
                "wind_speed",
                "weekday_sin",
                "weekday_cos",
                "hour_sin",
                "hour_cos",
            ]
        ]

        # extract datetimes from index
        datetimes = df.index

        return (df.reset_index(drop=True), datetimes)


if __name__ == "__main__":

    dataHandler = DataHandler(
        influx_id="localhost",
        db_name="smart_home_zukalovi",
        db_username="root",
        db_password="root",
        relay_entity_id="shelly1pm_84cca8b07eae",
        relay_power_entity_id="shelly1pm_84cca8b07eae_power",
        tmp_boiler_case_entity_id="esphome_web_c771e8_tmp3",
        tmp_output_water_entity_id="esphome_web_c771e8_ntc_temperature_b_constant_2",
        start_of_data=datetime(2024, 3, 1, 0, 0, 0, 0),
    )
