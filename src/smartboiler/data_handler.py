from pathlib import Path

print("Running" if __name__ == "__main__" else "Importing", Path(__file__).resolve())
from operator import le
from click import group
from matplotlib.dates import drange
import pandas as pd
from influxdb import DataFrameClient, InfluxDBClient

import matplotlib.pyplot as plt
from datetime import date, datetime, timedelta
import numpy as np
import json
import logging
from math import dist
import numpy as np
from geopy.distance import geodesic
from haversine import haversine


class DataHandler:
    def __init__(
        self,
        influx_id,
        db_name,
        db_username,
        db_password,
        relay_entity_id,
        relay_power_entity_id,
        tmp_boiler_case_entity_id,
        tmp_output_water_entity_id,
        tmp_output_water_entity_id_2,
        device_tracker_entity_id,
        home_longitude,
        home_latitude,
        start_of_data=datetime(2023, 1, 1, 0, 0, 0, 0),
    ):
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

        self.start_of_data = start_of_data
        self.last_time_data_update = datetime.now()

    def get_actual_boiler_stats(
        self,
        group_by_time_interval="10m",
        limit=300,
        left_time_interval=datetime.now() - timedelta(hours=6),
        right_time_interval=datetime.now() - timedelta(hours=1),
    ):

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
        left_time_interval,
        right_time_interval,
    ):

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
        
    } 

    def haversine_dist(self, x1, x2, y1, y2):
        return haversine((x1, x2), (y1, y2), unit="km")

    # Data Processing

    def extract_features_from_longitude_latitude(self, df_old):
        #drop na in columns mean_latitude and mean_longitude
        df = df_old.copy()
        
        df = df.dropna(subset=["mean_latitude", "mean_longitude"])
        df.loc[:,"distance_from_home"] = np.vectorize(self.haversine_dist)(
            df["mean_latitude"],
            df["mean_longitude"],
            self.home_latitude,
            self.home_longitude,
        )

        df.loc[:,"heading_to_home"] = np.arctan2(
            df["mean_latitude"] - self.home_latitude,
            df["mean_longitude"] - self.home_longitude,
        )
        df.loc[:,"heading_to_home_sin"] = np.sin(df["heading_to_home"])
        df.loc[:,"heading_to_home_cos"] = np.cos(df["heading_to_home"])
        # resample by 10m mean
        df.loc[:,"time_stamp"] = df.index
        # calculate the speed of device
        df.loc[:,"time_diff"] = (
            df["time_stamp"].diff().dt.total_seconds() / 3600
        )  # Convert seconds to hours
        df.loc[:,"distance"] = np.vectorize(self.haversine_dist)(
            df["mean_latitude"],
            df["mean_longitude"],
            df["mean_latitude"].shift(1),
            df["mean_longitude"].shift(1),
        )  # calculate haversine distance
        # df['hours'] = (df['time_stamp'].astype(int) / 10**9) / 60*60 # convert to seconds
        # df['time_taken'] = df['hours'] - df['hours'].shift(1) # calculate time difference

        df.loc[:,"speed"] = df["distance"] / df["time_diff"]  # cal speed
        df.loc[df["speed"] > 200, "speed"] = 0
        df.loc[:,"speed_towards_home"] = df["speed"] * df["heading_to_home_cos"]
        return df

    def get_lowest_area_tmp(
        self,
        entity_id,
        measurement="°C",
        left_time_interval=datetime.now() - timedelta(days=21),
        right_time_interval=datetime.now(),
    ):
        query = f'SELECT min("value") AS "temperature" FROM "{self.db_name}"."autogen"."{measurement}" WHERE "entity_id"=\'{entity_id}\' ORDER BY "temperature" ASC LIMIT {limit}'
        result = self.dataframe_client.query(query)
        return result

    def get_df_from_queries(self, queries):
        df_all_list = []
        # iterate over key an value in data
        for key, value in queries.items():

            # print(value["sql_query"])
            result = self.dataframe_client.query(value["sql_query"])[
                value["measurement"]
            ]
            df = pd.DataFrame(result)
            df_all_list.append(df)

        df = pd.concat(df_all_list, axis=1)

        return df

    def process_kWh_water_consumption(self, df):
        df = df.resample("1min").mean()
        # df = df.reset_index(drop=True)
        df["consumed_heat_kJ"] = (
            df["water_flow_L_per_minute_mean"]
            * (df["water_temperature_mean"] - 10)
            * 4.186
            * 0.6
        )  

        df = self.extract_features_from_longitude_latitude(df)
        # all value in speed larger than 200 set to 0
        df.loc[df["speed"] > 200, "speed"] = 0

        df = df.groupby(pd.Grouper(freq="60T"))
        df = df.agg(
            {
                "consumed_heat_kJ": "sum",
                "water_flow_L_per_minute_mean": "mean",
                "water_temperature_mean": "mean",
                "outside_temperature_mean": "mean",
                "outside_humidity_mean": "mean",
                "outside_wind_speed_mean": "mean",
                "device_presence_distinct_count": "sum",
                "mean_longitude": "mean",
                "mean_latitude": "mean",
                "speed": "mean",
                "speed_towards_home": "mean",
                "distance_from_home": "mean",
                "heading_to_home_sin": "mean",
                "heading_to_home_cos": "mean",
            }
        )
        df["consumed_heat_kWh"] = df["consumed_heat_kJ"] / 3600
        df = df.drop(columns=["consumed_heat_kJ"])

        return df

    def get_data_for_training_model(
        self,
        left_time_interval=None,
        right_time_interval=datetime.now(),
        predicted_column="longtime_mean",
    ):
        if left_time_interval is None:
            left_time_interval = self.start_of_data
        queries = self.get_database_queries(
            left_time_interval=left_time_interval,
            right_time_interval=right_time_interval,
        )
        df_all = self.get_df_from_queries(queries)
        df_all = self.process_kWh_water_consumption(df_all)

        return self.transform_data_for_ml(df_all, predicted_column=predicted_column)

    def get_data_for_prediction(
        self,
        left_time_interval=datetime.now() - timedelta(days=5),
        right_time_interval=datetime.now(),
    ):
        
        left_time_interval = left_time_interval.replace(minute=0, second=0, microsecond=0)
        right_time_interval = right_time_interval.replace(minute=0, second=0, microsecond=0)
        
        queries = self.get_database_queries(
            left_time_interval=left_time_interval,
            right_time_interval=right_time_interval,
        )
        df_all = self.get_df_from_queries(queries)
        df_all = self.process_kWh_water_consumption(df_all)
        df_all.index = df_all.index.tz_localize(None)
        df_all, datetimes = self.transform_data_for_ml(df_all, predicted_column="longtime_mean")

        return df_all, datetimes

    def write_forecast_to_influxdb(self, forecast_list, measurement_name):
        current_time = datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ")


       # create a dict from list with keys from 0
        result_dict = dict(enumerate(forecast_list, 0))
        # Create a dictionary
        measurement_dict = {
            "measurement": "prediction",
            "time": current_time,
            "fields": result_dict,
        }

        self.influxdb_client.write_points([measurement_dict])
        # return measurement_dict

    def get_high_tarif_schedule(self):
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
        self, df, days_from_beginning_ignored=0, predicted_column="longtime_mean"
    ):
        # read pickles from data/pickles

        freq = 1
        freq_hour = f"{freq}H"

        df.index = pd.to_datetime(df.index)

        df.loc[:, "weekday"] = df.index.weekday
        df.loc[:, "hour"] = df.index.hour
        # df.loc[:, "minute"] = df.index.minute
        df.loc[:, "minute"] = 0

        # delete rows with weekday nan
        # df = df.dropna(subset=["weekday"])
        df["consumed_heat_kWh"] = df["consumed_heat_kWh"].fillna(0)

        # fill negative values with 0
        df["consumed_heat_kWh"] = df["consumed_heat_kWh"].clip(lower=0)

        # fill na in df based on column
        df["temperature"] = df[f"outside_temperature_mean"].fillna(method="ffill")
        df["humidity"] = df[f"outside_humidity_mean"].fillna(method="ffill")
        df["wind_speed"] = df[f"outside_wind_speed_mean"].fillna(method="ffill")
        df["count"] = df["device_presence_distinct_count"].fillna(method="ffill")
        df["mean_longitude"] = df["mean_longitude"].fillna(method="ffill")
        df["mean_latitude"] = df["mean_latitude"].fillna(method="ffill")
        df["speed"] = df["speed"].fillna(method="ffill")
        df["speed_towards_home"] = df["speed_towards_home"].fillna(method="ffill")
        df["distance_from_home"] = df["distance_from_home"].fillna(method="ffill")
        df["heading_to_home_sin"] = df["heading_to_home_sin"].fillna(method="ffill")
        df["heading_to_home_cos"] = df["heading_to_home_cos"].fillna(method="ffill")

        # add to column 'consumed_heat_kWh' 1,25/6 to each row
        # df = df.drop(df[df["consumed_heat_kWh"] == 0].sample(frac=0.7).index)
        df["consumed_heat_kWh"] += 1.25 / (24 // freq)
        # drop randomly 60 percent of rows where consumed_heat_kWh is 0
        window = 3

        df["longtime_mean"] = (
            df["consumed_heat_kWh"]
            .rolling(window=window, min_periods=1, center=True)
            .mean()
        )


        # drop consumed_heat_kWh
        
        df['last_3_week_mean'] = df.groupby([df.index.weekday, df.index.hour])['consumed_heat_kWh'].rolling(window=3, min_periods=1).mean().reset_index(level=[0,1], drop=True)
        df['last_3_week_mean'] = df['last_3_week_mean'].fillna(method='ffill')
        
        df['last_3_week_std'] = df.groupby([df.index.weekday, df.index.hour])['consumed_heat_kWh'].rolling(window=3, min_periods=1).std().reset_index(level=[0,1], drop=True)
        df['last_3_week_std'] = df['last_3_week_std'].fillna(method='ffill')
        
        df['last_3_week_max'] = df.groupby([df.index.weekday, df.index.hour])['consumed_heat_kWh'].rolling(window=3, min_periods=1).max().reset_index(level=[0,1], drop=True)
        df['last_3_week_max'] = df['last_3_week_max'].fillna(method='ffill')
        
        df['last_3_week_min'] = df.groupby([df.index.weekday, df.index.hour])['consumed_heat_kWh'].rolling(window=3, min_periods=1).min().reset_index(level=[0,1], drop=True)
        df['last_3_week_min'] = df['last_3_week_min'].fillna(method='ffill')
        
        df['last_3_week_skew'] = df.groupby([df.index.weekday, df.index.hour])['consumed_heat_kWh'].rolling(window=3, min_periods=1).skew().reset_index(level=[0,1], drop=True)
        df['last_3_week_skew'] = df['last_3_week_skew'].fillna(method='ffill')
        #fill 
        
        df['last_3_week_median'] = df.groupby([df.index.weekday, df.index.hour])['consumed_heat_kWh'].rolling(window=3, min_periods=1).median().reset_index(level=[0,1], drop=True)
        df['last_3_week_median'] = df['last_3_week_median'].fillna(method='ffill') 
        
        
        
        df_reverse = df.iloc[::-1]

        nan_indices = df_reverse[df_reverse['last_3_week_skew'].isna()].index

        for index in nan_indices:

            rolling_skew = df_reverse.loc[index:, 'consumed_heat_kWh'].rolling(window=3, min_periods=1).skew()
            rolling_std = df_reverse.loc[index:, 'consumed_heat_kWh'].rolling(window=3, min_periods=1).std()
            df_reverse.loc[index:, 'last_3_week_skew'] = np.where(df_reverse.loc[index:, 'last_3_week_skew'].isna(), rolling_skew, df_reverse.loc[index:, 'last_3_week_skew'])
            df_reverse.loc[index:, 'last_3_week_std'] = np.where(df_reverse.loc[index:, 'last_3_week_std'].isna(), rolling_std, df_reverse.loc[index:, 'last_3_week_std'])

        df['last_3_week_skew'] = df_reverse['last_3_week_skew'].iloc[::-1]
        df['last_3_week_std'] = df_reverse['last_3_week_std'].iloc[::-1]
        
        
        df = df.drop(columns=["consumed_heat_kWh"])
        # transform weekday, minute, hour to sin cos
        df["weekday_sin"] = np.sin(2 * df["weekday"] * np.pi / 7)
        df["weekday_cos"] = np.cos(2 * df["weekday"] * np.pi / 7)

        df["hour_sin"] = np.sin(2 * df["hour"] * np.pi / 24)
        df["hour_cos"] = np.cos(2 * df["hour"] * np.pi / 24)

        df["minute_sin"] = np.sin(2 * df["minute"] * np.pi / 60)
        df["minute_cos"] = np.cos(2 * df["minute"] * np.pi / 60)

        df = df[
            [
                "longtime_mean",
                "last_3_week_skew",
                "last_3_week_std",
                "distance_from_home",
                "speed_towards_home",
                "count",
                "heading_to_home_sin",
                "heading_to_home_cos",
                "temperature",
                "humidity",
                "wind_speed",
                "weekday_sin",
                "weekday_cos",
                "hour_sin",
                "hour_cos",

            ]
        ]
        # df = df[
        #     [
        #         predicted_column,
        #         "weekday_sin",
        #         "weekday_cos",
        #         "hour_sin",
        #         "hour_cos",
        #         "minute_sin",
        #         "minute_cos",
        #         "hea"
        #     ]
        # ]

        # df = df.dropna(subset=["longtime_mean", "distance_from_home"])

        # extract datetimes from index
        datetimes = df.index

        return (df.reset_index(drop=True), datetimes)

    def write_data(self, data):
        with open(self.data_file, "w") as file:
            json.dump(data, file)


if __name__ == "__main__":
    # test

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
    # df_from_db = data_handler.get_data_for_training_model()
    # dropna - remove rows with NaN
    # result = data_handler.transform_data_for_ml(df_from_db)
    dataHandler = dataHandler.get_data_for_high_tarif_info()

    # data_handler.get_data_for_prediction()
    # data_handler.get_actual_data()
    # data_handler.transform_data_for_ml()
    # data_handler.write_data()
