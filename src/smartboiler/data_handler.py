from pathlib import Path
print('Running' if __name__ == '__main__' else 'Importing', Path(__file__).resolve())
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
        self.dataframe_client = DataFrameClient(
            host=self.influx_id,
            port=8086,
            username=self.db_username,
            password=self.db_password,
            database=self.db_name,
        )
        self.influxdb_client = InfluxDBClient(host = self.influx_id, port=8086, username=self.db_username, password=self.db_password, database=self.db_name,retries=5, timeout=1)

        self.start_of_data = start_of_data
        self.last_time_data_update = datetime.now()
        
        
    def get_actual_boiler_stats(self, group_by_time_interval = "10m", limit = 300, left_time_interval = datetime.now() - timedelta(hours=6), right_time_interval = datetime.now() - timedelta(hours=1)):

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
        
        boiler_case_tmp = data['boiler_case_tmp'].dropna().reset_index()
        is_boiler_on = data['is_boiler_on'].dropna().reset_index()
        
        boiler_case_tmp['index'] = pd.to_datetime(boiler_case_tmp['index'])
        is_boiler_on['index'] = pd.to_datetime(is_boiler_on['index'])
        
        boiler_case_last_time_entry = boiler_case_tmp['index'].iloc[-1]
        boiler_case_tmp = boiler_case_tmp['boiler_case_tmp'].iloc[-1]
        is_boiler_on_last_time_entry = is_boiler_on['index'].iloc[-1]
        print(is_boiler_on)
        is_boiler_on = bool(is_boiler_on['is_boiler_on'].iloc[-1])
        
        return {'boiler_case_tmp': boiler_case_tmp, 'is_boiler_on': is_boiler_on, 'boiler_case_last_time_entry': boiler_case_last_time_entry, 'is_boiler_on_last_time_entry': is_boiler_on_last_time_entry}
        


    def get_database_queries(
        self,
        left_time_interval,
        right_time_interval,
    ):

        group_by_time_interval = '5s'
        
        # format datetime to YYYY-MM-DDTHH:MM:SSZ
        left_time_interval = f"'{left_time_interval.strftime('%Y-%m-%dT%H:%M:%SZ')}'"
        right_time_interval = f"'{right_time_interval.strftime('%Y-%m-%dT%H:%M:%SZ')}'"

        queries = {
            "water_flow": {
                "sql_query": f'SELECT mean("value") AS "water_flow_L_per_minute_mean" FROM "{self.db_name}"."autogen"."L/min" WHERE time > {left_time_interval} AND time < {right_time_interval} GROUP BY time({group_by_time_interval}) FILL(0)',
                "measurement": "L/min",
            },
            "water_temperature": {
                "sql_query": f'SELECT mean("value") AS "water_temperature_mean" FROM "{self.db_name}"."autogen"."°C" WHERE time > {left_time_interval} AND time < {right_time_interval} AND "entity_id"=\'{self.tmp_output_water_entity_id}\' GROUP BY time({group_by_time_interval}) FILL(null)',
                "measurement": "°C",
            },
            "temperature": {
                "sql_query": f'SELECT mean("temperature") AS "outside_temperature_mean" FROM "{self.db_name}"."autogen"."state" WHERE time > {left_time_interval} AND time < {right_time_interval} AND "domain"=\'weather\' AND "entity_id"=\'domov\' GROUP BY time({group_by_time_interval}) FILL(null)',
                "measurement": "state",
            },
            "humidity": {
                "sql_query": f'SELECT mean("humidity") AS "outside_humidity_mean" FROM "{self.db_name}"."autogen"."state" WHERE time > {left_time_interval} AND time < {right_time_interval} AND "domain"=\'weather\' AND "entity_id"=\'domov\' GROUP BY time({group_by_time_interval}) FILL(null)',
                "measurement": "state",
            },
            "wind_speed": {
                "sql_query": f'SELECT mean("wind_speed") AS "outside_wind_speed_mean" FROM "{self.db_name}"."autogen"."state" WHERE time > {left_time_interval} AND time < {right_time_interval} AND "entity_id"=\'domov\' GROUP BY time({group_by_time_interval}) FILL(null)',
                "measurement": "state",
            },
            "presence": {
                "sql_query": f'SELECT count(distinct("friendly_name_str")) AS "device_presence_distinct_count" FROM "{self.db_name}"."autogen"."state" WHERE time > {left_time_interval} AND time < {right_time_interval} AND "domain"=\'device_tracker\' AND "state"=\'home\' GROUP BY time({group_by_time_interval}) FILL(0)',
                "measurement": "state",
            },
            "boiler_water_temperature": {
                "sql_query": f'SELECT mean("value") AS "boiler_water_temperature_mean" FROM "{self.db_name}"."autogen"."°C" WHERE time > {left_time_interval} AND time < {right_time_interval} AND "entity_id"=\'{self.tmp_boiler_case_entity_id}\' GROUP BY time({group_by_time_interval}) FILL(null)',
                "measurement": "°C",
            },
            "boiler_relay_status": {"sql_query": f'SELECT last("value") AS "boiler_relay_status" FROM "{self.db_name}"."autogen"."state" WHERE time > {left_time_interval} AND time < {right_time_interval} AND "entity_id"=\'{self.relay_entity_id}\' GROUP BY time({group_by_time_interval}) FILL(null)',
                                    "measurement": "state"},
        }
        return queries

    def get_df_from_queries(self, queries):
        df_all_list = []
        # iterate over key an value in data
        for key, value in queries.items():
            print("Querying: ", key, value["sql_query"])
            # get data from influxdb
            result = self.dataframe_client.query(value["sql_query"])[
                value["measurement"]
            ]
            
            df = pd.DataFrame(result)
            df_all_list.append(df)
            
        df = pd.concat(df_all_list, axis=1)
        
        return df
    def process_kWh_water_consumption(self, df):
        df = df.resample('1min').mean()
        # df = df.reset_index(drop=True)
        df["consumed_heat_kJ"] = (
            df["water_flow_L_per_minute_mean"]
            * (df["water_temperature_mean"] - 10)
            * 4.186
            *0.6
        )
        df = df.groupby(pd.Grouper(freq='30T'))
        df = df.agg({'consumed_heat_kJ': 'sum', 'water_flow_L_per_minute_mean': 'mean', 'water_temperature_mean': 'mean', 'outside_temperature_mean': 'mean', 'outside_humidity_mean': 'mean', 'outside_wind_speed_mean': 'mean', 'device_presence_distinct_count': 'mean'})
        df["consumed_heat_kWh"] = df["consumed_heat_kJ"] / 3600
        df = df.drop(columns=["consumed_heat_kJ"])
        
        return df
    
    def get_data_for_training_model(self, left_time_interval=None, right_time_interval=datetime.now(), predicted_column = 'longtime_mean'):
        if left_time_interval is None:
            left_time_interval = self.start_of_data
        print("left_time_interval", left_time_interval)
        print("right_time_interval", right_time_interval)
        queries = self.get_database_queries(left_time_interval=left_time_interval, right_time_interval=right_time_interval)
        df_all = self.get_df_from_queries(queries)
        df_all = self.process_kWh_water_consumption(df_all)

        return self.transform_data_for_ml(df_all, predicted_column=predicted_column)


    def get_data_for_prediction(self, left_time_interval=datetime.now() - timedelta(days=2), right_time_interval=datetime.now(), predicted_column = 'longtime_mean'):
        queries = self.get_database_queries(left_time_interval=left_time_interval, right_time_interval=right_time_interval)
        df_all = self.get_df_from_queries(queries)
        print(df_all)
        df_all = self.process_kWh_water_consumption(df_all)
        df_all.index = df_all.index.tz_localize(None)
        df_all, _= self.transform_data_for_ml(df_all, predicted_column='longtime_mean')

        return df_all
        
        
        return self.transform_data_for_ml(df_all, predicted_column=predicted_column)
        
        # df_predict = pd.DataFrame({'datetime': pd.date_range(left_time_interval, right_time_interval, freq='30min')})
        # df_predict[predicted_column] = 0
        # df_predict['weekday_sin'] = np.sin(2 * np.pi * df_predict['datetime'].dt.weekday / 7)
        # df_predict['weekday_cos'] = np.cos(2 * np.pi * df_predict['datetime'].dt.weekday / 7)
        # df_predict['hour_sin'] = np.sin(2 * np.pi * df_predict['datetime'].dt.hour / 24)
        # df_predict['hour_cos'] = np.cos(2 * np.pi * df_predict['datetime'].dt.hour / 24)
        # df_predict['minute_sin'] = np.sin(2 * np.pi * df_predict['datetime'].dt.minute / 60)
        # df_predict['minute_cos'] = np.cos(2 * np.pi * df_predict['datetime'].dt.minute / 60)
        # # delete column datetime
        # df_predict = df_predict.drop(columns='datetime')
        
        return df_predict
    
    def write_forecast_to_influxdb(self, df, measurement_name):
        df = df.reset_index()
        df['time'] = df['time'].dt.strftime('%Y-%m-%dT%H:%M:%SZ')
        df = df.rename(columns={'time': 'time', 'longtime_mean': 'value'})
        df['measurement'] = measurement_name
        json_body = df.to_dict(orient='records')
        print(json_body)
        self.influxdb_client.write_points(json_body)
        return json_body

    def get_high_tarif_schedule(self):
        print("Getting high tarif schedule")
        left_time_interval = datetime.now() - timedelta(days=14)
        queries = self.get_database_queries(left_time_interval=left_time_interval, right_time_interval=datetime.now())
        df_all = self.get_df_from_queries(queries)
        df_all.index = df_all.index.tz_localize(None)
        data_resampled = df_all.resample('10T').max()
        data_resampled = data_resampled['boiler_relay_status']
        data_resampled = data_resampled.notna().astype(int)
        data_resampled = data_resampled.resample('30T').sum() * 10
        # group by weekday and hour and minute and calculate max
        grouped = data_resampled.groupby([data_resampled.index.weekday, data_resampled.index.hour, data_resampled.index.minute]).max()

        # not null values to 1, null as 0
        grouped = grouped.notna().astype(int)
        df_reset = grouped.reset_index()
        df_reset['time'] = df_reset['level_1'].astype(str) + ':' + df_reset['level_2'].astype(str)
        df_reset['time'] = pd.to_datetime(df_reset['time'], format='%H:%M').dt.time
        df_reset['weekday'] = df_reset['level_0']
        df_reset['unavailable_minutes'] = abs(30 - df_reset['boiler_relay_status'])
        df_reset = df_reset.drop(columns=['level_0', 'level_1', 'level_2', 'boiler_relay_status'])
        

        return df_reset
    
    
    def transform_data_for_ml(self, df, days_from_beginning_ignored = 0, predicted_column = 'longtime_mean'):
        # read pickles from data/pickles

        freq = 0.5
        freq_hour = f"{freq}H"

        df.index = pd.to_datetime(df.index)

        df.loc[:,"weekday"] = df.index.weekday
        df.loc[:,"hour"] = df.index.hour
        df.loc[:,"minute"] = df.index.minute
        
        # delete rows with weekday nan
        df = df.dropna(subset=["weekday"])
        df["consumed_heat_kWh"] = df["consumed_heat_kWh"].fillna(0)


        # fill na in df based on column
        df["temperature"] = df[f'outside_temperature_mean'].fillna(method="ffill")
        df["humidity"] = df[f'outside_humidity_mean'].fillna(method="ffill")
        df["wind_speed"] = df[f'outside_wind_speed_mean'].fillna(method="ffill")
        df["count"] = df[
            "device_presence_distinct_count"
        ].fillna(method="ffill")

        # add to column 'consumed_heat_kWh' 1,25/6 to each row
        df["consumed_heat_kWh"] += 1.25 / (24 // freq)

        window = 6

        df["longtime_mean"] = (
            df["consumed_heat_kWh"]
            .rolling(window=window, min_periods=1, center=True)
            .mean()
        )
        df["longtime_std"] = (
            df["consumed_heat_kWh"].rolling(window=window, min_periods=1).std()
        )
        df["longtime_min"] = (
            df["consumed_heat_kWh"].rolling(window=window, min_periods=1).min()
        )
        df["longtime_max"] = (
            df["consumed_heat_kWh"].rolling(window=window, min_periods=1).max()
        )
        df["longtime_median"] = (
            df["consumed_heat_kWh"].rolling(window=window, min_periods=1).median()
        )
        df["longtime_skew"] = (
            df["consumed_heat_kWh"].rolling(window=window, min_periods=1).skew()
        )
        
        # drop consumed_heat_kWh
        df = df.drop(columns=["consumed_heat_kWh"])


        df["longtime_std"] = df["longtime_std"].fillna(method="ffill")
        df["longtime_std"] = df["longtime_std"].fillna(method="ffill")
        df["longtime_skew"] = df["longtime_mean"].fillna(method="ffill")

        # transform weekday, minute, hour to sin cos
        df["weekday_sin"] = np.sin(2 * df["weekday"] * np.pi / 7)
        df["weekday_cos"] = np.cos(2 * df["weekday"] * np.pi / 7)

        df["hour_sin"] = np.sin(2 * df["hour"] * np.pi / 24)
        df["hour_cos"] = np.cos(2 * df["hour"] * np.pi / 24)

        df["minute_sin"] = np.sin(2 * df["minute"] * np.pi / 60)
        df["minute_cos"] = np.cos(2 * df["minute"] * np.pi / 60)
        
        # df = df[['temperature','humidity','wind_speed','count','weekday_sin','weekday_cos','hour_sin', 'hour_cos', 'longtime_mean', 'minute_sin', 'minute_cos']]
        df = df[[predicted_column,'weekday_sin','weekday_cos','hour_sin', 'hour_cos', 'minute_sin', 'minute_cos']]

        df = df.dropna()
        
        # extract datetimes from index
        datetimes = df.index
        
        df['longtime_mean'] = df['longtime_mean']
        
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
    start_of_data=datetime(2024, 3, 1, 0, 0, 0, 0))
    # df_from_db = data_handler.get_data_for_training_model()
    # dropna - remove rows with NaN
    # result = data_handler.transform_data_for_ml(df_from_db)
    dataHandler = dataHandler.get_data_for_high_tarif_info()

    # data_handler.get_data_for_prediction()
    # data_handler.get_actual_data()
    # data_handler.transform_data_for_ml()
    # data_handler.write_data()
