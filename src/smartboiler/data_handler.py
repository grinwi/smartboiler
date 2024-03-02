from pathlib import Path
print('Running' if __name__ == '__main__' else 'Importing', Path(__file__).resolve())
from operator import le
from click import group
from matplotlib.dates import drange
import pandas as pd
from influxdb import DataFrameClient
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import numpy as np
import json


data_zukalovi_actual_boiler_stats = {
    "boiler_temperature": {
        "sql_query": 'SELECT mean("value") AS "mean_value" FROM "smart_home_zukalovi"."autogen"."°C" WHERE time > :dashboardTime: AND time < :upperDashboardTime: AND "entity_id"=\'esphome_web_c771e8_tmp3\' GROUP BY time({group_by_time_interval}) FILL(null) ORDER BY DESC LIMIT {6}',
        "measurement": "°C",
    },
    "is_boiler_on": {
        "sql_query": 'SELECT last("value") AS "mean_value" FROM "smart_home_zukalovi"."autogen"."state" WHERE time > :dashboardTime: AND time < :upperDashboardTime: AND "entity_id"=\'shelly1pm_84cca8b07eae\' GROUP BY time({group_by_time_interval}) FILL(null) ORDER BY DESC LIMIT {6}',
        "measurement": "state",
    },
}



class DataHandler:
    def __init__(
        self,
        influx_id,
        db_name,
        db_username,
        db_password,
        relay_entity_id,
        tmp_boiler_case_entity_id,
        start_of_data=datetime(2023, 1, 1, 0, 0, 0, 0),
    ):
        self.influx_id = influx_id
        self.db_name = db_name
        self.db_username = db_username
        self.db_password = db_password
        self.group_by_time_interval = "30min"
        self.relay_entity_id = relay_entity_id
        self.dataframe_client = DataFrameClient(
            host=self.influx_id,
            port=8086,
            username=self.db_username,
            password=self.db_password,
            database=self.db_name,
        )
        self.start_of_data = start_of_data
        self.last_time_data_update = datetime.now()
        
        
    def get_actual_zuka_boiler_stats(self, group_by_time_interval = "1min"):
        data_zukalovi_actual_boiler_stats = {
        "boiler_temperature": {
            "sql_query": f'SELECT mean("value") AS "mean_value" FROM "smart_home_zukalovi"."autogen"."°C" WHERE "entity_id"=\'{self.tmp_boiler_case_entity_id}\' GROUP BY time({group_by_time_interval}) FILL(null) ORDER BY DESC LIMIT {6}',
            "measurement": "°C",
        },
        "is_boiler_on": {
            "sql_query": f'SELECT last("value") AS "mean_value" FROM "smart_home_zukalovi"."autogen"."state" WHERE "entity_id"=\'{self.relay_entity_id}\' GROUP BY time({group_by_time_interval}) FILL(null) ORDER BY DESC LIMIT {6}',
            "measurement": "state",
        },
}

    def get_database_queries(
        self,
        left_time_interval=None,
        right_time_interval=None,
    ):
        if (left_time_interval == None):
            left_time_interval = self.start_of_data
        if (right_time_interval == None):
            right_time_interval = datetime.now()
        group_by_time_interval = '5s'
        
        # format datetime to YYYY-MM-DDTHH:MM:SSZ
        left_time_interval = f"'{left_time_interval.strftime('%Y-%m-%dT%H:%M:%SZ')}'"
        right_time_interval = f"'{right_time_interval.strftime('%Y-%m-%dT%H:%M:%SZ')}'"

        queries = {
            "water_flow": {
                "sql_query": f'SELECT mean("value") AS "water_flow_L_per_minute_mean" FROM "{self.db_name}"."autogen"."L/min" WHERE time > {left_time_interval} AND time < {right_time_interval} AND "entity_id"=\'esphome_boiler_temps_current_water_usage\' GROUP BY time({group_by_time_interval}) FILL(0)',
                "measurement": "L/min",
            },
            "water_temperature": {
                "sql_query": f'SELECT mean("value") AS "water_temperature_mean" FROM "{self.db_name}"."autogen"."°C" WHERE time > {left_time_interval} AND time < {right_time_interval} AND "entity_id"=\'esphome_boiler_temps_ntc_temperature_b_constant\' GROUP BY time({group_by_time_interval}) FILL(null)',
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
                "sql_query": f'SELECT mean("value") AS "boiler_water_temperature_mean" FROM "{self.db_name}"."autogen"."°C" WHERE time > {left_time_interval} AND time < {right_time_interval} AND "entity_id"=\'{self.relay_entity_id}_temperature\' GROUP BY time({group_by_time_interval}) FILL(null)',
                "measurement": "°C",
            },
            # "boiler_relay_status": {"sql_query": f'SELECT last("value") AS "boiler_relay_status" FROM "{self.db_name}"."autogen"."state" WHERE time > {time_interval_left} AND time < {time_interval_right} AND "entity_id"=\'{self.relay_entity_id}\' GROUP BY time({group_by_time_interval}) FILL(previous)',
            #                        "measurement": "state"},
        }
        return queries

    def get_df_from_queries(self, queries):
        df_all_list = []
        # iterate over key an value in data_formankovi
        for key, value in queries.items():
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
        df[f"consumed_heat_kWh"] += 0.4*7

        
        return df
    
    def get_data_for_training_model(self, left_time_interval=None, right_time_interval=None, predicted_column = 'longtime_mean'):

        queries = self.get_database_queries(left_time_interval=left_time_interval, right_time_interval=right_time_interval)
        df_all = self.get_df_from_queries(queries)
        df_all = self.process_kWh_water_consumption(df_all)

        return self.transform_data_for_ml(df_all, predicted_column=predicted_column)


    def get_data_for_prediction(self, left_time_interval=None, right_time_interval=None, predicted_column = 'longtime_mean'):
        queries = self.get_database_queries(left_time_interval=left_time_interval, right_time_interval=right_time_interval)
        df_all = self.get_df_from_queries(queries)
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


    def get_actual_data(self):
        queries = self.get_actual_zuka_boiler_stats(self.relay_entity_id, 'esphome_web_c771e8_tmp3')
        df = self.get_df_from_queries(queries)
        # return last row of dataframe ordered by time
        return df.iloc[-1]

    def transform_data_for_ml(self, df, days_from_beginning_ignored = 0, predicted_column = 'longtime_mean'):
        # read pickles from data/pickles

        freq = 0.5
        freq_hour = f"{freq}H"

        df.index = pd.to_datetime(df.index)
        # delete first week data
        first_date = df.index[0] + pd.Timedelta(days=days_from_beginning_ignored)
        df = df.loc[first_date:]
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
        
        df['longtime_mean'] = df['longtime_mean'] *  100
        
        return (df.reset_index(drop=True), datetimes)

    def write_data(self, data):
        with open(self.data_file, "w") as file:
            json.dump(data, file)


if __name__ == "__main__":
    # test
    data_handler = DataHandler(
        "localhost",
        "smart_home_zukalovi",
        "root",
        "root",
        "shelly1pm_34945475a969",
        "esphome_web_c771e8_tmp3"
    )
    # df_from_db = data_handler.get_data_for_training_model()
    # dropna - remove rows with NaN
    # result = data_handler.transform_data_for_ml(df_from_db)
    data_handler.get_data_for_prediction()

    # data_handler.get_data_for_prediction()
    # data_handler.get_actual_data()
    # data_handler.transform_data_for_ml()
    # data_handler.write_data()
