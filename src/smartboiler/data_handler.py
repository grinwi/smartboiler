import pandas as pd
from influxdb import InfluxDBClient
from influxdb import DataFrameClient
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import numpy as np
import json


data_formankovi_actual_boiler_stats = {
    "boiler_temperature": {
        "sql_query": 'SELECT mean("value") AS "boiler_temperature_{group_by_time_interval}_mean" FROM "smart_home_formankovi"."autogen"."°C" WHERE "entity_id"=\'esphome_boiler_temps_ntc_temperature_a_constant\' GROUP BY time({group_by_time_interval}) FILL(null) ORDER BY DESC LIMIT {6}',
        "measurement": "°C",
    },
}


class DataHandler:
    def __init__(
        self,
        influx_id,
        db_name,
        db_username,
        db_password,
        interested_variables,
        relay_entity_id,
    ):
        self.influx_id = influx_id
        self.db_name = db_name
        self.db_username = db_username
        self.db_password = db_password
        self.interested_variables = interested_variables
        self.group_by_time_interval = "30m"
        self.relay_entity_id = "shelly1pm_34945475a969"
        self.dataframe_client = DataFrameClient(
            host=self.influx_id,
            port=8086,
            username=self.db_username,
            password=self.db_password,
            database=self.db_name,
        )

    def concat_measurement_and_entity_id(self, entity_id, measurement):
        # replace / in name of entity_id with _
        entity_id = entity_id.replace("/", "_")
        measurement = measurement.replace("/", "_")
        return entity_id + "_" + measurement

    def get_database_queries(
        self,
        group_by_time_interval,
        time_interval_left=datetime(2023, 1, 1, 0, 0, 0, 0),
        time_interval_right=datetime.now(),
    ):
        # format datetime to YYYY-MM-DDTHH:MM:SSZ
        time_interval_left = f"'{time_interval_left.strftime('%Y-%m-%dT%H:%M:%SZ')}'"
        time_interval_right = f"'{time_interval_right.strftime('%Y-%m-%dT%H:%M:%SZ')}'"

        queries = {
            "water_flow": {
                "sql_query": f'SELECT mean("value") AS "water_flow_L_per_hour_{group_by_time_interval}_mean" FROM "{self.db_name}"."autogen"."L/min" WHERE time > {time_interval_left} AND time < {time_interval_right} AND "entity_id"=\'esphome_boiler_temps_current_water_usage\' GROUP BY time({group_by_time_interval}) FILL(0)',
                "measurement": "L/min",
            },
            "water_temperature": {
                "sql_query": f'SELECT mean("value") AS "water_temperature_{group_by_time_interval}_mean" FROM "{self.db_name}"."autogen"."°C" WHERE time > {time_interval_left} AND time < {time_interval_right} AND "entity_id"=\'esphome_boiler_temps_ntc_temperature_b_constant\' GROUP BY time({group_by_time_interval}) FILL(null)',
                "measurement": "°C",
            },
            "temperature": {
                "sql_query": f'SELECT mean("temperature") AS "outside_temperature_{group_by_time_interval}_mean" FROM "{self.db_name}"."autogen"."state" WHERE time > {time_interval_left} AND time < {time_interval_right} AND "domain"=\'weather\' AND "entity_id"=\'domov\' GROUP BY time({group_by_time_interval}) FILL(null)',
                "measurement": "state",
            },
            "humidity": {
                "sql_query": f'SELECT mean("humidity") AS "outside_humidity_{group_by_time_interval}_mean" FROM "{self.db_name}"."autogen"."state" WHERE time > {time_interval_left} AND time < {time_interval_right} AND "domain"=\'weather\' AND "entity_id"=\'domov\' GROUP BY time({group_by_time_interval}) FILL(null)',
                "measurement": "state",
            },
            "wind_speed": {
                "sql_query": f'SELECT mean("wind_speed") AS "outside_wind_speed_{group_by_time_interval}_mean" FROM "{self.db_name}"."autogen"."state" WHERE time > {time_interval_left} AND time < {time_interval_right} AND "entity_id"=\'domov\' GROUP BY time({group_by_time_interval}) FILL(null)',
                "measurement": "state",
            },
            "presence": {
                "sql_query": f'SELECT count(distinct("friendly_name_str")) AS "device_presence_{group_by_time_interval}_distinct_count" FROM "{self.db_name}"."autogen"."state" WHERE time > {time_interval_left} AND time < {time_interval_right} AND "domain"=\'device_tracker\' AND "state"=\'home\' GROUP BY time({group_by_time_interval}) FILL(0)',
                "measurement": "state",
            },
            "boiler_water_temperature": {
                "sql_query": f'SELECT mean("value") AS "boiler_water_temperature_{group_by_time_interval}_mean" FROM "{self.db_name}"."autogen"."°C" WHERE time > {time_interval_left} AND time < {time_interval_right} AND "entity_id"=\'{self.relay_entity_id}_temperature\' GROUP BY time({group_by_time_interval}) FILL(null)',
                "measurement": "°C",
            },
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

    def get_data_for_training_model(self, group_by_time_interval="30m"):

        queries = self.get_database_queries(self.group_by_time_interval)
        df_all = self.get_df_from_queries(queries)
        # plot data
        df_all[f"consumed_heat_kJ_{self.group_by_time_interval}"] = (
            df_all[f"water_flow_L_per_hour_{self.group_by_time_interval}_mean"]
            * (df_all[f"water_temperature_{self.group_by_time_interval}_mean"] - 10)
            * 4.186
            / 2
        )  # divided by 2 is because we have {group_by_time_interval} data with mean L/hour
        return df_all

    def get_data_for_prediction(self):
        queries = self.get_database_queries(
            self.group_by_time_interval,
            time_interval_left=datetime.now() - timedelta(hours=2.5),
        )
        df_all = self.get_df_from_queries(queries)
        # plot data
        df_all[f"consumed_heat_kJ_{self.group_by_time_interval}"] = (
            df_all[f"water_flow_L_per_hour_{self.group_by_time_interval}_mean"]
            * (df_all[f"water_temperature_{self.group_by_time_interval}_mean"] - 10)
            * 4.186
            / 2
        )  # divided by 2 is because we have {group_by_time_interval} data with mean L/hour
        print(df_all)
        return df_all

    def get_actual_data(self):
        queries = self.get_database_queries(
            "1m", time_interval_left=datetime.now() - timedelta(hours=1)
        )
        df = self.get_df_from_queries(queries)
        # return last row of dataframe ordered by time
        return df.iloc[-1]

    def transform_data_for_ml(self, df):
        # read pickles from data/pickles

        freq = 0.5
        freq_hour = f"{freq}H"

        lookback = 6  # window used for prediction6
        delay = 1  # predict target freq hours in advance1
        batch_size = 3  # number of samples per batch3

        df.index = pd.to_datetime(df.index)
        # delete first week data
        first_date = df.index[0] + pd.Timedelta(days=7)
        df = df.loc[first_date:]
        df["weekday"] = df.index.weekday
        df["hour"] = df.index.hour
        df["minute"] = df.index.minute
        # delete rows with weekday nan
        df = df.dropna(subset=["weekday"])
        df["consumed_heat_kJ_30m"] = df["consumed_heat_kJ_30m"].fillna(0)

        df = df.resample(freq_hour).sum()

        # fill na in df based on column
        print(df.columns)
        df["temperature"] = df[f'outside_temperature_{self.group_by_time_interval}_mean'].fillna(method="ffill")
        df["humidity"] = df[f'outside_humidity_{self.group_by_time_interval}_mean'].fillna(method="ffill")
        df["wind_speed"] = df[f'outside_wind_speed_{self.group_by_time_interval}_mean'].fillna(method="ffill")
        df["device_presence_30m_distinct_count"] = df[
            "device_presence_30m_distinct_count"
        ].fillna(method="ffill")

        # add to column 'consumed_heat_kJ_30m' 1,25/6 to each row
        df["consumed_heat_kJ_30m"] += 1.25 / (24 // freq)
        # add column longtime mean for consumed_heat_kJ_30m
        # window = int(24*7//freq)
        window = 6

        df["longtime_mean"] = (
            df["consumed_heat_kJ_30m"]
            .rolling(window=window, min_periods=1, center=True)
            .mean()
        )
        df["longtime_std"] = (
            df["consumed_heat_kJ_30m"].rolling(window=window, min_periods=1).std()
        )
        df["longtime_min"] = (
            df["consumed_heat_kJ_30m"].rolling(window=window, min_periods=1).min()
        )
        df["longtime_max"] = (
            df["consumed_heat_kJ_30m"].rolling(window=window, min_periods=1).max()
        )
        df["longtime_median"] = (
            df["consumed_heat_kJ_30m"].rolling(window=window, min_periods=1).median()
        )
        df["longtime_skew"] = (
            df["consumed_heat_kJ_30m"].rolling(window=window, min_periods=1).skew()
        )

        predicted_column = "longtime_mean"

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

        num_of_features = len(df.columns) - 1

        df = df.dropna()
        return df

    def write_data(self, data):
        with open(self.data_file, "w") as file:
            json.dump(data, file)


if __name__ == "__main__":
    # test
    data_handler = DataHandler(
        "localhost",
        "smart_home_formankovi",
        "root",
        "root",
        [
            "water_flow",
            "water_temperature",
            "boiler_temperature",
            "presence",
            "accuweather",
        ],
        "shelly1pm_34945475a969",
    )
    df_from_db = data_handler.get_data_for_training_model()
    # dropna - remove rows with NaN
    result = data_handler.transform_data_for_ml(df_from_db)
    print(result)

    # data_handler.get_data_for_prediction()
    # data_handler.get_actual_data()
    # data_handler.transform_data_for_ml()
    # data_handler.write_data()
