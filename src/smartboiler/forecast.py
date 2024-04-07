from distutils.command import build
from pathlib import Path

print("Running" if __name__ == "__main__" else "Importing", Path(__file__).resolve())
from datetime import timedelta, datetime
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ReduceLROnPlateau

import tensorflow as tf
from tensorflow.keras.optimizers import SGD
from keras.layers import LSTM
from keras.layers import Dropout
from keras.models import Model
from keras.layers import Input, Dense
from keras.models import load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint
import keras.backend as K
import numpy as np
import pandas as pd
from scipy import stats
from pickle import load
from pickle import dump


from smartboiler.data_handler import DataHandler


class Forecast:
    def __init__(
        self,
        dataHandler: DataHandler,
        start_of_data: datetime,
        model_path=None,
        scaler_path=None,
        predicted_columns=None,
    ):
        self.batch_size = 16
        self.lookback = 32

        self.delay = 1
        self.predicted_columns = predicted_columns
        self.dataHandler = dataHandler
        self.scaler = RobustScaler()
        self.start_of_data = start_of_data

        self.model_path = model_path
        self.scaler_path = scaler_path

    def train_model(
        self,
        begin_of_training=None,
        end_of_training=None,
        df_training_data=None,
    ):
        if df_training_data is None:
            if begin_of_training is None:
                begin_of_training = self.start_of_data
            if end_of_training is None:
                end_of_training = datetime.now()
            print("begin of training: ", begin_of_training)
            print("end of training : ", end_of_training)
            df_training_data, _ = self.dataHandler.get_data_for_training_model(
                left_time_interval=begin_of_training,
                right_time_interval=end_of_training,
                predicted_columns=self.predicted_columns,
                dropna=False,
            )

        self.num_of_features = len(df_training_data.columns) - 1
        self.df_train_norm = df_training_data.copy()
        self.df_train_norm[df_training_data.columns] = self.scaler.fit_transform(
            df_training_data
        )
        dump(self.scaler, open(self.scaler_path, "wb"))

        self.train_gen = self.mul_generator(
            dataframe=self.df_train_norm,
            target_names=self.predicted_columns,
            lookback=self.lookback,
            delay=self.delay,
            min_index=0,
            max_index=int(df_training_data.shape[0] * 0.8),
            step=1,
            shuffle=True,
            batch_size=self.batch_size,
        )

        self.valid_gen = self.mul_generator(
            dataframe=self.df_train_norm,
            target_names=self.predicted_columns,
            lookback=self.lookback,
            delay=self.delay,
            min_index=int(df_training_data.shape[0] * 0.8),
            max_index=None,
            step=1,
            shuffle=False,
            batch_size=self.batch_size,
        )

        self.val_steps = int(
            (self.df_train_norm.shape[0] * 0.1 - self.lookback) // self.batch_size
        )
        # This is how many steps to draw from `train_gen`
        # in order to see the whole train set:
        self.train_steps = int(
            (self.df_train_norm.shape[0] * 0.9 - self.lookback) // self.batch_size
        )

        callbacks = [
            EarlyStopping(
                monitor="loss",
                min_delta=0,
                patience=10,
                verbose=2,
                mode="auto",
                restore_best_weights=True,
            ),
            ModelCheckpoint(
                verbose=1,
                filepath=self.model_path,
                save_best_only=True,
                save_weights_only=True,
            ),
        ]

        print("Start training")
        history = self.model.fit(
            self.train_gen,
            steps_per_epoch=self.train_steps,
            epochs=100,
            shuffle=False,
            validation_data=self.valid_gen,
            validation_steps=self.val_steps,
            callbacks=callbacks,
            verbose=2,
        )

        self.model.save(self.model_path)
        print("End training")

    def load_model(
        self,
        left_time_interval=datetime.now() - timedelta(days=4),
        right_time_interval=datetime.now(),
    ):

        self.scaler = load(open(self.scaler_path, "rb"))
        self.model.load_weights(self.model_path)

    def generator(
        self,
        dataframe,
        target_name,
        lookback,
        delay,
        min_index,
        max_index,
        shuffle=False,
        batch_size=128,
        step=6,
    ):
        data_without_target = dataframe.copy()
        data_without_target = data_without_target.drop(columns=[target_name]).values
        data_without_target = data_without_target.astype(np.float32)

        data = dataframe.values
        data = data.astype(np.float32)
        target_indx = dataframe.columns.get_loc(target_name)

        if max_index is None:
            max_index = len(data) - delay - 1
        i = min_index + lookback
        while 1:
            if shuffle:
                rows = np.random.randint(
                    min_index + lookback, max_index, size=batch_size
                )
            else:
                if i + batch_size >= max_index:
                    i = min_index + lookback
                rows = np.arange(i, min(i + batch_size, max_index))
                i += len(rows)

            samples = np.zeros(
                (len(rows), lookback // step, data_without_target.shape[-1])
            )
            targets = np.zeros((len(rows),))

            for j, row in enumerate(rows):
                indices = range(rows[j] - lookback, rows[j], step)
                # samples without column with target
                samples[j] = data_without_target[indices]
                targets[j] = data[rows[j] + delay][target_indx]
            yield samples, targets

    def r2_keras(self, y_true, y_pred):
        """Coefficient of Determination"""
        SS_res = K.sum(K.square(y_true - y_pred))
        SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
        return 1 - SS_res / (SS_tot + K.epsilon())

    def build_model(self):

        model = Sequential()
        model.add(Input(shape=(None, 14)))       
     
        # Add LSTM layer
        model.add(LSTM(100))
        model.add(Dense(1))

        self.model = model
        self.model.compile(loss="mae", optimizer="adam")
        return model



    def add_empty_row(self, df, date_time, predicted_value):

        last_row_values = df.iloc[-1].values
        prev_week_values = df.iloc[-24 * 7].values

        new_row_df = pd.DataFrame(
            columns=df.columns,
            data=[
                [
                    predicted_value,# longtime_mean
                    prev_week_values[1], # 3 week skew
                    prev_week_values[2], # 3 weeak std
                    last_row_values[3], # distance_from_home
                    last_row_values[4], # speed_towards_home
                    last_row_values[5], # count
                    last_row_values[6], # heading to home sin
                    last_row_values[7], # heading to home cos
                    last_row_values[8], # temperature
                    last_row_values[9], # humidity
                    last_row_values[10], # wind_speed
                    
                    np.sin(2 * np.pi * date_time.weekday() / 7),
                    np.cos(2 * np.pi * date_time.weekday() / 7),
                    np.sin(2 * np.pi * date_time.hour / 24),
                    np.cos(2 * np.pi * date_time.hour / 24),

                ]
            ],
        )
        df = pd.concat([df, new_row_df], ignore_index=True)
        df = df.reset_index(drop=True)

        return df

    def mul_generator(
        self,
        dataframe,
        target_names,
        lookback,
        delay,
        min_index,
        max_index,
        shuffle=False,
        batch_size=128,
        step=6,
    ):
        data = dataframe.values
        data = data.astype(np.float32)

        data_without_targets = dataframe.copy()
        data_without_targets = data_without_targets.drop(columns="longtime_mean")
        data_without_targets = data_without_targets.values
        data_without_targets = data_without_targets.astype(np.float32)

        # Get the column indices for the target names
        target_indices = [
            dataframe.columns.get_loc(target_name) for target_name in target_names
        ]

        if max_index is None:
            max_index = len(data) - delay - 1
        i = min_index + lookback
        while 1:
            if shuffle:
                rows = np.random.randint(
                    min_index + lookback, max_index, size=batch_size
                )
            else:
                if i + batch_size >= max_index:
                    i = min_index + lookback
                rows = np.arange(i, min(i + batch_size, max_index))
                i += len(rows)

            samples = np.zeros(
                (len(rows), lookback // step, data_without_targets.shape[-1])
            )

            # Modify targets array to accommodate multiple target columns
            targets = np.zeros((len(rows), len(target_indices)))

            for j, row in enumerate(rows):
                indices = range(rows[j] - lookback, rows[j], step)
                samples[j] = data_without_targets[indices]

                # Assign values for each target column
                for k, target_indx in enumerate(target_indices):
                    targets[j][k] = data[rows[j] + delay][target_indx]

            yield samples, targets

    def get_forecast_next_steps(
        self, left_time_interval=None, right_time_interval=None
    ):
        # Define the indices for the different predictions and truths
        if left_time_interval is None:
            left_time_interval = datetime.now() - timedelta(days=30)
        if right_time_interval is None:
            right_time_interval = datetime.now()

        df_all, datetimes = self.dataHandler.get_data_for_prediction(
            left_time_interval=left_time_interval,
            right_time_interval=right_time_interval,
        )

        num_targets = len(self.predicted_columns)
        len_columns = len(df_all.columns)
        forecast_future = pd.DataFrame()

        # print last row of df all

        forecast_future = []
            
        current_forecast_begin_date = right_time_interval + timedelta(hours=1)
        
        df_all = self.add_empty_row(df_all, current_forecast_begin_date, 0)
        current_forecast_begin_date += timedelta(hours=1)

        # prediction for next 6 hours
        for i in range(0, 6):

            df_test_norm = df_all.reset_index(drop=True).copy()

            # df_test_norm = df_test_zuka.copy()
            df_test_norm[df_test_norm.columns] = self.scaler.transform(df_test_norm)
            
            
            
            df_test_norm = df_test_norm[-self.lookback*4:]
            test_gen = self.mul_generator(
                dataframe=df_test_norm,
                target_names=self.predicted_columns,
                lookback=self.lookback,
                delay=self.delay,
                min_index=0,
                max_index=None,
                step=1,
                shuffle=False,
                batch_size=self.batch_size,
            )
            
            last_batch = next(test_gen)

            # Step 3: Extract the last batch of features (X_batch) and target values (y_truth_batch)
            (X_batch, y_truth) = last_batch
            
            

            # Step 4: Make predictions with your model on the last batch
            num_targets = len(self.predicted_columns)
            len_columns = len(df_test_norm.columns)
            num_features = len_columns - num_targets
            
            y_truth_concat = np.concatenate(
                (y_truth, np.zeros((y_truth.shape[0], num_features))), axis=1
            )
            y_truth_concat = self.scaler.inverse_transform(y_truth_concat)
            
            y_truth_inv = y_truth_concat[-1, 0]
            
            y_pred = self.model.predict(X_batch, verbose=0)
            y_pred_inv = np.concatenate(
                (y_pred, np.zeros((y_pred.shape[0], num_features))), axis=1
            )

            y_pred_inv = self.scaler.inverse_transform(y_pred_inv)
            # get last predicted value
            y_pred_inv = y_pred_inv[-1, 0]
            # y_pred_inv[0] is min 0
            if y_pred_inv < 0:
                y_pred_inv = 0
                
            if (i == 0):
                print("y_pred_inv: ", y_pred_inv)
                print("y_truth_inv: ", y_truth_inv)

            # set last longtimemean value
            df_all.iloc[-1, df_all.columns.get_loc("longtime_mean")] = y_pred_inv

            # df_all = df_all[1:]

            forecast_future.append(y_pred_inv)
            
            
            df_all = self.add_empty_row(df_all, current_forecast_begin_date, 0)
            current_forecast_begin_date += timedelta(hours=1)
            
        # create a dataframe with forecast and datetime as index
        self.dataHandler.write_forecast_to_influxdb(
            forecast_future, "prediction_longtime_mean"
        )
        return forecast_future


if __name__ == "__main__":
    dataHandler = DataHandler(
        "localhost",
        "smart_home_formankovi",
        "root",
        "root",
        "shelly1pm_34945475a969",
        "esphome_web_c771e8_tmp3",
    )
    forecast = Forecast(dataHandler)
    forecast.train_model()
    forecast.build_model()
    forecast.get_forecast_next_steps(30)
