# Created as a part of Master's Thesis "Using machine learning methods to save energy in a smart home"
# Faculty of Information Technology, Brno University of Technology, 2024
# Author: Adam Grünwald
#
# This module is used for training the model for prediction and creating predictions.

from typing import Optional
from datetime import timedelta, datetime
from sklearn.preprocessing import RobustScaler
from keras.models import Sequential

from keras.layers import LSTM
from keras.layers import Input, Dense, Dropout, BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

import keras.backend as K
import numpy as np
import pandas as pd
from pickle import load
from pickle import dump
# import Adam
from keras.optimizers import Adam

from smartboiler.data_handler import DataHandler


class Forecast:
    """Class for training model for prediction and creating predictions"""

    def __init__(
        self,
        dataHandler: DataHandler,
        start_of_data: datetime,
        model_path: Optional[str] = None,
        scaler_path: Optional[datetime] = None,
        predicted_columns: Optional[list] = None,
    ):
        """Initialize the class of the forecast.

        Args:
            dataHandler (DataHandler): Instance of the DataHandler class
            start_of_data (datetime): Datetime of the start of the data
            model_path (Optional[str], optional): Path of the model. Defaults to None.
            scaler_path (Optional[datetime], optional): Path of the scaler. Defaults to None.
            predicted_columns (Optional[list], optional): List of columns for prediction. Defaults to None.
        """
        self.batch_size = 128
        self.lookback = 32
        self.delay = 1
        self.step = 1

        self.num_of_features = 18

        self.predicted_columns = predicted_columns
        self.dataHandler = dataHandler
        self.scaler = RobustScaler()
        self.start_of_data = start_of_data

        self.model_path = model_path
        self.scaler_path = scaler_path
        self.quantiles = [0.1, 0.3, 0.5, 0.7, 0.9]

    def train_model(
        self,
        begin_of_training: Optional[datetime] = None,
        end_of_training: Optional[datetime] = None,
        df_training_data: Optional[pd.DataFrame] = None,
    ) -> None:
        """Method for training the model

        Args:
            begin_of_training (_type_, optional): Datetime of beggining of the data for training. Defaults to None.
            end_of_training (_type_, optional): Datetime of end of the data used for training. Defaults to None.
            df_training_data (_type_, optional): Dataframe with data for training.
                                                If not None, this data will be used for training. Defaults to None.
        """

        # if the data for training is not provided, get the data from the dataHandler
        if df_training_data is None:
            if begin_of_training is None:
                begin_of_training = self.start_of_data
            if end_of_training is None:
                end_of_training = datetime.now()
            df_training_data = self.dataHandler.get_data_for_prediction(
                left_time_interval=begin_of_training,
                right_time_interval=end_of_training,
            )
        # get the number of features
        self.num_of_features = len(df_training_data.columns) - 1

        # fit the scaler
        self.df_train_norm = df_training_data.copy()
        self.df_train_norm[df_training_data.columns] = self.scaler.fit_transform(
            df_training_data
        )
        # save the scaler
        dump(self.scaler, open(self.scaler_path, "wb"))

        # create a train validation generator
        self.train_gen = self.mul_generator(
            dataframe=self.df_train_norm,
            target_names=self.predicted_columns,
            min_index=0,
            max_index=int(df_training_data.shape[0] * 0.8),
            shuffle=True,
        )

        self.valid_gen = self.mul_generator(
            dataframe=self.df_train_norm,
            target_names=self.predicted_columns,
            min_index=int(df_training_data.shape[0] * 0.8),
            max_index=None,
            shuffle=False,
        )

        # devide validity and train steps
        self.val_steps = int(
            (self.df_train_norm.shape[0] * 0.1 - self.lookback) // self.batch_size
        )
        self.train_steps = int(
            (self.df_train_norm.shape[0] * 0.9 - self.lookback) // self.batch_size
        )

        # create a callbacks for training of the model

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

        # fit the model
        self.history = self.model.fit(
            self.train_gen,
            steps_per_epoch=self.train_steps,
            epochs=100,
            shuffle=False,
            validation_data=self.valid_gen,
            validation_steps=self.val_steps,
            callbacks=callbacks,
            verbose=2,
        )

        # save the weights of the model
        self.model.save_weights(self.model_path, overwrite=True)

    def load_model(
        self,
    ) -> None:
        """Load model and scaler from the files"""
        self.scaler = load(open(self.scaler_path, "rb"))
        self.model.load_weights(self.model_path, skip_mismatch=False)

    def quantile_loss(self, q, y_true, y_pred) -> float:
        """Quantile loss function used for training the model

        Args:
            q (float): quantil
            y_true (float): value of the true data
            y_pred (float): value of the predicted data

        Returns:
            float: the loss
        """
        e = y_true - y_pred
        return K.mean(K.maximum(q * e, (q - 1) * e), axis=-1)

    def build_model(self) -> None:
        """Method for building the model"""

        # Use the Sequential with LSTM layer with 100 units and Dense layer with 1 unit
        model = Sequential()
        model.add(Input(shape=(None, self.num_of_features)))
        model.add(LSTM(100, return_sequences=False))
        # model.add(BatchNormalization())
        # model.add(LSTM(50))
        # model.add(Dropout(0.5))
        model.add(Dense(1))

        # compile the model with the quantile loss and adam optimizer
        self.model = model
        self.model.compile(
          loss=[
                lambda y_true, y_pred: self.quantile_loss(q, y_true, y_pred)
                for q in self.quantiles
            ],

            optimizer=Adam(learning_rate=0.0001),
        )

    def add_empty_row(
        self, df: pd.DataFrame, date_time: datetime, predicted_value: float
    ) -> pd.DataFrame:
        """Methot adding an empty row to the dataframe

        Args:
            df (pd.DataFrame): Dataframe
            date_time (datetime): Datetime of the new row
            predicted_value (float): Predicted value from previous step

        Returns:
            pd.DataFrame: Dataframe with the new row
        """
        # get the last row values
        last_row_values = df.iloc[-1].values
        # get values from previous week
        prev_week_values = df.iloc[-24 * 7].values

        new_row_df = pd.DataFrame(
            columns=df.columns,
            data=[
                [
                    predicted_value,  # longtime_mean
                    prev_week_values[1],  # 3 week skew
                    prev_week_values[2],  # 3 weeak std
                    last_row_values[3],  # distance_from_home
                    last_row_values[4],  # speed_towards_home
                    last_row_values[5],  # distance_from_home 2
                    last_row_values[6],  # speed_towards_home 2
                    last_row_values[7],  # count
                    last_row_values[8],  # heading to home sin
                    last_row_values[9],  # heading to home cos
                    last_row_values[10],  # heading to home sin 2
                    last_row_values[11],  # heading to home cos 2
                    last_row_values[12],  # temperature
                    last_row_values[13],  # humidity
                    last_row_values[14],  # wind_speed
                    np.sin(2 * np.pi * date_time.weekday() / 7),
                    np.cos(2 * np.pi * date_time.weekday() / 7),
                    np.sin(2 * np.pi * date_time.hour / 24),
                    np.cos(2 * np.pi * date_time.hour / 24),
                ]
            ],
        )

        # concat the new row to the dataframe
        df = pd.concat([df, new_row_df], ignore_index=True)
        df = df.reset_index(drop=True)

        return df

    def mul_generator(
        self,
        dataframe: pd.DataFrame,
        target_names: list,
        min_index: int,
        max_index: int,
        shuffle: Optional[bool] = False,
        batch_size=None,
    ):
        """
        Method to create a generator for the model.
        Concept of the generator is taken from the time-series-h2o-automl-example repository.
        https://github.com/SeanPLeary/time-series-h2o-automl-example by Leary, Sean P.


        Args:
            dataframe (pd.DataFrame): Dataframe with the data
            target_names (list): Names of the target values
            min_index (int): Min index of the data
            max_index (int): Max index of the data
            shuffle (Optional[bool], optional): Choose if shuffle the data. Defaults to False.

        Yields:
            _type_: The data for the model
        """

        if batch_size is None:
            batch_size = self.batch_size
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
            max_index = len(data) - self.delay - 1
        i = min_index + self.lookback
        while 1:
            if shuffle:
                rows = np.random.randint(
                    min_index + self.lookback, max_index, size=batch_size
                )
            else:
                if i + batch_size >= max_index:
                    i = min_index + self.lookback
                rows = np.arange(i, min(i + batch_size, max_index))
                i += len(rows)

            samples = np.zeros(
                (len(rows), self.lookback // self.step, data_without_targets.shape[-1])
            )

            # Modify targets array to accommodate multiple target columns
            targets = np.zeros((len(rows), len(target_indices)))

            for j, row in enumerate(rows):
                indices = range(rows[j] - self.lookback, rows[j], self.step)
                samples[j] = data_without_targets[indices]

                # Assign values for each target column
                for k, target_indx in enumerate(target_indices):
                    targets[j][k] = data[rows[j] + self.delay][target_indx]

            yield samples, targets

    def get_forecast_next_steps(
        self,
        left_time_interval: Optional[datetime] = None,
        right_time_interval: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """Method for getting the forecast of the consumed heat prediction for the next steps for the next 6 hours.

        Args:
            left_time_interval (Optional[datetime], optional): Left time datetime of interval. Defaults to None.
            right_time_interval (Optional[datetime], optional): Right time datetime of interval. Defaults to None.

        Returns:
            pd.DataFrame: Dataframe with consumed heat prediction for next 6 hours.
        """
        if left_time_interval is None:
            left_time_interval = datetime.now() - timedelta(days=30)
        if right_time_interval is None:
            right_time_interval = datetime.now()

        # get data for creatig a prediction
        df_all = self.dataHandler.get_data_for_prediction(
            left_time_interval=left_time_interval,
            right_time_interval=right_time_interval,
        )

        number_of_targets = len(self.predicted_columns)
        number_of_columns = len(df_all.columns)
        number_of_features = number_of_columns - number_of_targets

        # dataframe with forecast
        forecast_future = pd.DataFrame()

        current_forecast_begin_date = right_time_interval + timedelta(hours=1)

        # add an empty row to the dataframe
        df_all = self.add_empty_row(df_all, current_forecast_begin_date, 0)
        current_forecast_begin_date += timedelta(hours=1)

        # prediction for next 6 hours
        for i in range(0, 6):

            df_test_norm = df_all.reset_index(drop=True).copy()

            # df_test_norm = df_test_zuka.copy()
            df_test_norm[df_test_norm.columns] = self.scaler.transform(df_test_norm)

            # get data for the last lookback * 4 hours
            df_test_norm = df_test_norm[-self.lookback * 4 :]
            # create a generator for the model
            test_gen = self.mul_generator(
                dataframe=df_test_norm,
                target_names=self.predicted_columns,
                batch_size=self.batch_size,
                min_index=0,
                max_index=None,
                shuffle=False,
            )

            last_batch = next(test_gen)

            (X_batch, y_truth) = last_batch

            # do the prediction of next step
            y_pred = self.model.predict(X_batch, verbose=0)

            # inverse transform the prediction
            y_pred_inv = np.concatenate(
                (y_pred, np.zeros((y_pred.shape[0], number_of_features))), axis=1
            )
            y_pred_inv = self.scaler.inverse_transform(y_pred_inv)

            # get last predicted value
            y_pred_inv = y_pred_inv[-1, 0]

            # y_pred_inv[0] is min 0
            if y_pred_inv < 0:
                y_pred_inv = 0

            # set last longtime_mean value
            df_all.iloc[-1, df_all.columns.get_loc("longtime_mean")] = y_pred_inv

            # add the prediction to the forecast
            forecast_future = pd.concat(
                [
                    forecast_future,
                    df_all.iloc[[-1], df_all.columns.get_loc("longtime_mean")],
                ],
                axis=0,
            )
            forecast_future = forecast_future.reset_index(drop=True)

            # add an empty row to the dataframe
            df_all = self.add_empty_row(df_all, current_forecast_begin_date, 0)
            current_forecast_begin_date += timedelta(hours=1)

        # create a dataframe with forecast and datetime as index
        self.dataHandler.write_forecast_to_influxdb(forecast_future)
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
