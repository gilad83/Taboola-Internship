from old_files.BatchModel import Data
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, RepeatVector, TimeDistributed
import matplotlib.pyplot as plt
import seaborn as sns

TIME_STEPS = 288*7
ALPHA = 0.9
DATA_POINT_TO_PREDICT = 3


class AutoEncoder:
    def __init__(self):
        self.data = Data()
        # self.data.fetch_data(path="Data\\total_downtime_passover_8_4\\US")
        self.data.fetch_data(path="Data\\Single servers\\AM\\40 cores")


    def create_dataset(self, X, y, time_steps=1):
        Xs, ys = [], []
        for i in range(len(X) - time_steps):
            v = X.iloc[i:(i + time_steps)].values
            Xs.append(v)
            u = y.iloc[i:(i + time_steps)].values
            ys.append(u)
        return np.array(Xs), np.array(ys)

    def split_train_test(self, test_size=0.2):
        df = self.data.raw_dataset
        train_size = int(len(df) * (1 - test_size))
        self.train, self.test = df.iloc[0:train_size], df.iloc[train_size:len(df)]

    def split_X_Y(self, data_point_to_predict=0):
        self.X_train, self.Y_train = self.create_dataset(self.train, self.train, TIME_STEPS)
        self.X_test, self.Y_test = self.create_dataset(self.test, self.test, TIME_STEPS)
        if (data_point_to_predict > 0):
            print(self.X_train)
            self.X_train = self.X_train[slice(None, self.X_train.shape[0] - data_point_to_predict)]
            print(self.X_train)
            self.X_test = self.X_test[slice(None, self.X_test.shape[0] - data_point_to_predict)]
            print(self.Y_train)
            self.Y_train = self.Y_train[slice(data_point_to_predict, None)]
            print(self.Y_train)
            self.Y_test = self.Y_test[slice(data_point_to_predict, None)]

    def normalize(self):
        scaler = MinMaxScaler().fit(self.train)
        self.train = pd.DataFrame(scaler.transform(self.train))
        self.test = pd.DataFrame(scaler.transform(self.test))

    def build_model(self):
        self.model = Sequential()
        self.model.add(LSTM(units=128, input_shape=(self.X_train.shape[1], self.X_train.shape[2])))
        self.model.add(Dropout(rate=0.2))
        self.model.add(RepeatVector(n=self.X_train.shape[1]))
        self.model.add(LSTM(units=128, return_sequences=True))
        self.model.add(Dropout(rate=0.2))
        self.model.add(TimeDistributed(Dense(self.X_train.shape[2])))
        # self.model.add(Activation('relu'))
        self.model.compile(optimizer='adam', loss='mse')

    def fit_model(self, epochs, batch_size):
        self.history = self.model.fit(self.X_train, self.Y_train, epochs=epochs, batch_size=batch_size,
                                      validation_split=0.1,
                                      shuffle=False)

    def plot_train_loss(self):
        plt.figure(3)
        plt.plot(self.history.history['loss'], label='train')
        plt.plot(self.history.history['val_loss'], label='val_loss')
        plt.legend()
        plt.show()

    def calculate_loss_threshold(self):
        plt.figure(4)
        self.Y_train_pred = self.model.predict(self.X_train)
        self.train_loss = np.mean(np.abs(self.Y_train_pred - self.Y_train), axis=1)
        self.train_loss_mean = [np.mean(x) for x in self.train_loss.transpose()]
        self.train_loss_std = [np.std(x) for x in self.train_loss.transpose()]
        sns.distplot(self.train_loss, bins=50)
        plt.show()

    def predict(self):
        self.Y_test_pred = self.model.predict(self.X_test)
        x = np.array(self.Y_test.copy())
        self.test_loss = np.mean(np.abs(self.Y_test_pred - self.Y_test), axis=1)
        self.Y_test = x

    def plot_results(self):
        plt.figure(1)
        num_of_data_points = self.Y_test.shape[0]
        plt.subplots(figsize=(11, 11))
        fet1, = plt.plot(self.Y_test[0:num_of_data_points, 0, 0])
        fet2, = plt.plot(self.Y_test[0:num_of_data_points, 0, 1])
        fet3, = plt.plot(self.Y_test[0:num_of_data_points, 0, 2])
        fet4, = plt.plot(self.Y_test[0:num_of_data_points, 0, 3])
        fet5, = plt.plot(self.Y_test[0:num_of_data_points, 0, 4])
        fet6, = plt.plot(self.Y_test[0:num_of_data_points, 0, 5])
        plt.legend([fet1, fet2, fet3, fet4, fet5, fet6, ], self.data.feacher_names)
        plt.title("Real data")
        plt.show()

        plt.figure(2)
        plt.subplots(figsize=(11, 11))
        fet1, = plt.plot(self.Y_test_pred[0:num_of_data_points, 0, 0])
        fet2, = plt.plot(self.Y_test_pred[0:num_of_data_points, 0, 1])
        fet3, = plt.plot(self.Y_test_pred[0:num_of_data_points, 0, 2])
        fet4, = plt.plot(self.Y_test_pred[0:num_of_data_points, 0, 3])
        fet5, = plt.plot(self.Y_test_pred[0:num_of_data_points, 0, 4])
        fet6, = plt.plot(self.Y_test_pred[0:num_of_data_points, 0, 5])
        plt.legend([fet1, fet2, fet3, fet4, fet5, fet6, ], self.data.feacher_names)
        plt.title("Predicted data")
        plt.show()



    def run(self):
        sns.set()
        self.data.add_features(data_point_to_predict=DATA_POINT_TO_PREDICT,
                               prediction='total_success_action_conversions')
        self.split_train_test()
        self.normalize()
        self.split_X_Y(data_point_to_predict=DATA_POINT_TO_PREDICT)
        self.build_model()
        self.fit_model(20, 128)
        self.plot_train_loss()
        self.calculate_loss_threshold()
        self.predict()
        self.plot_results()


AC = AutoEncoder()
AC.run()