import csv
import os, glob
from itertools import zip_longest
from os import path
import pandas as pd
from functools import reduce
from sklearn.preprocessing import MinMaxScaler
from pandas import DataFrame
from pandas import concat
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from matplotlib import pyplot
import matplotlib.pyplot as plt
import argparse





class BatchModel(object):
    data_path_cross_Dc = '../Data/Cross DC'

    def getCsv(self,data_path, country, core_path, metric_path, name_of_metric):
        if data_path == self.data_path_cross_Dc:
            all_files = glob.glob(os.path.join(data_path + country + metric_path, "*.csv"))
        else:
            all_files = glob.glob(os.path.join(data_path + country + core_path + metric_path, "*.csv"))
        all_csv = (pd.read_csv(f, sep=',') for f in all_files)
        new_csv = pd.concat(all_csv, ignore_index=True)
        new_csv.columns = ['dates', name_of_metric]
        return new_csv


    def import_data(self, paths,data_path,country,cores_path):
        csv_data_cores = [self.getCsv(data_path,country,cores_path, path[0], path[1]) for path in paths]
        csv_data_cores = reduce(lambda left, right: pd.merge(left, right, on=['dates'],
                                                             how='outer'), csv_data_cores)
        csv_data_cores = csv_data_cores.dropna()
        data_to_scale_cores = csv_data_cores.drop('dates', 1)
        data_to_scale_cores = data_to_scale_cores.drop('avg_memory', 1)
        data_to_scale_cores = data_to_scale_cores.drop('avg_num_cores', 1)
        # data_to_scale_cores.drop_duplicates(subset=None, inplace=True)
        # data_to_scale_cores.drop(data_to_scale_cores.columns[[0]], axis=1, inplace=True)
        self.dataset = data_to_scale_cores
        values = data_to_scale_cores.values
        return values

    # convert series to supervised learning
    def series_to_supervised(self, data, n_in=1, n_out=1, dropnan=True):
        n_vars = 1 if type(data) is list else data.shape[1]
        df = DataFrame(data)
        cols, names = list(), list()
        # input sequence (t-n, ... t-1)
        for i in range(n_in, 0, -1):
            cols.append(df.shift(i))
            names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
        # forecast sequence (t, t+1, ... t+n)
        for i in range(0, n_out):
            cols.append(df.shift(-i))
            if i == 0:
                names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
            else:
                names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
        # put it all together
        agg = concat(cols, axis=1)
        agg.columns = names
        # drop rows with NaN values
        if dropnan:
            agg.dropna(inplace=True)
        return agg

    # normalize features
    def normalize_features(self, values):
        # ensure all data is float
        values = values.astype('float32')
        # normalize features
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled = scaler.fit_transform(values)
        # frame as supervised learning
        reframed = self.series_to_supervised(scaled, 1, 1)
        values = reframed.values
        return values

    def split_train_test(self, values, trainSize):
        n_train_hours = (int)(len(self.dataset) * trainSize)
        train = values[:n_train_hours, :]
        test = values[n_train_hours:, :]
        # split into input and outputs
        train_X, train_y = train[:, :-1], train[:, -1]
        test_X, test_y = test[:, :-1], test[:, -1]
        # reshape input to be 3D [samples, timesteps, features]
        train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
        test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))

        self.train_X = train_X
        self.train_y = train_y
        self.test_X = test_X
        self.test_y = test_y

    def create_model(self):
        # design network
        self.model = Sequential()
        self.model.add(LSTM(50, input_shape=(self.train_X.shape[1], self.train_X.shape[2])))
        self.model.add(Dense(1))
        self.model.compile(loss='mean_squared_error', optimizer='adam')
        # fit network
        train_X = self.train_X[:len(self.train_X) - 0, :]
        train_y = self.train_y[0:]

        test_X = self.test_X[:len(self.test_X) - 0, :]
        test_y = self.test_y[0:]
        self.history = self.model.fit(train_X, train_y, epochs=50, batch_size=72, validation_data=(test_X, test_y),
                                      verbose=2, shuffle=False)

    def plot_history(self):
        # plot history
        pyplot.plot(self.history.history['loss'], label='train')
        pyplot.plot(self.history.history['val_loss'], label='test')
        pyplot.legend()
        pyplot.show()

    def make_a_prediction(self):
        # Predict
        Predict = self.model.predict(self.test_X, verbose=1)
        print(Predict)
        # Plot
        plt.figure(2)
        plt.scatter(self.test_y, Predict)
        plt.show(block=False)

        with open('../resultsBatch0.csv', 'w') as file:
            writer = csv.writer(file)
            d = [Predict, (map(lambda x: [x], self.test_y))]
            export_data = zip_longest(*d, fillvalue='')
            writer.writerows(export_data)

        plt.figure(3)
        Test, = plt.plot(self.test_y)
        Predict, = plt.plot(Predict)
        plt.legend([Predict, Test], ["Predicted Data", "Real Data"])
        plt.show()



# single server
avg_cpu_load = '/avg_cpu_load'
avg_heap = '/avg_heap'
avg_memory = '/avg_memory'
avg_num_cores = '/avg_num_cores'
cpu_user_util = '/cpu_user_util'
max_cpu_load = '/max_cpu_load'
max_heap = '/max_heap'
p99_response_time = '/p99_response_time'
reco_rate = '/reco_rate'
load_score_meter = '/load_score_meter'

# cross dc
avg_cpu_load_DC = '/avg(node_load15{hostname=~_^water._}) by (domain)'
avg_heap_DC = '/avg_heap'
avg_memory_Dc = '/avg(avg(node_memory_MemTotal_bytes{hostname=~_^water._})) by (hostname)'
avg_num_cores_Dc = '/avg(count (node_cpu_seconds_total{mode=_idle_,hostname=~_^water._,job=~_node_exporter_}) by (hostname))'
max_cpu_load_Dc = '/max(node_load15{hostname=~_^water._}) by (domain)'
max_heap_Dc = '/max_heap'
p99_response_time_Dc = '/trc_requests_timer_p99_weighted_dc'
reco_rate_Dc = '/recommendation_requests_5m_rate_dc'

paths_cross_dc = [[avg_cpu_load_DC, 'avg_cpu_load'], [avg_heap_DC, 'avg_heap'], [avg_memory_Dc, 'avg_memory']
    , [avg_num_cores_Dc, 'avg_num_cores'], [max_cpu_load_Dc, 'cpu_user_util'],
                  [max_cpu_load_Dc, 'max_cpu_load'], [max_heap_Dc, 'max_heap']
    , [p99_response_time_Dc, 'p99_response_time'], [reco_rate_Dc, 'reco_rate']]

paths_server = [[avg_cpu_load, 'avg_cpu_load'], [avg_heap, 'avg_heap'], [avg_memory, 'avg_memory']
    , [avg_num_cores, 'avg_num_cores'], [cpu_user_util, 'cpu_user_util'],
                [max_cpu_load, 'max_cpu_load'], [max_heap, 'max_heap']
    , [p99_response_time, 'p99_response_time'], [reco_rate, 'reco_rate'], [load_score_meter, 'load_score_meter']]

# Data/Single servers/AM/40 cores 187.35 GB
data_path_servers = '../Data/Single servers'
data_path_cross_Dc = '../Data/Cross DC'
cores_32_path = '32 cores 125.6 GB'
cores_40_path = '40 cores 187.35 GB'
cores_48_path = '48 cores 187.19 GB'
cores_72_path = '72 cores 251.63GB'
cores_40_path_copy = '40 cores 187.35 GB - Copy'
country_AM = '/AM/'
country_IL = '/IL/'
country_LA = '/LA/'

# parser = argparse.ArgumentParser()
# parser.add_argument("path", help="Data path")
# parser.add_argument("train_size", type=float, help="Train size")
# args = parser.parse_args()

BM = BatchModel()
# dataPath = args.path
values = BM.import_data(paths_server,data_path_servers,country_AM,cores_48_path)
values = BM.normalize_features(values)
BM.split_train_test(values, 0.7)
BM.create_model()
BM.plot_history()
BM.make_a_prediction()