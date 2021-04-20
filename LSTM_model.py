import os, glob
from functools import reduce
from os import path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MultipleLocator
from matplotlib.ticker import OldAutoLocator
import matplotlib.ticker as ticker
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras import metrics

import plotly.graph_objects as go
import numpy as np

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
data_path_servers = 'Data/Single servers'
data_path_cross_Dc = 'Data/Cross DC'
cores_32_path = '32 cores 125.6 GB'
cores_40_path = '40 cores 187.35 GB'
cores_48_path = '48 cores 187.19 GB'
cores_72_path = '72 cores 251.63GB'
cores_40_path_copy = '40 cores 187.35 GB - Copy'
country_AM = '/AM/'
country_IL = '/IL/'
country_LA = '/LA/'


def getCsv(data_path, country, core_path, metric_path, name_of_metric):
    if data_path == data_path_cross_Dc:
        all_files = glob.glob(os.path.join(data_path + country + metric_path, "*.csv"))
    else:
        all_files = glob.glob(os.path.join(data_path + country + core_path + metric_path, "*.csv"))
    all_csv = (pd.read_csv(f, sep=',') for f in all_files)
    new_csv = pd.concat(all_csv, ignore_index=True)
    new_csv.columns = ['dates', name_of_metric]
    return new_csv


def getDataSet(paths, data_path, country, cores_path, figure_num):
    csv_data_cores = [getCsv(data_path, country, cores_path, path[0], path[1]) for path in paths]
    csv_data_cores = reduce(lambda left, right: pd.merge(left, right, on=['dates'],
                                                         how='outer'), csv_data_cores)
    csv_data_cores = csv_data_cores.drop('avg_memory', 1)
    csv_data_cores = csv_data_cores.drop('avg_num_cores', 1)
    csv_data_cores = csv_data_cores.dropna()
    # drop date
    # csv_data_cores = add_isWeekend_feature(csv_data_cores)
    # csv_data_cores = add_trend(csv_data_cores)
    data_to_scale_cores = csv_data_cores.drop('dates', 1)
    return data_to_scale_cores, csv_data_cores


def scale(data_to_scale, predicted_metric):
    sc = MinMaxScaler()
    sc.fit(data_to_scale)
    data_to_scale = sc.fit_transform(data_to_scale)
    predicted_metric_reshape = predicted_metric.values.reshape(-1, 1)
    predicted_metric = sc.fit_transform(predicted_metric_reshape)
    return data_to_scale, predicted_metric


def add_trend(dataset):
    feature_names = ["p99_response_time"]
    i = 0
    for feature in feature_names:
        i += 1
        x = dataset[feature]
        trend = [b - a for a, b in zip(x[::1], x[1::1])]
        trend.append(0)
        dataset["trend_" + feature] = trend
    return dataset


def add_isWeekend_feature(dataset):
    dataset['is_weekend'] = dataset['dates'].str.split(' ', expand=True)[0]
    dataset['is_weekend'] = pd.to_datetime(dataset['is_weekend'], format='%Y-%m-%d')
    dataset['is_weekend'] = dataset['is_weekend'].dt.dayofweek
    is_weekend = dataset['is_weekend'].apply(lambda x: 1 if x >= 5.0 else 0)
    dataset['is_weekend'] = is_weekend
    return dataset


def model_settings(number_of_nodes, X_train, Y_train):
    model = Sequential()
    model.add(LSTM(number_of_nodes, activation='relu', input_shape=(1, X_train.shape[2]),
                   recurrent_activation='hard_sigmoid'))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=[metrics.mae])
    model.fit(X_train, Y_train, epochs=1, batch_size=72, verbose=2)
    return model


# get_data with no date, and all data in csv
data_to_scale_40_cores, csv_data_40_cores = getDataSet(paths_server, data_path_servers, country_AM, cores_40_path, 2)
# save predicted metric
cpu_user_util_csv = csv_data_40_cores['cpu_user_util']
save_dates = csv_data_40_cores['dates']

# data_to_scale_40_cores = data_to_scale_40_cores.drop('cpu_user_util', 1)  # no dates and no cpu util
# scale data
data_40_cores_scaled, cpu_user_util_csv_reshape_scaled = scale(data_to_scale_40_cores, cpu_user_util_csv)

# split into test & train
X_train, X_test, Y_train, Y_test = train_test_split(data_40_cores_scaled, cpu_user_util_csv_reshape_scaled,
                                                    test_size=0.25)

# shape test & train
timesteps_to_the_future = 1
X_train = X_train.reshape((X_train.shape[0], timesteps_to_the_future, X_train.shape[1]))
X_test = X_test.reshape((X_test.shape[0], timesteps_to_the_future, X_test.shape[1]))
# create the lstm model
number_of_nodes = 50
lstm_model = model_settings(number_of_nodes, X_train, Y_train)
predict = lstm_model.predict(X_test)

# plt.figure(2)
# plt.scatter(Y_test, predict)
# plt.show(block=False)


# Real, = plt.plot(save_dates.values[:Y_test.shape[0]],Y_test)
# Predict, = plt.plot(save_dates.values[:Y_test.shape[0]],predict)

fig = go.Figure([
    go.Scatter(
        name='Real',
        x=save_dates.values[Y_test.shape[0]:].reshape(-1),
        y=Y_test.reshape(-1),
        mode='markers+lines',
        marker=dict(color='red', size=1),
        showlegend=True
    ),
    go.Scatter(
        name='Predict',
        x=save_dates.values[Y_test.shape[0]:].reshape(-1),
        y=predict.reshape(-1),
        mode='lines',
        marker=dict(color="#444"),
        line=dict(width=1),
        showlegend=True
    )])
fig.show()
# Real, = plt.plot(Y_test)
# Predict, = plt.plot(predict)
# plt.title(country_AM + cores_40_path)
# plt.legend([Predict, Real], ["Predicted Data - CPU Util", "Real Data - CPU Util "])
# plt.show()
