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


avg_cpu_load = '/avg_cpu_load'
avg_heap = '/avg_heap'
avg_memory = '/avg_memory'
avg_num_cores = '/avg_num_cores'
cpu_user_util = '/cpu_user_util'
max_cpu_load = '/max_cpu_load'
max_heap = '/max_heap'
p99_response_time = '/p99_response_time'
reco_rate = '/reco_rate'

paths = [[avg_cpu_load, 'avg_cpu_load'], [avg_heap, 'avg_heap'], [avg_memory, 'avg_memory']
    , [avg_num_cores, 'avg_num_cores'], [cpu_user_util, 'cpu_user_util'],
         [max_cpu_load, 'max_cpu_load'], [max_heap, 'max_heap']
    , [p99_response_time, 'p99_response_time'], [reco_rate, 'reco_rate']]

#avg_count_node_cpu_seconds
#avg_node_memory-bytes

# paths = [[max_node_load_path, 'max_node_load15'], [p99_path, 'p99'], [sum_path, 'sum = recommendationReq,timer_count']
#     , [max_over_time_path, 'max_over_time-mem usage'], [avg_node_load15_path, 'avg_node_load15'],
#           [avg_over_time_path, 'avg_over_time- mem usage']]

data_path = 'Data/Single servers/AM/'

cores_40_path_copy = '40 cores 187.35 GB - Copy'
cores_32_path = '32 cores 125.6 GB'
cores_40_path = '40 cores 187.35 GB'
cores_48_path = '48 cores 187.19 GB'

def getCsv(data_path, core_path, metric_path, name_of_metric):
    all_files = glob.glob(os.path.join(data_path + core_path + metric_path, "*.csv"))
    all_csv = (pd.read_csv(f, sep=',') for f in all_files)
    new_csv = pd.concat(all_csv, ignore_index=True)
    new_csv.columns = ['dates', name_of_metric]
    return new_csv


# def plotting():

csv_data_40_cores = [getCsv(data_path, cores_32_path, path[0], path[1]) for path in paths]
csv_data_40_cores = reduce(lambda left, right: pd.merge(left, right, on=['dates'],
                                                        how='outer'), csv_data_40_cores)

csv_data_40_cores = csv_data_40_cores.dropna()

data_to_scale_40_cores = csv_data_40_cores.drop('dates', 1)
data_to_scale_40_cores = data_to_scale_40_cores.drop('avg_memory', 1)


normalized_df_40_cores= (data_to_scale_40_cores - data_to_scale_40_cores.min()) / (data_to_scale_40_cores.max() - data_to_scale_40_cores.min())
normalized_df_40_cores = normalized_df_40_cores.merge(
    right=csv_data_40_cores['dates'],
    left_index=True,
    right_index=True,
    suffixes=['', '_norm'])
normalized_df_40_cores = normalized_df_40_cores.melt('dates', var_name='cols', value_name='vals')
# g = sns.lineplot(x="dates", y="vals", hue='cols', data=normalized_df_40_cores)
# g.set(title="40 Cores Singel Server AM")
# g.xaxis.set_major_locator(MultipleLocator(200))
# # g.xaxis.set_major_formatter(OldAutoLocator())
# plt.xlim(0)
# # plt.show()


cpu_user_util_csv = csv_data_40_cores['cpu_user_util']
data_to_scale_40_cores = data_to_scale_40_cores.drop('cpu_user_util', 1) #no dates and noe cpu util



sc = MinMaxScaler()
sc.fit(data_to_scale_40_cores)
data_40_cores_scaled = sc.fit_transform(data_to_scale_40_cores)
cpu_user_util_csv_reshape = cpu_user_util_csv.values.reshape(-1,1)
cpu_user_util_csv_reshape_scaled = sc.fit_transform(cpu_user_util_csv_reshape)


X_train, X_test, Y_train, Y_test = train_test_split(data_40_cores_scaled, cpu_user_util_csv_reshape_scaled, test_size = 0.25)

X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

model = Sequential()

model.add(LSTM(20, activation='relu', input_shape=(1,7), recurrent_activation='hard_sigmoid'))

model.add(Dense(1))

#adam
#relu
model.compile(loss='mean_squared_error', optimizer='adam', metrics=[metrics.mae])

model.fit(X_train, Y_train, epochs=50, batch_size= 1 , verbose=2)

predict = model.predict(X_test)

plt.figure(2)
plt.scatter(Y_test, predict)
plt.show(block=False)

plt.figure(3)
Real , = plt.plot(Y_test)
Predict, = plt.plot(predict)
plt.title('CPU UTIL(32 cores)')
plt.legend([Predict, Real], ["Predicted Data - CPU Util", "Real Data - CPU Util "])
plt.show()






