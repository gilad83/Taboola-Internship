import os, glob
import time
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
cores_32_path = '32 cores 125.6 GB'
cores_40_path = '40 cores 187.35 GB'
cores_48_path = '48 cores 187.19 GB'
cores_40_path_copy = '40 cores 187.35 GB - Copy'


def getCsv(data_path, core_path, metric_path, name_of_metric):
    all_files = glob.glob(os.path.join(data_path + core_path + metric_path, "*.csv"))
    all_csv = (pd.read_csv(f, sep=',') for f in all_files)
    new_csv = pd.concat(all_csv, ignore_index=True)
    new_csv.columns = ['dates', name_of_metric]
    return new_csv


# def plotting():

def plot(cores_path,figure_num):
    csv_data_cores = [getCsv(data_path, cores_path, path[0], path[1]) for path in paths]
    csv_data_cores = reduce(lambda left, right: pd.merge(left, right, on=['dates'],
                                                         how='outer'), csv_data_cores)
    csv_data_cores = csv_data_cores.dropna()
    data_to_scale_cores = csv_data_cores.drop('dates', 1)
    data_to_scale_cores = data_to_scale_cores.drop('avg_memory', 1)
    normalized_df_cores = (data_to_scale_cores - data_to_scale_cores.min()) / (
                data_to_scale_cores.max() - data_to_scale_cores.min())
    normalized_df_cores = normalized_df_cores.merge(
        right=csv_data_cores['dates'],
        left_index=True,
        right_index=True,
        suffixes=['', '_norm'])
    normalized_df_cores = normalized_df_cores.melt('dates', var_name='cols', value_name='vals')
    g = sns.lineplot(x="dates", y="vals", hue='cols', data=normalized_df_cores)
    g.set(title=cores_path)
    g.xaxis.set_major_locator(MultipleLocator(200))
    plt.xlim(0)
    plt.figure(figure_num)
    plt.show(block=False)


# plot(cores_32_path,1)
# plot(cores_40_path,2)
plot(cores_48_path,1)
# plt.show()
