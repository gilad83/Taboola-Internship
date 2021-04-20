import os, glob
from functools import reduce
from os import path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MultipleLocator
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler

max_node_load_path = '/max(node_load15{hostname=~_water401_water427_water428_water449_}) by (domain)'
p99_path = '/p99_water401_water427_water428_water449'
sum_path = '/sum(rate(TRC_server_timer_count{_label_1=_recommendationRequests_, service=_taboola-trc)'
max_over_time_path = '/max(max_over_time(TRC_server_gauge{server=~_water401_water427_water428_water449_)'
avg_node_load15_path = '/avg(node_load15{hostname=~_water401_water427_water428_water449_}) by (domain)'
avg_count_node_cpu_seconds_path = '/avg(count (node_cpu_seconds_total{mode=_idle_,hostname=~_water401_water427_water428_water449_,job=~_node_exporter_}) by (hostname))'
avg_over_time_path = '/avg(avg_over_time(TRC_server_gauge{server=label_1=_MemoryUsage)'
avg_node_memory_path = '/avg(avg(node_memory_MemTotal_bytes{hostname=~_water401_water427_water428_water449_})) by (hostname)'

# paths = [[max_node_load_path, 'max_node_load15'], [p99_path, 'p99'], [sum_path, 'sum = recommendationReq,timer_count']
#     , [max_over_time_path, 'max_over_time-mem usage'], [avg_node_load15_path, 'avg_node_load15'],
#          [avg_count_node_cpu_seconds_path, 'avg_count_node_cpu_seconds'], [avg_over_time_path, 'avg_over_time- mem usage']
#     , [avg_node_memory_path, 'avg_node_memory-bytes']]

paths = [[max_node_load_path, 'max_node_load15'], [p99_path, 'p99'], [sum_path, 'sum = recommendationReq,timer_count']
    , [max_over_time_path, 'max_over_time-mem usage'], [avg_node_load15_path, 'avg_node_load15'],
          [avg_over_time_path, 'avg_over_time- mem usage']]

data_path = '../Data/Single servers/AM/'
cores_40_path = '40 cores 187.35 GB'
cores_40_path_copy = '40 cores 187.35 GB - Copy'


def getCsv(data_path, core_path, metric_path, name_of_metric):
    all_files = glob.glob(os.path.join(data_path + core_path + metric_path, "*.csv"))
    all_csv = (pd.read_csv(f, sep=',') for f in all_files)
    new_csv = pd.concat(all_csv, ignore_index=True)
    new_csv.columns = ['dates', name_of_metric]
    return new_csv


# def plotting():



csv_data_40_cores_6_days = [getCsv(data_path, cores_40_path_copy, path[0], path[1]) for path in paths]
csv_data_40_cores_6_days = reduce(lambda left, right: pd.merge(left, right, on=['dates'],
                                                        how='outer'), csv_data_40_cores_6_days)

data_to_scale = csv_data_40_cores_6_days.drop('dates', 1)

normalized_df=(data_to_scale-data_to_scale.min())/(data_to_scale.max()-data_to_scale.min())
normalized_df = normalized_df.merge(
    right=csv_data_40_cores_6_days['dates'],
    left_index=True,
    right_index=True,
    suffixes=['', '_norm'])
normalized_df = normalized_df.melt('dates', var_name='cols',  value_name='vals')
g1 = sns.lineplot(x="dates", y="vals", hue='cols', data=normalized_df)
g1.xaxis.set_major_locator(MultipleLocator(200))
plt.xlim(0)
plt.show()







# min_max_scaler = preprocessing.MinMaxScaler()
# scaled_data = min_max_scaler.fit_transform(data_to_scale)
# csv_data_40_cores_no_dates = pd.DataFrame(scaled_data)
# csv_data_40_cores = csv_data_40_cores.melt('dates', var_name='cols',  value_name='vals')
# g = sns.lineplot(x="dates", y="vals", hue='cols', data=csv_data_40_cores)
# plt.show()

# csv_data_40_cores = [  for df in csv_data_40_cores]
# p99_merged = getCsv(data_path, cores_40_path, p99_path, 'p99')

# sns.set(rc={'figure.figsize': (20, 13.27)})
# # ax = sns.lineplot(x = 'ds', y = 'y',label='max_node_load15', data = max_node_load15 )
# ax.set(xlabel='dates ', ylabel='usage', title='metrics')
# ax.xaxis.set_major_locator(MultipleLocator(200))
# plt.xlim(0)
# plt.show()
# plotting()
