import os, glob
from functools import reduce
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
# from keras.models import Sequential
# from keras.layers import Dense
# from keras.layers import LSTM
# from keras import metrics
import argparse
import plotly.graph_objects as go
import seaborn as sns

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


def getCsv(data_path, path, metric_path, name_of_metric):
    if data_path == data_path_cross_Dc:
        all_files = glob.glob(os.path.join(path + metric_path, "*.csv"))
    else:
        all_files = glob.glob(os.path.join(path + metric_path, "*.csv"))
    all_csv = (pd.read_csv(f, sep=',') for f in all_files)
    new_csv = pd.concat(all_csv, ignore_index=True)
    new_csv.columns = ['dates', name_of_metric]
    return new_csv


# Data/Single servers/AM/40 cores 187.35 GB
def get_paths(path, predict_metric_name):
    dirlist = [item for item in os.listdir(path) if os.path.isdir(os.path.join(path, item))]
    a = dirlist.index(predict_metric_name)
    b = len(dirlist) - 1
    dirlist[b], dirlist[a] = dirlist[a], dirlist[b]  # push the predict_metric_name to the end
    dirlist = [['/' + item, item] for item in dirlist]
    return dirlist


def getDataSet(predict_metric_name, path_org, data_path):
    paths = get_paths(path_org, predict_metric_name)
    csv_data_cores = [getCsv(data_path, path_org, path[0], path[1]) for path in paths]
    # gilad - if we keep this line the number of rows is 115,638 which doesn't make sense since 289 lines a day X 89 days = 25,721 rows in total - also after the merge.
    # csv_data_cores = reduce(lambda left, right: pd.merge(left, right, on=['dates'], how='outer'), csv_data_cores)
    csv_data_cores = reduce(lambda left, right: merge_and_drop_dups(left, right), csv_data_cores)
    csv_data_cores.drop(['avg_memory', 'avg_num_cores'], axis='columns', inplace=True)
    csv_data_cores.dropna(inplace=True)
    csv_data_cores.drop_duplicates(subset=['dates'], inplace=True)
    csv_data_cores.set_index('dates', inplace=True)
    csv_data_cores = csv_data_cores.sort_values(by=['dates'])
    csv_data_cores.reset_index(inplace=True)
    # print(len(csv_data_cores))
    # csv_data_cores = add_isWeekend_feature(csv_data_cores)
    # csv_data_cores = add_trend(csv_data_cores)
    # drop date
    data_witout_dates = csv_data_cores.drop('dates', 1)
    return data_witout_dates, csv_data_cores


def merge_and_drop_dups(left, right):
    left = pd.merge(left, right, on=['dates'], how='inner')
    left.drop_duplicates(inplace=True)
    return left


def scale(data_to_scale, predicted_metric):
    sc = MinMaxScaler()
    sc.fit(data_to_scale)
    data_to_scale = sc.fit_transform(data_to_scale)
    predicted_metric_reshape = predicted_metric.values.reshape(-1, 1)
    predicted_metric = sc.fit_transform(predicted_metric_reshape)
    return data_to_scale, predicted_metric


def add_multiply(dataset):
    feature_names1 = dataset.columns
    feature_names2 = dataset.columns

    for feature1 in feature_names1:
        feature_names2 = feature_names2[1:]
        for feature2 in feature_names2:
            if feature1 != feature2 and feature1 != "cpu_user_util" and feature2 != "cpu_user_util":
                to_add = dataset[feature1] * dataset[feature2]
                dataset[feature1 + " * " + feature2] = to_add
    return dataset


new_path = data_path_servers + country_AM + cores_40_path
data_to_scale_no_dates, csv_data_with_dates = getDataSet('cpu_user_util',new_path,data_path_servers)
cpu_user_util_csv = csv_data_with_dates['cpu_user_util']
data_scaled_no_dates = (data_to_scale_no_dates-data_to_scale_no_dates.min())/(data_to_scale_no_dates.max()-data_to_scale_no_dates.min())
# multiply_data = add_multiply(data_scaled_no_dates)
multiply_data = add_multiply(data_to_scale_no_dates)

# f, ax = plt.subplots(figsize=(36, 36))
correlated_features = set()
correlation_matrix = multiply_data.corr()
threshold = 0.75
last = len(data_scaled_no_dates.columns) - 1
# for i in range(len(correlation_matrix.columns)):
#     for j in range(i):
#         if abs(correlation_matrix.iloc[i, j]) < threshold and (correlation_matrix.columns[j] not in correlated_features):
#             colname = correlation_matrix.columns[i]
#             correlated_features.add(colname)
j = 0
for i in correlation_matrix['cpu_user_util']:
    if i < threshold:
        correlated_features.add(correlation_matrix['cpu_user_util'].keys()[j])
    j = j + 1


# use this handy way to swap the elements


# assign back, the order will now be swapped
correlation_matrix.drop(labels=correlated_features, axis=1, inplace=True)
correlation_matrix.drop(labels=correlated_features, inplace=True)
g1 = sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
# g2 = sns.heatmap(multiply_data.corr(), annot=True, cmap='coolwarm')
x = 5
