import os, glob
import time
from functools import reduce


import pandas as pd
import plotly.express as px


#single server
from sklearn.preprocessing import MinMaxScaler

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
#cross dc
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

paths_server = [[avg_cpu_load, 'avg_cpu_load'], [avg_memory, 'avg_memory']
    , [avg_num_cores, 'avg_num_cores'], [cpu_user_util, 'cpu_user_util'],
         [max_cpu_load, 'max_cpu_load'], [max_heap, 'max_heap']
    , [p99_response_time, 'p99_response_time'], [reco_rate, 'reco_rate'], [load_score_meter, 'load_score_meter']]



# Data/Single servers/AM/40 cores 187.35 GB
data_path_servers  = 'Data/Single servers'
data_path_cross_Dc = 'Data/Cross DC'
cores_32_path = '32 cores 125.6 GB'
cores_40_path = '40 cores 187.35 GB'
cores_48_path = '48 cores 187.19 GB'
cores_72_path = '72 cores 251.63GB'
cores_40_path_copy = '40 cores 187.35 GB - Copy'
country_AM = '/AM/'
country_IL = '/IL/'
country_LA = '/LA/'


day = True # day graph or 5min graph

def getCsv(data_path,country, core_path, metric_path, name_of_metric):
    if (data_path == data_path_cross_Dc):
        all_files = glob.glob(os.path.join(data_path + country + metric_path, "*.csv"))
    else:
        all_files = glob.glob(os.path.join(data_path + country + core_path + metric_path, "*.csv"))
    all_csv = (avg5minToDay(f) for f in all_files)
    new_csv = pd.concat(all_csv, ignore_index=True)
    new_csv.columns = ['dates', name_of_metric]
    return new_csv

def avg5minToDay(f):
    dateLength = 10
    df = pd.read_csv(f, sep=',')
    date = df['ds'][0]
    date = date[:dateLength]
    mean = df['y'].mean()
    mead_df = pd.DataFrame({'ds':[date],'y':[mean]})
    if (day):
        return mead_df
    else:
        return df


def get_paths(path):
	dirlist = [item for item in os.listdir(path) if os.path.isdir(os.path.join(path, item))]
	dirlist = [['/' + item, item] for item in dirlist]
	return dirlist

def merge_and_drop_dups(left, right):
	left = pd.merge(left, right, on=['dates'], how='inner')
	left.drop_duplicates(inplace=True)
	return left


def plot(data_path,country,cores_path, figure_num):
    paths = get_paths(data_path + country + cores_path)
    csv_data_cores = [getCsv(data_path, country, cores_path, path[0], path[1]) for path in paths]
    csv_data_cores = reduce(lambda left, right: merge_and_drop_dups(left, right), csv_data_cores)
    csv_data_cores.drop(['avg_memory', 'avg_num_cores'], axis='columns', inplace=True)
    csv_data_cores.dropna(inplace=True)
    csv_data_cores.drop_duplicates(subset=['dates'], inplace=True)
    csv_data_cores.set_index('dates', inplace=True)
    csv_data_cores = csv_data_cores.sort_values(by=['dates'])
    csv_data_cores.reset_index(inplace=True)
    data_to_scale_cores = csv_data_cores.drop('dates', 1)
    lst_of_features = list(data_to_scale_cores)
    scaler = MinMaxScaler()
    scaler.fit(data_to_scale_cores)
    data_to_scale_cores[lst_of_features] = scaler.fit_transform(data_to_scale_cores[lst_of_features])

    # normalized_df_cores = (data_to_scale_cores - data_to_scale_cores.min()) / (
    #         data_to_scale_cores.max() - data_to_scale_cores.min())
    normalized_df_cores = data_to_scale_cores.merge(
        right=csv_data_cores['dates'],
        left_index=True,
        right_index=True,
        suffixes=['', '_norm'])
    fig = px.line(normalized_df_cores, x='dates', y=normalized_df_cores.columns[1:], title=data_path+country+cores_path)
    # fig = go.Figure(go.Scatter(x=normalized_df_cores['dates'], y=normalized_df_cores[1:],
    #                            name='Share Prices (in USD)'))
    #
    # fig.update_layout(title=data_path+country+cores_path,
    #                   plot_bgcolor='rgb(230, 230,230)',
    #                   showlegend=True)
    return fig

# plot(cores_32_path,1)

# fig1 = plot(paths_server,data_path_servers,country_AM,cores_32_path, 2)
# fig1.show()
fig2 = plot(data_path_servers,country_AM,cores_40_path, 2)
fig2.show()
# fig3 = plot(paths_server,data_path_servers,country_AM,cores_48_path,2)
# fig3.show()
# fig4 = plot(paths_server,data_path_servers,country_IL,'48 cores 188.27GB',2)
# fig4.show()
# fig5 = plot(paths_server,data_path_servers,country_LA,cores_72_path,2)
# fig5.show()
# fig6 = plot(paths_cross_dc,data_path_cross_Dc,country_AM,'', 2)
# fig6.show()
#
#
# fig4.show()
# fig5.show()
# fig6.show()




# # fig4 = plot(paths_cross_dc,data_path_cross_Dc,country_AM,'', 2)
# fig1 = plot(paths_cross_dc,data_path_cross_Dc,country_AM,cores_32_path, 2)
# # fig1.show()
# fig2 = plot(paths_cross_dc,data_path_cross_Dc,country_AM,cores_40_path, 2)
# trace1 = fig1['data'][0]
# trace3 = fig1['data'][1]
# trace2 = fig2['data'][0]
#
# fig = make_subplots(rows=2, cols=1, shared_xaxes=False)
# fig.add_trace(trace1, row=1, col=1)
# fig.add_trace(trace3, row=1, col=1)
# fig.add_trace(trace2, row=2, col=1)
# fig.show()
# fig3 = plot(paths_cross_dc,data_path_cross_Dc,country_AM,cores_48_path,2)
# fig = make_subplots(rows=4, cols=1)
# fig.append_trace(fig1,row=1,col=1)
# fig.append_trace(fig2,row=2,col=1)
# fig.append_trace(fig3,row=3,col=1)
# fig.append_trace(fig4,row=4,col=1)
# plot(cores_48_path,1)
# plt.show()
