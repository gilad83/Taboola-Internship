import os, glob
from functools import reduce
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras import metrics
import argparse
import plotly.graph_objects as go

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

def getCsv(data_path, path, metric_path,name_of_metric):
	if data_path == data_path_cross_Dc:
		all_files = glob.glob(os.path.join(path + metric_path, "*.csv"))
	else:
		all_files = glob.glob(os.path.join(path + metric_path, "*.csv"))
	all_csv = (pd.read_csv(f, sep=',') for f in all_files)
	new_csv = pd.concat(all_csv, ignore_index=True)
	new_csv.columns = ['dates', name_of_metric]
	return new_csv
# Data/Single servers/AM/40 cores 187.35 GB
def get_paths(path,predict_metric_name):
	dirlist = [item for item in os.listdir(path) if os.path.isdir(os.path.join(path, item))]
	a = dirlist.index(predict_metric_name)
	b = len(dirlist) - 1
	dirlist[b], dirlist[a] = dirlist[a], dirlist[b] #push the predict_metric_name to the end
	dirlist = [['/' + item, item] for item in dirlist]
	return dirlist

def getDataSet(predict_metric_name, path_org,data_path):
	paths = get_paths(path_org,predict_metric_name)
	csv_data_cores = [getCsv(data_path, path_org, path[0], path[1]) for path in paths]
	# gilad - if we keep this line the number of rows is 115,638 which doesn't make sense since 289 lines a day X 89 days = 25,721 rows in total - also after the merge.
	#csv_data_cores = reduce(lambda left, right: pd.merge(left, right, on=['dates'], how='outer'), csv_data_cores)
	csv_data_cores = reduce(lambda left, right: merge_and_drop_dups(left, right), csv_data_cores)
	csv_data_cores.drop(['avg_memory', 'avg_num_cores'], axis='columns', inplace=True)
	csv_data_cores.dropna(inplace=True)
	csv_data_cores.drop_duplicates(subset=['dates'], inplace=True)
	csv_data_cores.set_index('dates', inplace=True)
	csv_data_cores = csv_data_cores.sort_values(by=['dates'])
	csv_data_cores.reset_index(inplace=True)
	#print(len(csv_data_cores))
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

def model_settings(X_train, Y_train,arguments):
	model = Sequential()
	num_of_features = X_train.shape[2]
	model.add(LSTM(arguments.number_of_nodes, activation='relu', input_shape=(1, num_of_features),recurrent_activation='hard_sigmoid'))
	model.add(Dense(1))
	model.compile(loss='mean_squared_error', optimizer='adam', metrics=[metrics.mae, 'accuracy'])
	model.fit(X_train, Y_train, epochs=arguments.epochs, batch_size=arguments.batch_size, verbose=2)
	return model

def split_train_test(n_time_steps, values, train_size):
	values_X, values_y = make_time_steps_data(values, n_time_steps)

	n_train_hours = int((len(values_X)) * train_size)
	train_X = values_X[:n_train_hours, :]
	train_y = values_y[:n_train_hours]

	test_X = values_X[n_train_hours:, :]
	test_y = values_y[n_train_hours:]

	# reshape input to be 3D [samples, timesteps, features]
	train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
	test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))

	return train_X, train_y, test_X, test_y

def make_time_steps_data(values, n_time_steps):
	# split into input and outputs - the last column will be the target metric
	values_to_train = values[:len(values)-n_time_steps, :-1]
	values_to_test = values[n_time_steps:, -1]
	return values_to_train, values_to_test

def main(arguments):
	# get_data with no date, and all data in csv
	new_path = data_path_servers + country_AM + cores_40_path
	data_to_scale_no_dates, csv_data_with_dates = getDataSet(arguments.predict_metric_name,new_path,data_path_servers)
	# save predicted metric
	cpu_user_util_csv = csv_data_with_dates['cpu_user_util']
	save_dates = csv_data_with_dates['dates']

	# scale data
	data_scaled_no_dates, cpu_user_util_csv_reshape_scaled = scale(data_to_scale_no_dates, cpu_user_util_csv)

	#TODO: Gilad - the cpu user util in our case shold be the Y_train and Y_test
	# split into test & train
	X_train, Y_train , X_test, Y_test = split_train_test(arguments.timesteps_to_the_future, data_scaled_no_dates, 0.75)


	# create the lstm model
	lstm_model = model_settings(X_train, Y_train,arguments)
	predict = lstm_model.predict(X_test)


	fig = go.Figure([
		go.Scatter(
			name='Real',
			x=save_dates.values[Y_train.shape[0]:].reshape(-1),
			y=Y_test.reshape(-1),
			mode='markers+lines',
			marker=dict(color='red', size=1),
			showlegend=True,
			connectgaps=False

		),
		go.Scatter(
			name='Predict',
			x=save_dates.values[Y_train.shape[0]:].reshape(-1),
			y=predict.reshape(-1),
			mode='lines',
			marker=dict(color="#444"),
			line=dict(width=1),
			showlegend=True,
			connectgaps=False
		)])
	fig.update_layout(
		title=  new_path + "\n"+ "**predicted metric = "+arguments.predict_metric_name+", time steps = "+str(arguments.timesteps_to_the_future)+"**",
		xaxis_title="dates",
		yaxis_title="vals",
		legend_title="Legend Title",
		)

	fig.show()
	pass

# Real, = plt.plot(Y_test)
# Predict, = plt.plot(predict)
# plt.title(country_AM + cores_40_path)
# plt.legend([Predict, Real], ["Predicted Data - CPU Util", "Real Data - CPU Util "])
# plt.show()

# plt.figure(2)
# plt.scatter(Y_test, predict)
# plt.show(block=False)


# Real, = plt.plot(save_dates.values[:Y_test.shape[0]],Y_test)
# Predict, = plt.plot(save_dates.values[:Y_test.shape[0]],predict)


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='LSTM supervised')
	parser.add_argument('--timesteps_to_the_future', dest='timesteps_to_the_future', type=int, required=True, help='timesteps to predict', default=6)
	parser.add_argument('--batch_size', dest='batch_size', type=int, required=False, help='batch size', default=128)
	parser.add_argument('--epochs', dest='epochs', type=int, required=False, help='epochs', default=20)
	parser.add_argument('--number_of_nodes', dest='number_of_nodes', type=int, required=False, help='number of nodes', default=50)
	parser.add_argument('--predict_metric_name', dest='predict_metric_name', type=str, required=False, help='predict metric name',
						default='cpu_user_util')
	args = parser.parse_args()
	main(args)

