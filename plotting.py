#Using AMS DC from Drive
import os ,glob
from functools import reduce
from os import path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MultipleLocator
print('sep')
print(os.sep)
print('sep')
def import_data(dataPath):
    # combine all dates in 5M

    # combine all dates in P99
    f_path = dataPath + os.sep + "trc_requests_timer_p99_weighted_dc"
    dfs = [pd.read_csv(path.join(f_path, x)) for x in os.listdir(f_path) if path.isfile(path.join(f_path, x))]
    dataset_P99 = pd.concat(dfs)
    dataset_P99.columns = ['ds', 'y']
    dataset_P99.plot()
    # merge
    dfs = [dataset_P99]
    dataset = reduce(lambda left, right: pd.merge(left, right, on='ds'), dfs)
    dataset.drop_duplicates(subset=None, inplace=True)
    dataset.drop('ds', 1)
    dataset.drop(dataset.columns[[0]], axis=1, inplace=True)
    return dataset
def single_plot(dataset):

    sns.set(rc={'figure.figsize':(18,12.27)})
    ax = sns.lineplot(x = 'ds', y = 'y', data = dataset )
    ax.set(xlabel='dates ', ylabel='requests', title='requests_timer_p99_weighted_dc_2021-01-05')
    ax.xaxis.set_major_locator(MultipleLocator(32))
    plt.show()

data_path = 'Data/Dc/AM/'


all_files = glob.glob(os.path.join(data_path+'trc_requests_timer_p99_weighted_dc', "trc_*.csv"))

all_csv = (pd.read_csv(f, sep=',') for f in all_files)
df_merged_p99   = pd.concat(all_csv, ignore_index=True)