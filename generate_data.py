#!/usr/bin/env python

"""
Description: Generate Data
Date:        24/02/2020
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
import func as fx
import math
import glob
import os

dataset_path = './dataset'
data_file = 'data.csv'
classes = fx.class_names
cutoff = [(6.5, 14), (5, 11), (3, 20), (4, 21)]


# process data
def process_data(file_name, min_point, max_point):
    print(file_name, min_point, max_point)
    data = pd.read_csv(file_name, ',', low_memory=False)
    trim_start = 24
    data = data['Format'][trim_start:]

    all_data = fx.exp_moving_average(list(map(float, data)), 10)
    average_amplitude = np.mean(all_data)
    all_data = fx.filter_data(all_data, average_amplitude)
    transient_points, _ = fx.get_transient_points(all_data, None, min_point, max_point)

    return ','.join([str(i) for i in transient_points])




"""
for _tx in classes:
    for f in range(11, 71):
        file_name = '{}/{}/DS00{}.CSV'.format(dataset_path, _tx, f)
        _tx_index = classes.index(_tx)
        cutoff_points = cutoff[_tx_index]
        data += '{}, {}\n'.format(_tx_index, process_data(file_name, cutoff_points[0], cutoff_points[1]))
"""
for directory in fx.directory_index:
    for file_name in glob.iglob(dataset_path + '/{}/**'.format(directory), recursive=True):
        if os.path.isfile(file_name):
            print(file_name)
            _tx_index = int(directory)
            cutoff_points = cutoff[_tx_index]
            data = '{}, {}\n'.format(_tx_index, process_data(file_name, cutoff_points[0], cutoff_points[1]))
            file_handler = open(data_file, 'a')
            file_handler.write(data)
            file_handler.close()




