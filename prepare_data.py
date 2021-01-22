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

data_file = 'data.csv'
new_data_file = './p_data.csv'
t_file = './dataset.csv'

file_handler = open(new_data_file, 'w')
data = ''
# reading data file
for _ in open(data_file, 'r').readlines():
    d = _.strip()
    d = d.split(',')

    if len(d) >= (fx.min_points + 1):
        data += ','.join(d[:fx.min_points + 1]) + '\n'
file_handler.write(data)
file_handler.close()

# generating training data file
file_handler = open(t_file, 'w')
# sample file format: label, amplitudes, phase

data = ''
for _ in open(new_data_file, 'r').readlines():
    d = _.strip()
    params = d.split(',')
    label = params[0]
    _params = params[1:]
    t_points = np.fft.fft(_params)
    mag = np.abs(t_points)
    phase = np.angle(t_points)
    # print(len(mag), len(phase))
    _data = label + ','
    _data += ','.join([str(i) for i in mag]) + ','
    _data += ','.join([str(i) for i in phase]) + '\n'
    data += _data

file_handler.write(data)
file_handler.close()
