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
import statistics as stats
from scipy.stats import skew, kurtosis

data_file = 'data.csv'
t_file = './dataset.csv'


# generating training data file
file_handler = open(t_file, 'w')

# sample file format: label, amplitudes, phase
data = 'Label,MS,MK,MV,PS,PK,PV\n'

for _ in open(data_file, 'r').readlines():
    d = _.strip()
    params = d.split(',')
    label = params[0]
    """
    # testing tx1 - tx3
    if label == '1' or label == '3':
        continue
    """
    if len(params[1:]) > 1:
        _params = [ float(x.strip()) for x in params[1:] ]
        t_points = np.abs(np.fft.fft(_params))
        #mag = stats.mean(t_points)
        #phase = stats.mean(np.angle(t_points))
        mag = t_points
        phase = np.angle(fx.format_angle(t_points))
        if len(phase) > 1 and len(mag) > 1:
            _data = '{},{},{},{},{},{},{}\n'.format(label, skew(mag), kurtosis(mag), stats.variance(mag), skew(phase), kurtosis(phase), stats.variance(phase))
            if len(_data.rstrip().split(',')) == 7:
                data += _data

file_handler.write(data)
file_handler.close()
