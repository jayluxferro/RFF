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

data_file = 'data.csv'
t_file = './dataset.csv'


# generating training data file
file_handler = open(t_file, 'w')

# sample file format: label, amplitudes, phase
#data = 'Label,Magnitude,Phase\n'
data = 'Label,Mean,GMean,HMean,Median,MedianL,MedianH,MedianG,Variance,Stdev\n'

for _ in open(data_file, 'r').readlines():
    d = _.strip()
    params = d.split(',')
    label = params[0]
    if len(params[1:]) > 1:
        _params = [ float(x.strip()) for x in params[1:] ]
        data += '{},{},{},{},{},{},{},{},{},{}\n'.format(label, stats.mean(_params), stats.geometric_mean(_params), stats.harmonic_mean(_params), stats.median(_params), stats.median_low(_params), stats.median_high(_params), stats.median_grouped(_params), stats.variance(_params), stats.stdev(_params))
file_handler.write(data)
file_handler.close()
