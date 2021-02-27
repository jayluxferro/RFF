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
t_file = './dataset.csv'


# generating training data file
file_handler = open(t_file, 'w')

# sample file format: label, amplitudes, phase
data = 'Label,Magnitude,Phase\n'

for _ in open(data_file, 'r').readlines():
    d = _.strip()
    params = d.split(',')
    label = params[0]
    if len(params[1:]) > 1:
        _params = [ float(x.strip()) for x in params[1:] ]
        t_points = np.fft.fft(_params)
        mag = np.abs(t_points)
        phase = np.angle(t_points)

        index = 0
        for x in mag:
            data +='{},{},{}\n'.format(label, x, phase[index])
            index += 1

file_handler.write(data)
file_handler.close()
