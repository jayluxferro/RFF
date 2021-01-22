#!/usr/bin/env python

"""
Description: Transient Detection
Date:        24/02/2020
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
import func as fx
import math


def usage():
    print("Usage: python {} [file.csv] <min> <max>".format(sys.argv[0]))
    sys.exit()

if len(sys.argv) < 2:
    usage()
else:
    if sys.argv[1].lower().split('.')[-1] != 'csv':
        usage()

# check if boundary parameters were passed
try:
    min_amplitude_threshold = float(sys.argv[2])
except:
    min_amplitude_threshold = fx.min_amplitude_threshold

try:
    max_amplitude_threshold = float(sys.argv[3])
except:
    max_amplitude_threshold = fx.max_amplitude_threshold

"""
files = ['TX1', 'TX2']

file_name = files[1] + '.CSV'  # test file
"""
file_name = sys.argv[1]

data = pd.read_csv(file_name, ',', low_memory=False)
trim_start = 24
data = data['Format'][trim_start:]

all_data = fx.exp_moving_average(list(map(float, data)), 10)
average_amplitude = np.mean(all_data)
all_data = fx.filter_data(all_data, average_amplitude)
plt.figure()
plt.plot(np.linspace(1, len(all_data), len(all_data)), all_data)
plt.xlabel('X')
plt.ylabel('Amplitude')
#plt.show()

data = all_data
data = data[:math.ceil(len(data) / 2)]
x = np.linspace(1, len(data), len(data))

# print(data)
# sys.exit()

plt.figure()
plt.plot(fx.generate_x(all_data), all_data, '-o')
plt.title('Signal Waveform')
plt.xlabel('X')
plt.ylabel('Amplitude')
# plt.show()



# getting transient points
transient_points, transient_points_pos = fx.get_transient_points(all_data, None, min_amplitude_threshold, max_amplitude_threshold)
print(transient_points)
print(transient_points_pos)
"""
# print(np.min(v_i), np.max(v_i))

plt.figure()
plt.plot(fx.generate_x(v_i), v_i, '-og')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Variance')
# plt.show()
"""

plt.figure()
plt.plot(transient_points_pos, transient_points, '-or')
plt.xlabel('X')
plt.ylabel('Amplitude')
plt.title('Transient ')
plt.xlim(0, len(all_data))
# plt.show()

# feature extraction
transient_points_mean = np.mean(transient_points)
"""
var = fx.variance(transient_points, transient_points_mean)
print("Variance: {}".format(var))

skew = fx.skew(transient_points, transient_points_mean)
print("Skewness: {}".format(skew))

kurt = fx.kurt(transient_points, transient_points_mean)
print("Kurtosis: {}".format(kurt))

# fast fourier transform
transient_points_fft = np.fft.fft(transient_points)
print(transient_points_fft)
print("Magnitude: {}".format(np.abs(transient_points_fft)))
print("Phase: {}".format(np.angle(transient_points_fft)))
plt.figure()
plt.plot(transient_points_fft)
"""
plt.show()
