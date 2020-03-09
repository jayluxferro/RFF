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

files = ['TX1', 'TX2']

file_name = files[0] + '.CSV'  # test file

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


# computing mean of data points
mean_waveform_points = np.mean(data)
sliding_window = 2
v_i = []
transient_points = []
transient_points_pos = []
variance_cutoff = 1.1 * math.pow(10, -5)
max_amplitude_threshold = 20.4
max_amplitude_threshold_found = False
counter = 1
previous_data_point = 0
for _ in data:
    _compute = 1 / (sliding_window - 1) * (mean_waveform_points - _) ** 2

    v_i.append(_compute)
    
    if _ >= max_amplitude_threshold:
        max_amplitude_threshold_found = True

    if _compute >= variance_cutoff and _ >= 0 and _ <= max_amplitude_threshold and _ > previous_data_point:
        if not max_amplitude_threshold_found:
            transient_points.append(_)
            transient_points_pos.append(counter)
            previous_data_point = _

    sliding_window += 1
    counter += 1

# print(np.min(v_i), np.max(v_i))


plt.figure()
plt.plot(fx.generate_x(v_i), v_i, '-og')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Variance')
# plt.show()


plt.figure()
plt.plot(transient_points_pos, transient_points, '-or')
plt.xlabel('X')
plt.ylabel('Amplitude')
plt.title('Transient ')
# plt.show()

# feature extraction
transient_points_mean = np.mean(transient_points)

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
plt.show()
