import numpy as np
import math


def exp_moving_average(values, window):
    weights = np.exp(np.linspace(-1., 0., window))
    weights /= weights.sum()
    a = np.convolve(values, weights, mode='full')[:len(values)]
    a[:window] = a[window]
    return a


def filter_data(data, avg):
    d = []
    for x in data:
        if x >= avg:
            d.append(x)

    return d


def generate_x(data):
    return np.linspace(1, len(data), len(data))


def variance(data, avg):
    _ = 1 / len(data)
    sum = 0

    for x in data:
        sum += math.pow((x - avg), 2)
    return _ * sum


def std(data, avg):
    return math.sqrt(variance(data, avg))


def skew(data, avg):
    _ = 1 / (len(data) * math.pow(std(data, avg), 3))
    sum = 0

    for x in data:
        sum += math.pow((x - avg), 3)
    return _ * sum


def kurt(data, avg):
    _ = 1 / (len(data) * math.pow(std(data, avg), 4))
    sum = 0

    for x in data:
        sum += math.pow((x - avg), 4)
    return _ * sum
