import numpy as np
import math
import seaborn as sns
from sklearn.metrics import confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt

variance_cutoff = 1.1 * math.pow(10, -5)
max_amplitude_threshold = 20.5
min_amplitude_threshold = 0
min_points = 5
class_names = ['TX1', 'TX2', 'TX3', 'TX4']
directory_index = ['0', '1', '2', '3']
dataset_filters = ['.CSV']

def get_transient_points(data, dataPos=None, min_amplitude_threshold=min_amplitude_threshold, max_amplitude_threshold=max_amplitude_threshold):
    mean_waveform_points = np.mean(data)
    print(np.max(data))
    sliding_window = 2
    v_i = []
    transient_points = []
    transient_points_pos = []
    max_amplitude_threshold_found = False
    counter = 1 if dataPos == None else 0
    previous_data_point = 0
    for _ in data:
        _compute = 1 / (sliding_window - 1) * (mean_waveform_points - _) ** 2

        v_i.append(_compute)

        if _ >= max_amplitude_threshold:
            max_amplitude_threshold_found = True

        if _compute >= variance_cutoff and _ >= min_amplitude_threshold and _ <= max_amplitude_threshold and _ > previous_data_point:
            if not max_amplitude_threshold_found:
                transient_points.append(_)
                if dataPos == None:
                    transient_points_pos.append(counter)
                else:
                    transient_points_pos.append(dataPos[counter])
                previous_data_point = _

        sliding_window += 1
        counter += 1
    return transient_points, transient_points_pos

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

def cm_analysis(y_true, y_pred, labels, filename, figsize=(10, 6), annot=True):
    """
    Generate matrix plot of confusion matrix with pretty annotations.
    The plot image is saved to disk.
    args:
      y_true:    true label of the data, with shape (nsamples,)
      y_pred:    prediction of the data, with shape (nsamples,)
      filename:  filename of figure file to save
    """
    cm = confusion_matrix(y_true, y_pred)
    #print(cm)
    cm = pd.DataFrame(cm, index=labels, columns=labels)
    cm.index.name = 'Actual'
    cm.columns.name = 'Predicted'
    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=annot, fmt='d')
    plt.savefig(filename)
