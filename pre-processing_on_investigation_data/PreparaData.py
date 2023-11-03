from random import random
import torch
import numpy
import numpy as np
import pandas as pd
from keras.utils.np_utils import to_categorical
from sklearn.preprocessing import StandardScaler
from discretization.Data.Smote import Smote


# Combine the scatters and stratigraphic labels for virtual boreholes
def prepare_data(points_positions, points_labels):
    print('=====================================')
    print('preparing_data...')
    positions = pd.read_csv(points_positions, header=None)
    label = pd.read_csv(points_labels, header=None)
    data = pd.concat([positions, label], axis=1).values  # combining
    row = 0
    for i in data:
        if data[row, 3] == -1 or data[row, 3] == 0:
            data = np.delete(data, row, axis=0)  # Remove invalid information, such as points above the surface
        else:
            row = row + 1
    data = pd.concat([pd.DataFrame(data)], axis=0)
    data.to_csv(path_or_buf='../data/points_train_Boreholes79c0.csv',
                index_label=None, header=None, index=None)
    return data


# the path
points_train = '../data/points_train_Boreholes79c0.txt'  # The coordinates of the scatters
points_labels_train = '../data/points_train_Boreholes79c0_label_8layers.txt'  # labels
dataset_train = prepare_data(points_train, points_labels_train)
