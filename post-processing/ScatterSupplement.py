import time
import math
import numpy as np
import pandas as pd
import random


# Select the scattered points near the interface and make them denser
def densify(x_point, y_point, z_point, x_interval, y_interval, z_interval, x_num, y_num, z_num, interface, sta):
    for i in range(x_num):
        for j in range(y_num):
            for k in range(z_num):
                interface.append([x_point,  y_point, z_point + z_interval * (k - (z_num - 1) / 2), sta])


boreholes_position = "../data/EMDCT_result/test_entropy.csv"
# the path to export
path0 = "../data/EMDCT_result/interfaces_predicted_coordinates/interface0.csv"
path1 = "../data/EMDCT_result/interfaces_predicted_coordinates/interface1.csv"
path2 = "../data/EMDCT_result/interfaces_predicted_coordinates/interface2.csv"
path3 = "../data/EMDCT_result/interfaces_predicted_coordinates/interface3.csv"
path4 = "../data/EMDCT_result/interfaces_predicted_coordinates/interface4.csv"
path5 = "../data/EMDCT_result/interfaces_predicted_coordinates/interface5.csv"
path6 = "../data/EMDCT_result/interfaces_predicted_coordinates/interface6.csv"

positions = pd.read_csv(boreholes_position, header=None).values  # 训练集中的坐标部分固定钻孔位置
# Create empty containers for different interface layers
interface0 = []
interface1 = []
interface2 = []
interface3 = []
interface4 = []
interface5 = []
interface6 = []
print(positions)

for idx in range(positions.shape[0]):
    # print(positions[1][idx])
    if idx == 0:
        continue
    if positions[idx][4] == -200:  # surface
        interface0.append([positions[idx-1][2], positions[idx-1][3], positions[idx-1][4]])
    if positions[idx][1] == positions[idx - 1][1] - 1:
        distance = (positions[idx][2] - positions[idx-1][2]) ** 2 + (positions[idx][3] - positions[idx-1][3]) ** 2 + (positions[idx][4] - positions[idx-1][4])** 2
        if distance < 500:
            if positions[idx][1] == 1:
                densify(positions[idx][2], positions[idx][3], positions[idx][4], 10, 10, 0.25, 1, 1, 40, interface1, 1)
            if positions[idx][1] == 2:
                densify(positions[idx][2], positions[idx][3], positions[idx][4], 10, 10, 0.25, 1, 1, 40, interface2, 2)
            if positions[idx][1] == 3:
                densify(positions[idx][2], positions[idx][3], positions[idx][4], 10, 10, 0.25, 1, 1, 40, interface3, 3)
            if positions[idx][1] == 4:
                densify(positions[idx][2], positions[idx][3], positions[idx][4], 10, 10, 0.25, 1, 1, 40, interface4, 4)
            if positions[idx][1] == 5:
                densify(positions[idx][2], positions[idx][3], positions[idx][4], 10, 10, 0.25, 1, 1, 40, interface5, 5)
            if positions[idx][1] == 6:
                densify(positions[idx][2], positions[idx][3], positions[idx][4], 10, 10, 0.25, 1, 1, 40, interface6, 6)


np.savetxt(path0, interface0, delimiter=',', fmt='%.4f')
np.savetxt(path1, interface1, delimiter=',', fmt='%.4f')
np.savetxt(path2, interface2, delimiter=',', fmt='%.4f')
np.savetxt(path3, interface3, delimiter=',', fmt='%.4f')
np.savetxt(path4, interface4, delimiter=',', fmt='%.4f')
np.savetxt(path5, interface5, delimiter=',', fmt='%.4f')
np.savetxt(path6, interface6, delimiter=',', fmt='%.4f')
