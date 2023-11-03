import time
import math
import numpy as np
import pandas as pd
import random


# Simply interpolate between two points
def interpolate(x0_point, y0_point, z0_point, x1_point, y1_point, z1_point, interface):
    # r = random.random()
    r = 0.5
    x = r * x0_point + (1 - r) * x1_point
    y = r * y0_point + (1 - r) * y1_point
    z = r * z0_point + (1 - r) * z1_point
    interface.append([x, y, z])


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
        distance = (positions[idx][2] - positions[idx-1][2]) ** 2 + (positions[idx][3] - positions[idx-1][3]) ** 2 + (positions[idx][4] - positions[idx-1][4]) ** 2
        if distance < 500:
            if positions[idx][1] == 1:
                interpolate(positions[idx][2], positions[idx][3], positions[idx][4], positions[idx-1][2], positions[idx-1][3], positions[idx-1][4], interface1)
            if positions[idx][1] == 2:
                interpolate(positions[idx][2], positions[idx][3], positions[idx][4], positions[idx-1][2], positions[idx-1][3], positions[idx-1][4], interface2)
            if positions[idx][1] == 3:
                interpolate(positions[idx][2], positions[idx][3], positions[idx][4], positions[idx-1][2], positions[idx-1][3], positions[idx-1][4], interface3)
            if positions[idx][1] == 4:
                interpolate(positions[idx][2], positions[idx][3], positions[idx][4], positions[idx-1][2], positions[idx-1][3], positions[idx-1][4], interface4)
            if positions[idx][1] == 5:
                interpolate(positions[idx][2], positions[idx][3], positions[idx][4], positions[idx-1][2], positions[idx-1][3], positions[idx-1][4], interface5)
            if positions[idx][1] == 6:
                interpolate(positions[idx][2], positions[idx][3], positions[idx][4], positions[idx-1][2], positions[idx-1][3], positions[idx-1][4], interface6)


np.savetxt(path0, interface0, delimiter=',', fmt='%.4f')
np.savetxt(path1, interface1, delimiter=',', fmt='%.4f')
np.savetxt(path2, interface2, delimiter=',', fmt='%.4f')
np.savetxt(path3, interface3, delimiter=',', fmt='%.4f')
np.savetxt(path4, interface4, delimiter=',', fmt='%.4f')
np.savetxt(path5, interface5, delimiter=',', fmt='%.4f')
np.savetxt(path6, interface6, delimiter=',', fmt='%.4f')
