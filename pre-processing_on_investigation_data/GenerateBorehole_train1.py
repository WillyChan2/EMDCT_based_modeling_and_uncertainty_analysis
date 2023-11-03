import random

import numpy
import numpy as np
import math
import pandas as pd
# Given a specific real borehole location to sample,and label them.
path_in = "../data/borehole6.csv"  # the path of structured borehole data.
'''The structure of the file: 
Name of borehole, 
x-coordinate, 
y-coordinate, 
z-coordinate(elevation), 
Sequential interface depth... '''
path_out_points = "../data/points_train_Boreholes6.csv"  # the path to store the coordinates of the sampling points
# Actual borehole coordinates (X,Y)
positions = pd.read_csv(path_in, header=None).values
z = 30  # the number of divided of each borehole
m = 20  # the number of random in a borehole part cylinder
r = 5  # the radius of impact region.
z_range = [-200.00, 600]  # 800
XYZL = []
_Z = np.linspace(z_range[0], z_range[1], z)


# Generates a random point within the range of a circle
def randPoint(radius, x_center, y_center):
    theta = random.random() * 2 * math.pi
    rho = math.sqrt(random.normalvariate(mu=0, sigma=radius) ** 2)
    x = x_center + rho * math.cos(theta)
    y = y_center + rho * math.sin(theta)
    return x, y


# read the Labels
# Take scatter points inside the cylinder
def readLabel1(id, height):
    label = -1
    for pos in positions:
        if id == pos[0]:  # Read the elevation of a borehole
            h = pos[3] - height
            if h < 0:
                label = 0
                break
            else:
                i = 0
                for layer in pos:
                    if i > 3:
                        if h < layer:
                            label = i - 3
                            break
                    label = i - 2
                    i += 1
    return label


# read the Labels
# Line the boreholes and take the scatter points
def readLabel2(id1, id2, height, x):
    label = -1
    for pos1 in positions:
        for pos2 in positions:
            if id1 == pos1[0] and id2 == pos2[0]:  # Read the line of a pair of holes
                i = 0
                r = (x - pos1[1]) / (pos2[1] - pos1[1])
                h = pos1[3] + r * (pos2[3] - pos1[3])  # The absolute elevation of the surface at the line
                for layer1 in pos1:
                    if height > h:
                        label = 0
                        break
                    else:
                        if i > 3 :
                            a = pos1[3] - pos1[i]
                            b = pos2[3] - pos2[i]
                            h1 = a + r * (b - a)  # The absolute level of the interface at the connection
                            if height < h1:
                                label = i - 3
                                break
                        label = i - 2
                        i += 1
    return label


def samplingPoints():
    # Number of boreholes
    i = 0
    # Take scatter points for training set
    for pos in positions:
        i += 1
        XX = pos[1] + random.random() * 10 - 5
        YY = pos[2] + random.random() * 10 - 5
        for k in range(z):  # Take scatter points inside the cylinder
            # XYZ.append([XX, YY, _Z[k]])
            for l in range(m):
                xi, yj = randPoint(r, XX, YY)
                zk = random.uniform(_Z[k]-(-z_range[0]+z_range[1])/z/2, _Z[k]+(-z_range[0]+z_range[1])/z/2)
                label1 = readLabel1(pos[0], zk)
                XYZL.append([xi, yj, zk, label1])
        line_num = 0  # Number of connections between boreholes
        for pos1 in positions:  # Line the boreholes and take the scatter points
            distance = ((pos[2] - pos1[2]) ** 2 + (pos[1] - pos1[1]) ** 2)  # Connect at a certain distance
            if 150 ** 2 > distance > 0:
                line_num += 1
                if line_num < 4:
                    for kk in range(30):
                        ran1 = random.random()
                        x_k = (-pos[1] + pos1[1]) * ran1 + pos[1]
                        y_k = (-pos[2] + pos1[2]) * ran1 + pos[2]
                        z_k = random.uniform(z_range[0], z_range[1])
                        label2 = readLabel2(pos1[0], pos[0], z_k, x_k)
                        XYZL.append([x_k, y_k, z_k, label2])


def prepareData(data):
    row = 0
    for i in data:
        data = np.array(data)
        if data[row, 3] == -1 or data[row, 3] == 0:
            data = np.delete(data, row, axis=0)  # Remove invalid information, and points above the surface
        else:
            row = row + 1
    data = pd.concat([pd.DataFrame(data)], axis=0)
    return data


samplingPoints()
XYZL = prepareData(XYZL)
# Save as txt
XYZL.to_csv(path_or_buf= path_out_points,
            index_label=None, header=None, index=None)
