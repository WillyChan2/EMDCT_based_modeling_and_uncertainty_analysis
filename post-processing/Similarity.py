import pandas as pd
import numpy as np
import math


# Calculated mean
def non_zero_mean(np_arr):
    exist = (np_arr != 0)
    num = np_arr.sum(axis=0)
    den = exist.sum(axis=0)
    return num/den


# Inserts a character at the specified position in the string
def str_insert(str_origin, pos, str_add):
    str_list = list(str_origin)
    str_list.insert(pos, str_add)
    str_out = ''.join(str_list)
    return str_out


# Main calculation method. Firstly, match points with the same (x,y) coordinates to compare Z coordinates
def match_similarity(rst_interface, std_surface):
    rst = pd.read_csv(rst_interface, header=None).values
    std = pd.read_csv(std_surface, header=None).values

    # constant
    Delta_xy = 100
    Den = 0
    Z0_bar = non_zero_mean(std)[2]
    Z1_bar = non_zero_mean(rst)[2]
    Loss = 0
    Z0_moment = 0
    Z1_moment = 0
    Z0Z1_moment = 0
    Z01_moment = 0

    for idx in range(std.shape[0]):
        for idx1 in range(rst.shape[0]):
            if std[idx][0] == rst[idx1][0]:
                if std[idx][1] == rst[idx1][1]:
                    b = True
                    loss = (std[idx][2] - rst[idx1][2]) ** 2
                    z0 = (std[idx][2] - Z0_bar) ** 2
                    z1 = (rst[idx1][2] - Z1_bar) ** 2
                    z0z1 = (std[idx][2] - Z0_bar) * (rst[idx1][2] - Z1_bar)
                    z01 = (rst[idx1][2] - std[idx][2]) ** 2
                    Loss += loss
                    Z0_moment += z0
                    Z1_moment += z1
                    Z0Z1_moment += z0z1
                    Z01_moment += z01
                    Den += 1
                    break

    Loss = math.sqrt(Loss / Den)
    R_sqar = 1 - Z01_moment / Z0_moment
    print('root mean square error (RMSE): \n', Loss)
    print('Correlation coefficient (R^2): \n', R_sqar)
    print('--------------------------------------------')


INTERFACE = 6  # the number of interfaces
rst_path = "../data/EMDCT_result/interfaces_predicted_coordinates/interface.csv"  # Each boundary surface's predicted coordinates
std_path = "../data/EMDCT_result/control_point/surface0.txt"  # Each boundary surface's control point(correct) coordinates
for i in range(INTERFACE):
    index = str(i+1)
    rst_interface_path = str_insert(rst_path, -4, index)
    std_surface_path = str_insert(std_path, -4, index)
    match_similarity(rst_interface_path, std_surface_path)


