import random
import numpy as np
import math
# The drilling position is randomly selected, virtual boreholes are taken to verify the approach
path = "../data/points_train_Boreholes79c0.txt"  # the path to store the coordinates of the sampling points
xy = 10  # the number of the boreholes we drug randomly in this area
z = 30  # the number of divided of each borehole
m = 20  # the number of random in a borehole part cylinder
r = 5  # the radius of impact region
x_range = [39602.775, 41955.266]  # 2350
y_range = [68605.701, 70985.310]  # 2400
z_range = [-200.00, 600]  # 800
_Z = np.linspace(z_range[0], z_range[1], z)
Y_random = []
X_random = []
XYZ = []
line_num = 0  # Number of connections between boreholes

for i in range(xy):
    X_random.append(random.uniform(x_range[0], x_range[1]))  # 随即钻孔
    Y_random.append(random.uniform(y_range[0], y_range[1]))
X_random.append(0.5*(x_range[1]+x_range[0]))
Y_random.append(y_range[1]-10)
X_random.append(x_range[1]-50)
Y_random.append(0.5*(y_range[1]+y_range[0]))
_Y = np.array(Y_random)
_X = np.array(X_random)


def randPoint(radius, x_center, y_center):
    theta = random.random() * 2 * math.pi
    rho = math.sqrt(random.normalvariate(mu=0, sigma=radius) ** 2)
    x = x_center + rho * math.cos(theta)
    y = y_center + rho * math.sin(theta)
    return x, y


for i in range(xy):
    print('{}, {}'.format(_X[i], _Y[i]))
    line_num = 0
    for j in range(z):  # Take scatter points inside the cylinder
        XYZ.append([_X[i], _Y[i], _Z[j]])
        for k in range(m-1):
            xi, yi = randPoint(r, _X[i], _Y[i])  # Randomly generate points within the scope of a circle
            zj = random.uniform(_Z[j]-(-z_range[0]+z_range[1])/z/2, _Z[j]+(-z_range[0]+z_range[1])/z/2)
            XYZ.append([xi, yi, zj])
    for ii in range(xy):  # Line the boreholes and take the scatter points
        distance = (_X[i] - _X[ii]) ** 2 + (_Y[i] - _Y[ii]) ** 2  # Connect at a certain distance
        if 150 ** 2 > distance > 0:
            line_num += 1
            if line_num < 3:
                for kk in range(30):
                    ran1 = random.random()
                    x_k = (-_X[i] + _X[ii]) * ran1 + _X[i]
                    y_k = (-_Y[i] + _Y[ii]) * ran1 + _Y[i]
                    z_k = random.uniform(z_range[0], z_range[1])
                    XYZ.append([x_k, y_k, z_k])
# Save as txt
np.savetxt(path, XYZ, delimiter=',')