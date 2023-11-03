import numpy as np
# Grid points of space for test set


def meshing(x1, x2, y1, y2, z1, z2, Nx, Ny, Nz, path_out):
    x_range = [x1, x2]
    y_range = [y1, y2]
    z_range = [z1, z2]
    _X = np.linspace(x_range[0], x_range[1], Nx)  # It is evenly divided into Nx parts in the x direction
    _Y = np.linspace(y_range[0], y_range[1], Ny)
    _Z = np.linspace(z_range[0], z_range[1], Nz)
    X, Y, Z = np.meshgrid(_X, _Y, _Z)  # meshing
    XYZ = np.concatenate((X.reshape(-1, 1), Y.reshape(-1, 1), Z.reshape(-1, 1)), axis=1)
    print(XYZ)
    np.savetxt(path_out, XYZ, delimiter=',') # the path to store the coordinates of the grid points


# dam site area: 100*130*80 = 1,040,000
# Dam site area is the main research area of dam construction, which needs detailed modeling
meshing(40100, 41100, 69300, 70600, -200, 600, 100, 130, 80, "../data/points_general62.csv")
# overall working area: 80*80*40 = 256,000
# Refine modeling is also required in other areas, but not as fine as in the dam site area
meshing(40100, 41100, 69300, 70600, -200, 600, 80, 80, 40, "../data/points_general61.csv")