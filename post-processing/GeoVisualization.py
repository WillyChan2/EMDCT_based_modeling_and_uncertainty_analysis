import PVGeo
import pyvista
import pandas as pd
import numpy as np

# Extract points, and the number of valid points
labelled_points = np.loadtxt("..data/EMDCT_result/test_entropy_1.csv", delimiter=",")  # Visual information entropy
# labelled_points = np.loadtxt("..data/EMDCT_result/test_points_2.csv", delimiter=",")  # Visual formation

# visualization
vtkpoints = PVGeo.points_to_poly_data(labelled_points)
voxelizer = PVGeo.filters.VoxelizePoints()
grid = voxelizer.apply(vtkpoints)
pyvista.set_plot_theme('document')

# slice
slices = grid.slice_orthogonal()
clip = grid.clip(normal='-x').clip(normal='y').threshold(0.0)
# clip = grid.threshold(value=0.0, preference='point')
p = pyvista.Plotter()
# p.add_mesh(slices)
p.add_mesh(clip)
p.show_grid()
p.show()



