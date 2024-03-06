# import, process and visualize a point cloud in python
## Source: Florent Poux, Ph.D.

## libraries
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

## Data loading
file_data_path = "./data/sample.xyz"
point_cloud = np.loadtxt(file_data_path, skiprows=1, max_rows=1000000)
point_cloud
## Point cloud has 6 attributes:  X, Y, Z, R, G, B.
## Extracting attributes
xyz = point_cloud[:, :3]
rgb = point_cloud[:, 3:]
mean_z = np.mean(point_cloud, axis=0)[2]
spatial_query = point_cloud[
    abs(point_cloud[:, 2] - mean_z) < 1
]  # height in Z within 1m of mean_z

## Create 3D visualisation
ax = plt.axes(projection="3d")
ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=rgb / 255, s=0.01)
plt.show()
