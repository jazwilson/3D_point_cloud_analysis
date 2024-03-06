# 3D Point Cloud Feature Extraction 
## Based on guide by Prof. Florent Poux 
## Guide: https://learngeodata.eu/point-cloud-feature-extraction-complete-guide/ 
## Data source: https://drive.google.com/drive/folders/1fwhE5OphpeW4RR0RY8W2jbqmlf5LH6dX 



# %% Import libraries

import numpy as np
from scipy.spatial import KDTree
import pyvista as pv

# %% Load and view data

pcd_pv = pv.read("data/MLS_UTWENTE_super_sample.ply")  ## pcd_pv x,y,z, n_points

## Access points
# pcd_pv.points

## plot quickly with EDL
pcd_pv.plot(eye_dome_lighting=True)


# %% test plotting specific features
pcd_pv["elevation"] = pcd_pv.points[:, 2]  # x,y,z, n_points
## render as sphere
# pv.plot(pcd_pv, scalars = pcd_pv['elevation'], point_size = 5, show_scalar_bar = False) #scalars = display colour info

# %% Define PCA computation function

sel = 1


def PCA(cloud):
    x = pcd_pv.points  # Compute PCA for subcloud
    mean = np.mean(x, axis=0)  # Compute mean of data
    centered_data = x - mean  # Center the data by subtracting mean
    cov_matrix = np.cov(centered_data, rowvar=False)  # compute covariance matrix
    eigen_values, eigen_vectors = np.linalg.eig(
        cov_matrix
    )  # Calc eigenvalues and eigenvectors
    sorted_index = np.argsort(eigen_values)[
        ::-1
    ]  # Srt eigenvectors by decreasing eigenvalues
    sorted_eigenvalue = eigen_values[sorted_index]
    sorted_vectors = eigen_vectors[:, sorted_index]

    return sorted_eigenvalue, sorted_vectors


# %% Define PCA featyrubg
def pca_featuring(val, vec):
    planarity = (val[1] - val[2]) / val[0]
    linearity = (val[0] - val[1]) / val[2]
    omnivariance = (val[0] * val[1] * val[2]) ** (1 / 3)
    normal = vec[:, 2]
    verticality = 1 - normal[2]

    return (
        planarity,
        linearity,
        omnivariance,
        verticality,
        normal[0],
        normal[1],
        normal[2],
    )


# %% test

# Build a KD-tree obj for each point
tree = KDTree(pcd_pv.points)
# query for each point, find 20 of the closest points
dists, indices = tree.query(pcd_pv.points, k=20)  # alt to use radius "query_ball_point"
# Get neighbor points for each point
neighbors = pcd_pv.points[indices]

# Compute PCA for a neighbor point cloud
f_val, f_vec = PCA(neighbors[1])

# Compute PCA-based features
planarity, linearity, omnivariance, verticality, nx, ny, nz = pca_featuring(
    f_val, f_vec
)
print("Planarity:", planarity)
print("Linearity:", linearity)
print("Omnivariance:", omnivariance)
print("Verticality:", verticality)
print("Normal X:", nx)
print("Normal Y:", ny)
print("Normal Z:", nz)

# %% Alternative to KDT query searches: Radius search (high memory requirements)

tree_temp = KDTree(pcd_pv.points)
idx_temp = tree_temp.query_ball_point(pcd_pv.points, 1)

# %% Alternative to KDT query searches:  Knowledge driven custom search (high memory requirements)
## drop Z, if interested in local maxima / minima

tree_2d = KDTree(pcd_pv.points[:, 0:2])
idx_2D_rad = tree_2d.query_ball_point(pcd_pv.point_cell_ids[:, 0:2], 1)
# %% Point cloud feature extraction: Relative featuring

sel = 1
selection = pcd_pv.points[idx_2D_rad[sel]]

## compute dist from lowest point to highest
d_high = np.array(np.max(selection, axis=0) - pcd_pv.points[sel])(2)
d_low = np.array(pcd_pv.points[sel] - np.min(selection, axis=0))[2]
