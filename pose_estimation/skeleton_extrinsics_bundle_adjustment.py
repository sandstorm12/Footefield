from __future__ import print_function

import urllib
import urllib.request
import bz2
import os
import numpy as np


BASE_URL = "http://grail.cs.washington.edu/projects/bal/data/ladybug/"
FILE_NAME = "problem-49-7776-pre.txt.bz2"
URL = BASE_URL + FILE_NAME
if not os.path.isfile(FILE_NAME):
    urllib.request.urlretrieve(URL, FILE_NAME)

def read_bal_data(file_name):
    with bz2.open(file_name, "rt") as file:
        n_cameras, n_points, n_observations = map(
            int, file.readline().split())

        camera_indices = np.empty(n_observations, dtype=int)
        point_indices = np.empty(n_observations, dtype=int)
        points_2d = np.empty((n_observations, 2))

        for i in range(n_observations):
            camera_index, point_index, x, y = file.readline().split()
            camera_indices[i] = int(camera_index)
            point_indices[i] = int(point_index)
            points_2d[i] = [float(x), float(y)]

        camera_params = np.empty(n_cameras * 9)
        for i in range(n_cameras * 9):
            camera_params[i] = float(file.readline())
        camera_params = camera_params.reshape((n_cameras, -1))

        points_3d = np.empty(n_points * 3)
        for i in range(n_points * 3):
            points_3d[i] = float(file.readline())
        points_3d = points_3d.reshape((n_points, -1))

    return camera_params, points_3d, camera_indices, point_indices, points_2d


def read_ba_data():
    import pickle
    with open('/home/hamid/Documents/phd/footefield/footefield/pose_estimation/keypoints_3d_ba/ba_a1.pkl', 'rb') as handle:
        ba_data = pickle.load(handle)

    return ba_data['params'], ba_data['poses_3d'], ba_data['camera_indices'], ba_data['point_indices'], ba_data['points_2d']


# camera_params, points_3d, camera_indices, point_indices, points_2d = read_bal_data(FILE_NAME)
camera_params, points_3d, camera_indices, point_indices, points_2d = read_ba_data()

print("camera_params", camera_params[0].keys())
print("points_3d", points_3d.shape)
print("camera_indices", camera_indices.shape)
print("point_indices", point_indices.shape)
print("points_2d", points_2d.shape)
print(point_indices)


n_cameras = len(camera_params)
n_points = points_3d.shape[0]

n = 9 * n_cameras + 3 * n_points
m = 2 * points_2d.shape[0]

print("n_cameras: {}".format(n_cameras))
print("n_points: {}".format(n_points))
print("Total number of parameters: {}".format(n))
print("Total number of residuals: {}".format(m))


def rotate(points, rot_vecs):
    """Rotate points by given rotation vectors.
    
    Rodrigues' rotation formula is used.
    """
    theta = np.linalg.norm(rot_vecs, axis=1)[:, np.newaxis]
    with np.errstate(invalid='ignore'):
        v = rot_vecs / theta
        v = np.nan_to_num(v)
    dot = np.sum(points * v, axis=1)[:, np.newaxis]
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    return cos_theta * points + sin_theta * np.cross(v, points) + dot * (1 - cos_theta) * v


# def project(points, camera_params):
#     """Convert 3-D points to 2-D by projecting onto images."""
#     points_proj = rotate(points, camera_params[:, :3])
#     points_proj += camera_params[:, 3:6]
#     points_proj = -points_proj[:, :2] / points_proj[:, 2, np.newaxis]
#     f = camera_params[:, 6]
#     k1 = camera_params[:, 7]
#     k2 = camera_params[:, 8]
#     n = np.sum(points_proj**2, axis=1)
#     r = 1 + k1 * n + k2 * n**2
#     points_proj *= (r * f)[:, np.newaxis]
#     return points_proj


def project(points, params):
    """Convert 3-D points to 2-D by projecting onto images using cv2 calibration data."""

    print(params)
    camera_matrix = params[12:16]
    dist_coeffs = params[16:]
    rotation = params[:9].reshape(3, 3)
    translation = params[9:12]

    # Extract camera parameters
    fx = camera_matrix[0, 0]
    fy = camera_matrix[1, 1]
    cx = camera_matrix[0, 2]
    cy = camera_matrix[1, 2]
    k1, k2, p1, p2, k3 = dist_coeffs[:5]  

    # Apply rotation and translation 
    points_proj = np.dot(points, rotation.T) + translation

    # Normalize by homogeneous coordinates
    points_proj = points_proj[:, :2] / points_proj[:, 2:]

    # Apply radial distortion
    r2 = np.sum(points_proj**2, axis=1)
    r4 = r2**2 
    radial_distortion = 1 + k1 * r2 + k2 * r4 + k3 * r2**3
    points_proj *= radial_distortion[:, np.newaxis]

    # Apply tangential distortion
    x, y = points_proj[:, 0], points_proj[:, 1]
    x_tangential = 2 * p1 * x * y + p2 * (r2 + 2 * x**2) 
    y_tangential = p1 * (r2 + 2 * y**2) + 2 * p2 * x * y
    points_proj[:, 0] = x + x_tangential
    points_proj[:, 1] = y + y_tangential

    return points_proj


def fun(params, n_cameras, n_points, camera_indices, point_indices, points_2d):
    """Compute residuals.
    
    `params` contains camera parameters and 3-D coordinates.
    """
    camera_params = params[:n_cameras * 21].reshape((n_cameras, 21))
    points_3d = params[n_cameras * 21:].reshape((n_points, 3))
    print(camera_indices)
    points_proj = project(points_3d[point_indices], camera_params[camera_indices])
    return (points_proj - points_2d).ravel()


from scipy.sparse import lil_matrix


def bundle_adjustment_sparsity(n_cameras, n_points, camera_indices, point_indices):
    m = camera_indices.size * 2
    n = n_cameras * 9 + n_points * 3
    A = lil_matrix((m, n), dtype=int)

    i = np.arange(camera_indices.size)
    for s in range(9):
        A[2 * i, camera_indices * 9 + s] = 1
        A[2 * i + 1, camera_indices * 9 + s] = 1

    for s in range(3):
        A[2 * i, n_cameras * 9 + point_indices * 3 + s] = 1
        A[2 * i + 1, n_cameras * 9 + point_indices * 3 + s] = 1

    return A


def ravel(camera_params):
    cp_ravel = np.empty((len(camera_params), 21), dtype=float)

    for i in range(len(camera_params)):
        cp_ravel[i, :9] = camera_params[i]['extrinsics'][:3, :3].ravel()
        cp_ravel[i, 9:12] = camera_params[i]['extrinsics'][:3, 3].ravel()
        cp_ravel[i, 12] = camera_params[i]['mtx'][0, 0]
        cp_ravel[i, 13] = camera_params[i]['mtx'][1, 1]
        cp_ravel[i, 14] = camera_params[i]['mtx'][0, 2]
        cp_ravel[i, 15] = camera_params[i]['mtx'][1, 2]
        cp_ravel[i, 16:] = camera_params[i]['dist'].ravel()

    return cp_ravel.ravel()


import matplotlib.pyplot as plt
x0 = np.hstack((ravel(camera_params), points_3d.ravel()))
f0 = fun(x0, n_cameras, n_points, camera_indices, point_indices, points_2d)
plt.plot(f0)
plt.show()


A = bundle_adjustment_sparsity(n_cameras, n_points, camera_indices, point_indices)
import time
from scipy.optimize import least_squares
t0 = time.time()
res = least_squares(fun, x0, jac_sparsity=A, verbose=2, x_scale='jac', ftol=1e-4, method='trf',
                    args=(n_cameras, n_points, camera_indices, point_indices, points_2d))
points_3d = res['x'][36:]
t1 = time.time()

print("Optimization took {0:.0f} seconds".format(t1 - t0))

print(res)  
print("points_3d", points_3d.shape)
import pickle
with open('points_3d.pkl', 'wb') as handle:
    pickle.dump(points_3d, handle, protocol=pickle.HIGHEST_PROTOCOL)

plt.plot(res.fun)
plt.show()