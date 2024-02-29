from __future__ import print_function

import os
import cv2
import glob
import pickle
import numpy as np
import matplotlib.pyplot as plt

from scipy.sparse import lil_matrix
from scipy.optimize import least_squares


DIR_INPUT = './keypoints_3d'
DIR_STORE = './keypoints_3d_ba'


def read_ba_data(path):
    with open(path, 'rb') as handle:
        ba_data = pickle.load(handle)

    return ba_data['params'], \
        ba_data['poses_3d'], \
        ba_data['camera_indices'], \
        ba_data['point_indices'], \
        ba_data['points_2d']


def project(points, params):
    """Convert 3-D points to 2-D by projecting onto images using
    cv2 calibration data.
    """

    camera_matrix = np.zeros((params.shape[0], 3, 3), dtype=float)
    camera_matrix[:, 0, 0] = params[:, 12]
    camera_matrix[:, 1, 1] = params[:, 13]
    camera_matrix[:, 0, 2] = params[:, 14]
    camera_matrix[:, 1, 2] = params[:, 15]
    dist_coeffs = params[:, 16:]
    rotation = params[:, :9].reshape(-1, 3, 3)
    translation = params[:, 9:12]

    points_proj = []
    for idx, point in enumerate(points):
        points_proj.append(
            cv2.projectPoints(
                point, rotation[idx], translation[idx],
                camera_matrix[idx], dist_coeffs[idx])[0]
        )

    points_proj = np.asarray(points_proj).squeeze()

    return points_proj


def fun(params, n_cameras, n_points, camera_indices, point_indices,
        points_2d):
    camera_params = params[:n_cameras * 21].reshape((n_cameras, 21))
    points_3d = params[n_cameras * 21:].reshape((n_points, 3))
    points_proj = project(
        points_3d[point_indices], camera_params[camera_indices])

    return (points_proj - points_2d).ravel()


def bundle_adjustment_sparsity(n_cameras, n_points, camera_indices,
                               point_indices):
    m = camera_indices.size * 2
    n = n_cameras * 21 + n_points * 3
    A = lil_matrix((m, n), dtype=int)

    i = np.arange(camera_indices.size)
    for s in range(21):
        A[2 * i, camera_indices * 21 + s] = 1
        A[2 * i + 1, camera_indices * 21 + s] = 1

    for s in range(3):
        A[2 * i, n_cameras * 21 + point_indices * 3 + s] = 1
        A[2 * i + 1, n_cameras * 21 + point_indices * 3 + s] = 1

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


def visualize_error(x0, n_cameras, n_points, camera_indices, point_indices,
                    points_2d):
    f0 = fun(x0, n_cameras, n_points, camera_indices, point_indices,
             points_2d)
    plt.plot(f0)
    plt.show()


def optimize(n_cameras, n_points, camera_indices, point_indices,
             points_2d):
    jac_sparsity = bundle_adjustment_sparsity(
        n_cameras, n_points, camera_indices, point_indices)

    res = least_squares(
        fun, x0, jac_sparsity=jac_sparsity, verbose=2,
        x_scale='jac', ftol=1e-4, method='trf',
        args=(n_cameras, n_points, camera_indices,
              point_indices, points_2d))

    return res


def store_results(results, experiment):
    if not os.path.exists(DIR_STORE):
        os.mkdir(DIR_STORE)

    output = {
        'params': results['x'][:84],
        'points_3d': results['x'][84:]
    }

    path = os.path.join(DIR_STORE, f'keypoints3d_{experiment}_ba.pkl')

    with open(path, 'wb') as handle:
        pickle.dump(output, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    for name in glob.glob(os.path.join(DIR_INPUT, '*_ba.pkl')):
        experiment = name.split('.')[-2].split('_')[-2]
        camera_params, \
            points_3d, \
            camera_indices, \
            point_indices, \
            points_2d = read_ba_data(name)

        n_cameras = len(camera_params)
        n_points = points_3d.shape[0]

        n = 21 * n_cameras + 3 * n_points
        m = 2 * points_2d.shape[0]

        # Visualizing initial error error
        x0 = np.hstack((ravel(camera_params), points_3d.ravel()))

        visualize_error(x0, n_cameras, n_points, camera_indices,
                        point_indices, points_2d)

        results = optimize(n_cameras, n_points, camera_indices,
                       point_indices, points_2d)
        
        store_results(results, experiment)

        visualize_error(results['x'], n_cameras, n_points, camera_indices,
                        point_indices, points_2d)
