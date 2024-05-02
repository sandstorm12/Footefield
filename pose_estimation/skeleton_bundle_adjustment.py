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
PARAM_CALIB_SIZE = 21
PARAM_CORRECT_DISTORTION = False


def read_ba_data(path):
    with open(path, 'rb') as handle:
        ba_data = pickle.load(handle)

    return ba_data['params'], \
        ba_data['poses_3d'], \
        ba_data['camera_indices'], \
        ba_data['point_indices'], \
        ba_data['points_2d'], \
        ba_data['points_2d_confidence']


def project(points, params, params_org):
    camera_matrix = np.zeros((params.shape[0], 3, 3), dtype=float)
    camera_matrix[:, 0, 0] = np.array([item['mtx'][0, 0] for item in params_org])
    camera_matrix[:, 1, 1] = np.array([item['mtx'][1, 1] for item in params_org])
    camera_matrix[:, 0, 2] = np.array([item['mtx'][0, 2] for item in params_org])
    camera_matrix[:, 1, 2] = np.array([item['mtx'][1, 2] for item in params_org])
    # camera_matrix[:, 0, 0] = params[:, 12]
    # camera_matrix[:, 1, 1] = params[:, 13]
    # camera_matrix[:, 0, 2] = params[:, 14]
    # camera_matrix[:, 1, 2] = params[:, 15]
    camera_matrix[:, 2, 2] = 1.0
    dist_coeffs = np.array([item['dist'] for item in params_org])
    # dist_coeffs = params[:, 16:]
    # rotation = np.array([item['extrinsics'][:3, :3] for item in params_org])
    rotation = params[:, :9].reshape(-1, 3, 3)
    # translation = np.array([item['extrinsics'][:3, 3] for item in params_org])
    translation = params[:, 9:12]

    points_proj = []
    for idx, point in enumerate(points):
        points_proj.append(
            cv2.projectPoints(
                point, rotation[idx], translation[idx],
                camera_matrix[idx],
                dist_coeffs[idx] if PARAM_CORRECT_DISTORTION else None)[0]
        )

    points_proj = np.asarray(points_proj).squeeze()

    return points_proj


def fun(params, n_cameras, n_points, camera_indices, point_indices,
        points_2d, points_2d_confidence):
    camera_params = params[:n_cameras * PARAM_CALIB_SIZE].reshape(
        (n_cameras, PARAM_CALIB_SIZE))
    points_3d = params[n_cameras * PARAM_CALIB_SIZE:].reshape((n_points, 3))
    points_proj = project(
        points_3d[point_indices],
        camera_params[camera_indices],
        [camera_params_org[idx] for idx in camera_indices])
    
    points_2d_confidence = np.tile(np.expand_dims(
        points_2d_confidence, 1), (1, 2))
    points_2d_confidence[points_2d_confidence < .3] = 0

    return ((points_proj - points_2d) * points_2d_confidence).ravel()


def bundle_adjustment_sparsity(n_cameras, n_points, camera_indices,
                               point_indices):
    m = camera_indices.size * 2
    n = n_cameras * PARAM_CALIB_SIZE + n_points * 3
    A = lil_matrix((m, n), dtype=int)

    i = np.arange(camera_indices.size)
    for s in range(PARAM_CALIB_SIZE):
        A[2 * i, camera_indices * PARAM_CALIB_SIZE + s] = 1
        A[2 * i + 1, camera_indices * PARAM_CALIB_SIZE + s] = 1

    for s in range(3):
        A[2 * i, n_cameras * PARAM_CALIB_SIZE + point_indices * 3 + s] = 1
        A[2 * i + 1, n_cameras * PARAM_CALIB_SIZE + point_indices * 3 + s] = 1

    return A


def ravel(camera_params):
    cp_ravel = np.zeros((len(camera_params), PARAM_CALIB_SIZE), dtype=float)

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
                    points_2d, points_2d_confidence):
    f0 = fun(x0, n_cameras, n_points, camera_indices, point_indices,
             points_2d, points_2d_confidence)
    plt.plot(f0)
    plt.show()


def optimize(n_cameras, n_points, camera_indices, point_indices,
             points_2d, points_2d_confidence):
    jac_sparsity = bundle_adjustment_sparsity(
        n_cameras, n_points, camera_indices, point_indices)

    res = least_squares(
        fun, x0, jac_sparsity=jac_sparsity, verbose=2,
        x_scale='jac', ftol=1e-4, method='trf',
        args=(n_cameras, n_points, camera_indices,
              point_indices, points_2d, points_2d_confidence))

    return res


def reconstruct_params(results, n_cameras, camera_params_org):
    calib_params_size = n_cameras * PARAM_CALIB_SIZE
    params = results['x'][:calib_params_size].reshape(n_cameras, -1)

    params_reconstructed = []
    for idx in range(n_cameras):
        camera_matrix = np.zeros((3, 3), dtype=float)
        camera_matrix[0, 0] = camera_params_org[idx]['mtx'][0, 0]
        camera_matrix[1, 1] = camera_params_org[idx]['mtx'][1, 1]
        camera_matrix[0, 2] = camera_params_org[idx]['mtx'][0, 2]
        camera_matrix[1, 2] = camera_params_org[idx]['mtx'][1, 2]
        # camera_matrix[0, 0] = params[idx, 12]
        # camera_matrix[1, 1] = params[idx, 13]
        # camera_matrix[0, 2] = params[idx, 14]
        # camera_matrix[1, 2] = params[idx, 15]
        camera_matrix[2, 2] = 1.0
        dist_coeffs = camera_params_org[idx]['dist']
        # dist_coeffs = params[idx, 16:]
        # rotation = camera_params_org[idx]['extrinsics'][:3, :3]
        rotation = params[idx, :9].reshape(3, 3)
        # translation = camera_params_org[idx]['extrinsics'][:3, 3]
        translation = params[idx, 9:12]

        params_reconstructed.append(
            {
                'mtx': camera_matrix,
                'dist': dist_coeffs,
                'rotation': rotation,
                'translation': translation,
            }
        )

    return params_reconstructed


def store_results(results, experiment, n_cameras, camera_params_org):
    if not os.path.exists(DIR_STORE):
        os.mkdir(DIR_STORE)

    calib_params_size = n_cameras * PARAM_CALIB_SIZE

    params = reconstruct_params(results, n_cameras, camera_params_org)

    output = {
        'params': params,
        'points_3d': results['x'][calib_params_size:]
    }

    path = os.path.join(DIR_STORE, f'keypoints3d_{experiment}_ba.pkl')

    with open(path, 'wb') as handle:
        pickle.dump(output, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    for name in glob.glob(os.path.join(DIR_INPUT, '*_ba.pkl')):
        experiment = name.split('.')[-2].split('_')[-2]
        camera_params_org, \
            points_3d, \
            camera_indices, \
            point_indices, \
            points_2d, \
            points_2d_confidence = read_ba_data(name)

        n_cameras = len(camera_params_org)
        n_points = points_3d.shape[0]

        n = PARAM_CALIB_SIZE * n_cameras + 3 * n_points
        m = 2 * points_2d.shape[0]

        # Visualizing initial error
        x0 = np.hstack((ravel(camera_params_org), points_3d.ravel()))
        # x0 = np.hstack((points_3d.ravel(),))

        visualize_error(x0, n_cameras, n_points, camera_indices,
                        point_indices, points_2d, points_2d_confidence)

        # results = {'x': x0}
        results = optimize(n_cameras, n_points, camera_indices,
                       point_indices, points_2d, points_2d_confidence)
        
        store_results(results, experiment, n_cameras, camera_params_org)

        visualize_error(results['x'], n_cameras, n_points, camera_indices,
                        point_indices, points_2d, points_2d_confidence)
