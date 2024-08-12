from __future__ import print_function

import os
import cv2
import glob
import yaml
import argparse
import numpy as np
import matplotlib.pyplot as plt

from scipy.sparse import lil_matrix
from scipy.optimize import least_squares


DIR_INPUT = './keypoints_3d'
DIR_STORE = './keypoints_3d_ba'


def _get_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-c', '--config',
        help='Path to the config file',
        type=str,
        default='configs/skeleton_bundle_adjustment.yml',
    )

    args = parser.parse_args()

    return args


def _load_configs(path):
    with open(path, 'r') as yaml_file:
        configs = yaml.safe_load(yaml_file)

    return configs


def project(points, params):
    camera_matrix = np.zeros((params.shape[0], 3, 3), dtype=float)
    camera_matrix[:, 0, 0] = params[:, 12]
    camera_matrix[:, 1, 1] = params[:, 13]
    camera_matrix[:, 0, 2] = params[:, 14]
    camera_matrix[:, 1, 2] = params[:, 15]
    camera_matrix[:, 2, 2] = 1.0
    rotation = params[:, :9].reshape(-1, 3, 3)
    translation = params[:, 9:12]

    points_proj = []
    for cam_idx in range(params.shape[0]):
        points_proj.append(
            cv2.projectPoints(
                points, rotation[cam_idx], translation[cam_idx],
                camera_matrix[cam_idx],
                None)[0]
        )

    return points_proj


def fun(params, n_cameras, n_depth, n_points, points_2d, configs):
    camera_params = params[:n_cameras * configs['calibration_parameters_size']].reshape(
        (n_cameras, configs['calibration_parameters_size']))
    points_3d = params[n_cameras * configs['calibration_parameters_size']:].reshape((n_points, 3))
    
    points_proj = project(
        points_3d,
        camera_params)
    
    error = []
    for cam_idx, camera in enumerate(points_2d.keys()):
        points_proj_cam = points_proj[cam_idx].reshape(n_depth, -1, 26, 2)
        points_2d_cam = np.array(points_2d[camera]['pose'])

        diff = points_proj_cam[:, :points_2d_cam.shape[1], :, :] - points_2d_cam
        diff = np.array(diff).ravel()
        error.extend(diff)

    error = np.array(error)

    return error


def bundle_adjustment_sparsity(n_cameras, n_depth, n_points, camera_indices, point_indices, configs):
    m = camera_indices.size * 2
    n = n_cameras * configs['calibration_parameters_size'] + n_points * 3
    A = lil_matrix((m, n), dtype=int)

    print("m & n", m, n)

    i = np.arange(camera_indices.size)
    for s in range(configs['calibration_parameters_size']):
        A[2 * i, camera_indices * configs['calibration_parameters_size'] + s] = 1
        A[2 * i + 1, camera_indices * configs['calibration_parameters_size'] + s] = 1

    for s in range(3):
        A[2 * i, n_cameras * configs['calibration_parameters_size'] + point_indices * 3 + s] = 1
        A[2 * i + 1, n_cameras * configs['calibration_parameters_size'] + point_indices * 3 + s] = 1

    return A


def ravel(params, configs):
    params_ravel = np.zeros(
        (len(list(params.keys())),
         configs['calibration_parameters_size']),
        dtype=float)

    for i, camera in enumerate(params.keys()):
        params_ravel[i, :9] = np.array(params[camera]['rotation'], np.float32).ravel()
        params_ravel[i, 9:12] = np.array(params[camera]['translation'], np.float32).ravel()
        params_ravel[i, 12] = params[camera]['mtx'][0][0]
        params_ravel[i, 13] = params[camera]['mtx'][1][1]
        params_ravel[i, 14] = params[camera]['mtx'][0][2]
        params_ravel[i, 15] = params[camera]['mtx'][1][2]
        params_ravel[i, 16:] = np.array(params[camera]['dist'], np.float32).ravel()

    return params_ravel.ravel()


def visualize_error(x0, n_cameras, n_depth, n_points, points_2d, configs):
    f0 = fun(x0, n_cameras, n_depth, n_points, points_2d, configs)
    
    plt.plot(f0)
    plt.show()


def optimize(n_cameras, n_depth, n_points, poses_2d, camera_indices, point_indices, configs):
    jac_sparsity = bundle_adjustment_sparsity(
        n_cameras, n_depth, n_points, camera_indices, point_indices, configs)

    res = least_squares(
        fun, x0, jac_sparsity=jac_sparsity, verbose=2,
        x_scale='jac', ftol=1e-4, method='trf',
        args=(n_cameras, n_depth, n_points, poses_2d, configs))

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


def _ravel_params(params):
    params_ravel = []
    for camera in params.keys():
        mtx = np.array(params[camera]['mtx'], np.float32)
        dist = np.array(params[camera]['dist'], np.float32)
        rotation = np.array(params[camera]['rotation'], np.float32)
        translation = np.array(params[camera]['translation'], np.float32)

        params_ravel.append(np.concatenate(
            (mtx.ravel(),
             dist.ravel(),
             rotation.ravel(),
             translation.ravel())
        ))

    params_ravel = np.concatenate(params_ravel)

    return params_ravel


if __name__ == '__main__':
    args = _get_arguments()
    configs = _load_configs(args.config)

    print(f"Config loaded: {configs}")

    with open(configs['skeletons_2d']) as handler:
        poses_2d = yaml.safe_load(handler)

    with open(configs['skeletons_3d']) as handler:
        poses_3d = np.array(yaml.safe_load(handler), np.float32)

    with open(configs['params']) as handler:
        params = yaml.safe_load(handler)

    n_cameras = len(list(params.keys()))
    n_depth = poses_3d.shape[0]
    n_points = poses_3d.reshape(-1, 3).shape[0]
    n_points_2d = int(np.sum(
        [len(np.array(poses_2d[camera]['pose']).ravel()) / 2
         for camera in poses_2d.keys()]
    ))

    print([len(np.array(poses_2d[camera]['pose']).ravel()) for camera in poses_2d.keys()])
    camera_indices = [[cam_idx] * (len(np.array(poses_2d[camera]['pose']).ravel()) // 2)
                      for cam_idx, camera in enumerate(poses_2d.keys())]
    camera_indices = np.concatenate(camera_indices)

    point_indices = []
    for camera in poses_2d.keys():
        point_indices_cam = []
        for t in range(len(poses_2d[camera]['pose'])):
            num_points = np.array(poses_2d[camera]['pose'][t]).shape[:2]
            point_indices_cam.append(np.arange(num_points[0] * num_points[1]))

        print(np.array(point_indices_cam).shape)

        point_indices.append(np.array(point_indices_cam).ravel())

    point_indices = np.concatenate(point_indices)

    print("camera_indices", camera_indices.shape)
    print("point_indices", point_indices.shape)

    # # n = configs['calibration_parameters_size'] * n_cameras + 3 * n_points
    # # m = 2 * len(poses_2d)

    params_ravel = ravel(params, configs)
    # print(params_ravel.shape)

    # Visualizing initial error
    # x0 = np.hstack((params_ravel, poses_3d.ravel()))
    x0 = np.hstack((params_ravel, poses_3d.ravel()))
    # x0 = np.hstack((points_3d.ravel(),))

    visualize_error(x0, n_cameras, n_depth, n_points, poses_2d, configs)

    results = optimize(n_cameras, n_depth, n_points, poses_2d, camera_indices, point_indices, configs)
    
    # # store_results(results, experiment, n_cameras, camera_params_org)

    visualize_error(results['x'], n_cameras, n_depth, n_points, poses_2d, configs)
