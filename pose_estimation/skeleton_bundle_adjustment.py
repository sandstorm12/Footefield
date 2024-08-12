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


def project(points, params, params_org):
    rotation = params[:, :9].reshape(-1, 3, 3)
    translation = params[:, 9:12]

    points_proj = []
    for cam_idx, camera in enumerate(params_org):
        points_proj.append(
            cv2.projectPoints(
                points, rotation[cam_idx], translation[cam_idx],
                np.array(params_org[camera]['mtx'], np.float32),
                None)[0]
        )

    return points_proj


def fun(params, n_cameras, n_depth, n_points, points_2d, params_org, configs):
    camera_params = params[:n_cameras * configs['calib_param_size']].reshape(
        (n_cameras, configs['calib_param_size']))
    points_3d = params[n_cameras * configs['calib_param_size']:].reshape((n_points, 3))
    
    points_proj = project(
        points_3d,
        camera_params,
        params_org)
    
    error = []
    for cam_idx, camera in enumerate(points_2d.keys()):
        points_proj_cam = points_proj[cam_idx].reshape(n_depth, -1, 26, 2)
        points_2d_cam = np.array(points_2d[camera]['pose'])
        points_2d_conf_cam = np.array(points_2d[camera]['pose_confidence'])

        # print(points_2d_conf_cam.shape)
        # print(np.sum(points_2d_conf_cam < configs['conf_threshold']))

        diff = points_proj_cam[:, :points_2d_cam.shape[1], :, :] - points_2d_cam
        # diff[:, :, :, 0] *= points_2d_conf_cam ** 2
        # diff[:, :, :, 1] *= points_2d_conf_cam ** 2
        diff[points_2d_conf_cam < configs['conf_threshold']] = 0
        diff = np.array(diff).ravel()
        error.extend(diff)

    error = np.array(error)

    return error


def bundle_adjustment_sparsity(n_cameras, n_depth, n_points, camera_indices,
                               point_indices, configs):
    m = camera_indices.size * 2
    n = n_cameras * configs['calib_param_size'] + n_points * 3
    A = lil_matrix((m, n), dtype=int)

    i = np.arange(camera_indices.size)
    for s in range(configs['calib_param_size']):
        A[2 * i, camera_indices * configs['calib_param_size'] + s] = 1
        A[2 * i + 1, camera_indices * configs['calib_param_size'] + s] = 1

    for s in range(3):
        A[2 * i, n_cameras * configs['calib_param_size'] + point_indices * 3 + s] = 1
        A[2 * i + 1, n_cameras * configs['calib_param_size'] + point_indices * 3 + s] = 1

    return A


def ravel(params, configs):
    params_ravel = np.zeros(
        (len(list(params.keys())),
         configs['calib_param_size']),
        dtype=float)

    for i, camera in enumerate(params.keys()):
        params_ravel[i, :9] = np.array(params[camera]['rotation'], np.float32).ravel()
        params_ravel[i, 9:12] = np.array(params[camera]['translation'], np.float32).ravel()

    return params_ravel.ravel()


def visualize_error(x0, n_cameras, n_depth, n_points, points_2d, param_org, configs):
    f0 = fun(x0, n_cameras, n_depth, n_points, points_2d, param_org, configs)
    
    plt.plot(f0)
    plt.show()


def optimize(n_cameras, n_depth, n_points, poses_2d, camera_indices,
             point_indices, params_org, configs):
    jac_sparsity = bundle_adjustment_sparsity(
        n_cameras, n_depth, n_points, camera_indices, point_indices, configs)

    res = least_squares(
        fun, x0, jac_sparsity=jac_sparsity, verbose=2,
        x_scale='jac', ftol=1e-12, xtol=1e-20, gtol=1e-12,
        method='trf', args=(n_cameras, n_depth, n_points, poses_2d, params_org,
                            configs))

    return res


def reconstruct_params(results, cameras, params_org, configs):
    n_cameras = len(cameras)

    calib_params_size = n_cameras * configs['calib_param_size']
    params = results['x'][:calib_params_size].reshape(n_cameras, -1)

    params_reconstructed = {}
    for idx, camera in enumerate(cameras):
        camera_matrix = params_org[camera]['mtx']
        dist = params_org[camera]['dist']
        rotation = params[idx, :9].reshape(3, 3)
        translation = params[idx, 9:12]

        params_reconstructed[camera] = {
            'mtx': camera_matrix,
            'dist': dist,
            'rotation': rotation.tolist(),
            'translation': translation.tolist(),
        }

    return params_reconstructed


def store_results(results, poses_2d, poses_3d, params_org, configs):
    cameras = list(poses_2d.keys())
    
    params = reconstruct_params(
        results,
        cameras,
        params_org,
        configs)

    _store_artifacts(params, configs['output_params'])

    calib_params_size = n_cameras * configs['calib_param_size']
    points_3d = results['x'][calib_params_size:]
    points_3d = points_3d.reshape(poses_3d.shape).tolist()

    _store_artifacts(points_3d, configs['output_skeleton_3d'])


def _store_artifacts(artifact, output):
    with open(output, 'w') as handle:
        yaml.dump(artifact, handle)


def _load_inputs(configs):
    with open(configs['skeletons_2d']) as handler:
        poses_2d = yaml.safe_load(handler)

    with open(configs['skeletons_3d']) as handler:
        poses_3d = np.array(yaml.safe_load(handler), np.float32)

    with open(configs['params']) as handler:
        params = yaml.safe_load(handler)

    return poses_2d, poses_3d, params


def _get_camera_indices(poses_2d):
    camera_indices = [
        [cam_idx] * (len(np.array(poses_2d[camera]['pose']).ravel()) // 2)
        for cam_idx, camera in enumerate(poses_2d.keys())]
    camera_indices = np.concatenate(camera_indices)

    return camera_indices


def _get_point_indices(poses_2d):
    point_indices = []
    for camera in poses_2d.keys():
        point_indices_cam = []
        for t in range(len(poses_2d[camera]['pose'])):
            num_points = np.array(poses_2d[camera]['pose'][t]).shape[:2]
            point_indices_cam.append(np.arange(num_points[0] * num_points[1]))

        point_indices.append(np.array(point_indices_cam).ravel())

    point_indices = np.concatenate(point_indices)

    return point_indices


def _contruct_optimization_params(params, poses_3d, configs):
    params_ravel = ravel(params, configs)
    
    x0 = np.hstack((params_ravel, poses_3d.ravel()))

    return x0


if __name__ == '__main__':
    args = _get_arguments()
    configs = _load_configs(args.config)

    print(f"Config loaded: {configs}")

    poses_2d, poses_3d, params_org = _load_inputs(configs)

    n_cameras = len(list(params_org.keys()))
    n_depth = poses_3d.shape[0]
    n_points = poses_3d.reshape(-1, 3).shape[0]

    camera_indices = _get_camera_indices(poses_2d)
    point_indices = _get_point_indices(poses_2d)

    x0 = _contruct_optimization_params(params_org, poses_3d, configs)

    visualize_error(x0, n_cameras, n_depth, n_points, poses_2d,
                    params_org, configs)

    results = optimize(n_cameras, n_depth, n_points, poses_2d, camera_indices,
                       point_indices, params_org, configs)

    visualize_error(results['x'], n_cameras, n_depth, n_points, poses_2d,
                    params_org, configs)

    store_results(results, poses_2d, poses_3d, params_org, configs)
