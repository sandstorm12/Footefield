import cv2
import yaml
import argparse
import numpy as np
import matplotlib.pyplot as plt

from scipy.sparse import lil_matrix
from scipy.optimize import least_squares


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


def project(points, points_indices,
            points_cam_indices, params, params_org):
    # camera_matrix = np.zeros((params.shape[0], 3, 3), dtype=np.float32)
    # camera_matrix[:, 0, 0] = params[:, 12]
    # camera_matrix[:, 1, 1] = params[:, 13]
    # camera_matrix[:, 0, 2] = params[:, 14]
    # camera_matrix[:, 1, 2] = params[:, 15]
    # camera_matrix[:, 2, 2] = 1.0
    # dist = params[:, 16:]

    rotation = params[:, :9].reshape(-1, 3, 3)
    translation = params[:, 9:12]

    cameras = list(params_org.keys())

    points_proj = []
    for point_idx, point_global_idx in enumerate(points_indices):
        points_proj.append(
            cv2.projectPoints(
                points[point_global_idx],
                rotation[points_cam_indices[point_idx]],
                translation[points_cam_indices[point_idx]],
                np.array(
                    params_org[cameras[points_cam_indices[point_idx]]]['mtx'],
                    np.float32),
                None)[0]
        )

    points_proj = np.array(points_proj).squeeze()

    return points_proj


def fun(params, n_cameras, n_points, params_org,
        points_2d, points_2d_conf, points_indices, points_cam_indices, configs):
    camera_params = params[:n_cameras * configs['calib_param_size']].reshape(
        (n_cameras, configs['calib_param_size']))
    points_3d = params[n_cameras * configs['calib_param_size']:].reshape((n_points, 3))
    
    points_proj = project(
        points_3d,
        points_indices,
        points_cam_indices,
        camera_params,
        params_org)

    diff = (points_2d - points_proj)
    diff[:, 0] = diff[:, 0] * points_2d_conf ** configs['conf_power']
    diff[:, 1] = diff[:, 1] * points_2d_conf ** configs['conf_power']
    diff = diff.ravel()

    return diff


def bundle_adjustment_sparsity(n_cameras, n_points, camera_indices,
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
        # params_ravel[i, 12] = params[camera]['mtx'][0][0]
        # params_ravel[i, 13] = params[camera]['mtx'][1][1]
        # params_ravel[i, 14] = params[camera]['mtx'][0][2]
        # params_ravel[i, 15] = params[camera]['mtx'][1][2]
        # params_ravel[i, 16:] = np.array(params[camera]['dist'], np.float32).ravel()

    return params_ravel.ravel()


def visualize_error(x0, n_cameras, n_points, param_org, 
                    points_2d, points_2d_conf, points_indices, points_cam_indices, configs):
    f0 = fun(x0, n_cameras, n_points, param_org, 
             points_2d, points_2d_conf, points_indices, points_cam_indices, configs)
    
    plt.plot(f0)
    plt.show()


def optimize(n_cameras, n_points, params_org,
             points_2d, points_2d_conf, points_indices, points_cam_indices, configs):
    jac_sparsity = bundle_adjustment_sparsity(
        n_cameras, n_points, points_cam_indices,
        points_indices, configs)

    res = least_squares(
        fun, x0, jac_sparsity=jac_sparsity, verbose=2,
        x_scale='jac', ftol=1e-8, xtol=1e-8, gtol=1e-8,
        method='trf', args=(n_cameras, n_points, params_org,
                            points_2d, points_2d_conf, points_indices,
                            points_cam_indices, configs))

    return res


def reconstruct_params(results, cameras, params_org, configs):
    n_cameras = len(cameras)

    calib_params_size = n_cameras * configs['calib_param_size']
    params = results['x'][:calib_params_size].reshape(n_cameras, -1)

    params_reconstructed = {}
    for idx, camera in enumerate(cameras):
        camera_matrix = params_org[camera]['mtx']
        # camera_matrix = np.zeros((3, 3), dtype=np.float32)
        # camera_matrix[0, 0] = params[idx, 12]
        # camera_matrix[1, 1] = params[idx, 13]
        # camera_matrix[0, 2] = params[idx, 14]
        # camera_matrix[1, 2] = params[idx, 15]
        # camera_matrix[2, 2] = 1.0
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


def _contruct_optimization_params(params, poses_3d, configs):
    params_ravel = ravel(params, configs)
    
    x0 = np.hstack((params_ravel, poses_3d.ravel()))

    return x0


def _format_points(poses_3d, poses_2d, params_org):
    points_3d = []
    points_2d = []
    points_2d_conf = []
    points_indices = []
    points_cam_indices = []

    poses_3d_scene = poses_3d.reshape(poses_3d.shape[0], -1, 3)
    for timestep in range(poses_3d_scene.shape[0]):
        for point_idx_scene in range(poses_3d_scene.shape[1]):
            points_3d.append(poses_3d_scene[timestep, point_idx_scene])
            
            for idx_cam, camera in enumerate(params_org.keys()):
                points_2d_cam_time = np.array(
                    poses_2d[camera]['pose'][timestep+300]).reshape(-1, 2)
                points_2d_cam_time_conf = np.array(
                    poses_2d[camera]['pose_confidence'][timestep+300]).reshape(-1)
                if point_idx_scene < len(points_2d_cam_time) \
                        and points_2d_cam_time_conf[point_idx_scene] \
                            > configs['conf_threshold']:
                    points_2d.append(points_2d_cam_time[point_idx_scene])
                    points_2d_conf.append(points_2d_cam_time_conf[point_idx_scene])
                    points_indices.append(len(points_3d) - 1)
                    points_cam_indices.append(idx_cam)

    points_3d = np.array(points_3d)
    points_2d = np.array(points_2d)
    points_2d_conf = np.array(points_2d_conf)
    points_indices = np.array(points_indices)
    points_cam_indices = np.array(points_cam_indices)

    print(points_3d.shape, '--> all cams:', points_3d.shape[0] * 5)
    print(points_2d.shape)
    print("point ids", points_indices.shape)
    print("cam ids", points_cam_indices.shape)

    return points_3d, points_2d, points_2d_conf, points_indices, points_cam_indices


if __name__ == '__main__':
    args = _get_arguments()
    configs = _load_configs(args.config)

    print(f"Config loaded: {configs}")

    poses_2d, poses_3d, params_org = _load_inputs(configs)

    poses_3d = poses_3d[300:600]
    # poses_2d = poses_2d[300:600]
    print(poses_3d.shape)

    n_cameras = len(list(params_org.keys()))
    n_points = poses_3d.reshape(-1, 3).shape[0]

    points_3d, points_2d, points_2d_conf, points_indices, points_cam_indices = \
        _format_points(poses_3d, poses_2d, params_org)

    x0 = _contruct_optimization_params(params_org, points_3d, configs)

    visualize_error(x0, n_cameras, n_points,
                    params_org, points_2d, points_2d_conf, points_indices,
                    points_cam_indices, configs)

    results = optimize(n_cameras, n_points,
                       params_org, points_2d, points_2d_conf, points_indices,
                       points_cam_indices, configs)

    visualize_error(results['x'], n_cameras, n_points,
                    params_org, points_2d, points_2d_conf, points_indices,
                    points_cam_indices, configs)

    store_results(results, poses_2d, poses_3d, params_org, configs)
