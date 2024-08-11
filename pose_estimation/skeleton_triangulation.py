"""Extract initial 3D positions of the 2D skeleton keypoints by
multi-camera triangulation
"""

import sys
sys.path.append('../')

import yaml
import argparse
import numpy as np

import pycalib

from tqdm import tqdm


def _get_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-c', '--config',
        help='Path to the config file',
        type=str,
        default='configs/skeleton_triangulation.yml',
    )

    args = parser.parse_args()

    return args


def _load_configs(path):
    with open(path, 'r') as yaml_file:
        configs = yaml.safe_load(yaml_file)

    return configs


def _get_intrinsics(configs):
    with open(configs['intrinsics']) as handler:
        intrinsics = yaml.safe_load(handler)

    return intrinsics


def _get_extrinsics(configs):
    with open(configs['extrinsics']) as handler:
        extrinsics = yaml.safe_load(handler)

    return extrinsics


def _calc_extrinsics(cam, extrinsics):
    R = np.array(extrinsics["cam2_4"]['rotation'], np.float32)
    T = np.array(extrinsics["cam2_4"]['transition'], np.float32)
    R2 = np.array(extrinsics["cam1_5"]['rotation'], np.float32)
    T2 = np.array(extrinsics["cam1_5"]['transition'], np.float32)
    R3 = np.array(extrinsics["cam1_4"]['rotation'], np.float32)
    T3 = np.array(extrinsics["cam1_4"]['transition'], np.float32)
    R4 = np.array(extrinsics["cam3_4"]['rotation'], np.float32)
    T4 = np.array(extrinsics["cam3_4"]['transition'], np.float32)
    
    extrinsics = np.zeros((3, 4), dtype=float)
    if cam == "cam2_4":
        r = np.array([[1, 0, 0],
                     [0, 1, 0],
                     [0, 0, 1]])
        t = np.array([0, 0, 0])
        extrinsics[:3, :3] = r
        extrinsics[:3, 3] = t.reshape(3)
    elif cam == "cam1_5":
        extrinsics[:3, :3] = R
        extrinsics[:3, 3] = T.reshape(3)
    elif cam == "cam1_4":
        R2_com = np.dot(R2, R)
        T2_com = (np.dot(R2, T).reshape(3, 1) + T2).reshape(3,)
        extrinsics[:3, :3] = R2_com
        extrinsics[:3, 3] = T2_com
    elif cam == "cam3_4":
        R2_com = np.dot(R2, R)
        T2_com = (np.dot(R2, T).reshape(3, 1) + T2).reshape(3,)
        R3_com = np.dot(R3, R2_com)
        T3_com = (np.dot(R3, T2_com).reshape(3, 1) + T3).reshape(3,)
        extrinsics[:3, :3] = R3_com
        extrinsics[:3, 3] = T3_com
    elif cam == "cam3_5":
        R2_com = np.dot(R2, R)
        T2_com = (np.dot(R2, T).reshape(3, 1) + T2).reshape(3,)
        R3_com = np.dot(R3, R2_com)
        T3_com = (np.dot(R3, T2_com).reshape(3, 1) + T3).reshape(3,)
        R4_com = np.dot(R4, R3_com)
        T4_com = (np.dot(R4, T3_com).reshape(3, 1) + T4).reshape(3,)
        extrinsics[:3, :3] = R4_com
        extrinsics[:3, 3] = T4_com

    return extrinsics


# TODO: Too long
def calc_3d_skeleton(poses, configs):
    cameras = poses.keys()
    length = len(poses[list(cameras)[0]]['pose'])

    intrinsics = _get_intrinsics(configs)
    extrinsics = _get_extrinsics(configs)

    num_people = 0
    num_points = 0
    
    for camera in cameras:
        points = np.array(poses[camera]['pose']).shape[1:3]
        num_points = points[1]
        if num_people < points[0]:
            num_people = points[0]
    points_per_timestep = num_people * num_points

    points_3d = []
    for timestep in tqdm(range(length)):
        points_3d_timestep = []
        for point_idx in range(points_per_timestep):
            points_2d = []
            parameters = []
            for camera in cameras:
                points_timestep = np.array(
                    poses[camera]['pose'][timestep]).reshape(-1, 2)
                confidences_timestep = np.array(
                    poses[camera]['pose_confidence'][timestep]).reshape(-1)
                if point_idx < len(points_timestep) and \
                        confidences_timestep[point_idx] > configs['threshold']:
                    points_2d.append(points_timestep[point_idx])
                    
                    cam_mtx = np.array(intrinsics[camera]['mtx'], np.float32)
                    cam_extrinsics = _calc_extrinsics(camera, extrinsics)
                    parameters.append(cam_mtx @ cam_extrinsics)

            points_2d = np.expand_dims(np.array(points_2d), 1)
            parameters = np.array(parameters)
            points_3d_single = pycalib.triangulate_Npts(
                pt2d_CxPx2=points_2d, P_Cx3x4=parameters)
            
            points_3d_timestep.append(points_3d_single)
        points_3d.append(points_3d_timestep)

    points_3d = np.array(points_3d).reshape(length, num_people, num_points, 3)

    return points_3d


def _store_artifacts(artifact, output):
    with open(output, 'w') as handle:
        yaml.dump(artifact, handle)


if __name__ == "__main__":
    args = _get_arguments()
    configs = _load_configs(args.config)

    print(f"Config loaded: {configs}")

    with open(configs['skeletons']) as handler:
        poses = yaml.safe_load(handler)
        
    poses_3d = calc_3d_skeleton(poses, configs)

    _store_artifacts(poses_3d.tolist(), configs['output'])
