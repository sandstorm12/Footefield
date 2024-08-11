import sys
sys.path.append('../')

import os
import cv2
import yaml
import math
import argparse
import numpy as np

from tqdm import tqdm
from utils import data_loader

import matplotlib.pyplot as plt


def _get_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-c', '--config',
        help='Path to the config file',
        type=str,
        default='configs/skeleton_3d_temporal_smoothing.yml',
    )

    args = parser.parse_args()

    return args


def _load_configs(path):
    with open(path, 'r') as yaml_file:
        configs = yaml.safe_load(yaml_file)

    return configs


def gaussian_filter(size, sigma):
    r = range(-1 * size // 2, size // 2 + 1)
    
    return [1 / (sigma * math.sqrt(2*math.pi)) * math.exp(-float(x)**2/(2*sigma**2)) for x in r]


def _store_artifacts(artifact, output):
    with open(output, 'w') as handle:
        yaml.dump(artifact, handle)


# TODO: Make shorter
if __name__ == "__main__":
    args = _get_arguments()
    configs = _load_configs(args.config)

    print(f"Config loaded: {configs}")
    
    with open(configs['skeletons']) as handler:
        poses = yaml.safe_load(handler)

    smoothness_weight = configs['smoothness_weight']
    temporal_depth = configs['temporal_depth']
    sigma = configs['sigma']
    gaussian_kernel = np.array(gaussian_filter(
        temporal_depth * 2, sigma))

    poses_smooth = {}

    pose_cam_org = np.array(poses)

    pose_cam = pose_cam_org.reshape(pose_cam_org.shape[0], -1, 3)
    pose_smooth = np.empty_like(pose_cam)
    for point_idx in tqdm(range(pose_cam.shape[1])):
        point_motion = pose_cam[:, point_idx]

        for t in range(len(point_motion)):
            idx_start_img = t - temporal_depth
            idx_end_img = t + temporal_depth + 1
            idx_start = max(0, idx_start_img)
            idx_end = min(len(point_motion), idx_end_img)
            delta_left = idx_start - idx_start_img
            delta_right = idx_end_img - idx_end

            point_motion_window = point_motion[idx_start:idx_end]
            gaussian_kernel_window = gaussian_kernel[delta_left:len(gaussian_kernel)-delta_right]
            gaussian_kernel_window = gaussian_kernel_window / np.sum(gaussian_kernel_window)

            point_smooth_x = np.sum(point_motion_window[:, 0] * gaussian_kernel_window)
            point_smooth_y = np.sum(point_motion_window[:, 1] * gaussian_kernel_window)
            point_smooth_z = np.sum(point_motion_window[:, 2] * gaussian_kernel_window)

            pose_smooth[t, point_idx, 0] = \
                (point_motion[t][0] + smoothness_weight * point_smooth_x) \
                    / (1 + smoothness_weight)
            pose_smooth[t, point_idx, 1] = \
                (point_motion[t][1] + smoothness_weight * point_smooth_y) \
                    / (1 + smoothness_weight)
            pose_smooth[t, point_idx, 2] = \
                (point_motion[t][2] + smoothness_weight * point_smooth_z) \
                    / (1 + smoothness_weight)

        if configs['visualize']:
            plt.plot(point_motion[:, 0])
            plt.plot(pose_smooth[:, point_idx, 0])
            plt.show()

    poses_smooth = pose_smooth.reshape(pose_cam_org.shape).tolist()

    _store_artifacts(poses_smooth, configs['output'])
    