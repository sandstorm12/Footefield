import sys
from turtle import color
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
        default='configs/skeleton_2d_temporal_smoothing.yml',
    )

    args = parser.parse_args()

    return args


def _load_configs(path):
    with open(path, 'r') as yaml_file:
        configs = yaml.safe_load(yaml_file)

    return configs


def get_video_writer(camera, dir, fps, size):
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    writer = cv2.VideoWriter(
        os.path.join(
            dir,
            f'visualizer_skeleton_detection_{camera}.avi'
        ),
        fourcc,
        5,
        (1920, 1080)
    )
    
    return writer


def write_video(poses, camera, intrinsics, configs):
    dir = configs['calibration_folders'][camera]
    img_rgb_paths = data_loader.list_rgb_images(dir)

    if not os.path.exists(configs['output_dir']):
        os.makedirs(configs['output_dir'])

    mtx = np.array(intrinsics[camera]['mtx'], np.float32)
    dist = np.array(intrinsics[camera]['dist'], np.float32)

    writer = get_video_writer(camera, configs['output_dir'], configs['fps'], configs['size'])
    for idx, t in enumerate(poses.reshape(poses.shape[0], -1, 2)):
        img_rgb = cv2.imread(img_rgb_paths[idx])

        img_rgb = cv2.undistort(img_rgb, mtx, dist, None, None)
        for point in t:
            cv2.circle(img_rgb, (int(point[0]), int(point[1])),
                       3, (0, 255, 0), -1)

        connections = np.concatenate(
            [np.array(data_loader.HALPE_EDGES) + i * 26
             for i in range(poses.shape[1])]
        )
        for connection in connections:
            cv2.line(img_rgb,
                    (int(t[connection[0]][0]), int(t[connection[0]][1])),
                    (int(t[connection[1]][0]), int(t[connection[1]][1])),
                    (255, 255, 255), 1)

        writer.write(img_rgb)


def gaussian_filter(size, sigma):
    r = range(-1 * size // 2, size // 2 + 1)
    
    return [1 / (sigma * math.sqrt(2*math.pi)) * math.exp(-float(x)**2/(2*sigma**2)) for x in r]


def _store_artifacts(artifact, output):
    with open(output, 'w') as handle:
        yaml.dump(artifact, handle)


if __name__ == "__main__":
    args = _get_arguments()
    configs = _load_configs(args.config)

    print(f"Config loaded: {configs}")
    
    with open(configs['skeletons']) as handler:
        poses = yaml.safe_load(handler)

    alpha = configs['confidence_weight']
    beta = configs['uncertainty_weight']
    temporal_depth = configs['temporal_depth']
    sigma = configs['sigma']
    gaussian_kernel = np.array(gaussian_filter(
        temporal_depth * 2, sigma))

    poses_smooth = {}

    cameras = poses.keys()
    for idx_cam, camera in enumerate(cameras):
        pose_cam_org = np.array(poses[camera]['pose'])
        confidence_cam_org = np.array(poses[camera]['pose_confidence'])

        pose_cam = pose_cam_org.reshape(pose_cam_org.shape[0], -1, 2)
        pose_smooth = np.empty_like(pose_cam)
        confidence_cam = confidence_cam_org.reshape(confidence_cam_org.shape[0], -1)
        for point_idx in tqdm(range(pose_cam.shape[1])):
            point_motion = pose_cam[:, point_idx]
            confidence_motion = confidence_cam[:, point_idx]

            for t in range(len(point_motion)):
                idx_start_img = t - temporal_depth
                idx_end_img = t + temporal_depth + 1
                idx_start = max(0, idx_start_img)
                idx_end = min(len(point_motion), idx_end_img)
                delta_left = idx_start - idx_start_img
                delta_right = idx_end_img - idx_end

                point_motion_window = point_motion[idx_start:idx_end]
                confidence_motion_window = confidence_motion[idx_start:idx_end]
                gaussian_kernel_window = gaussian_kernel[delta_left:len(gaussian_kernel)-delta_right]
                gaussian_kernel_window = (gaussian_kernel_window * (alpha * confidence_motion_window))
                gaussian_kernel_window = gaussian_kernel_window * (1 / np.sum(gaussian_kernel_window))

                point_smooth_x = np.sum(point_motion_window[:, 0] * gaussian_kernel_window)
                point_smooth_y = np.sum(point_motion_window[:, 1] * gaussian_kernel_window)

                smoothness_weight = (1 / confidence_motion[t]) ** beta
                pose_smooth[t, point_idx, 0] = (point_motion[t][0] + smoothness_weight * point_smooth_x) / (1 + smoothness_weight)
                pose_smooth[t, point_idx, 1] = (point_motion[t][1] + smoothness_weight * point_smooth_y) / (1 + smoothness_weight)

            if configs['visualize']:
                plt.plot(point_motion[:, 0])
                plt.plot(pose_smooth[:, point_idx, 0])
                plt.show()

        poses_smooth[camera] = {}
        poses_smooth[camera]['pose'] = pose_smooth.reshape(pose_cam_org.shape).tolist()
        poses_smooth[camera]['pose_confidence'] = confidence_cam.reshape(confidence_cam_org.shape).tolist()

    _store_artifacts(poses_smooth, configs['output'])
    