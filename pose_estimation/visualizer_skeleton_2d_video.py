import sys
sys.path.append('../')

import os
import cv2
import yaml
import argparse
import numpy as np

from tqdm import tqdm
from utils import data_loader


def _get_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-c', '--config',
        help='Path to the config file',
        type=str,
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
            f'visualizer_skeleton_2d_{camera}.avi'
        ),
        fourcc,
        fps,
        size,
    )
    
    return writer


def write_video(poses, camera, idx_cam, intrinsics, configs):
    dir = configs['calibration_folders'][idx_cam]['path']
    cap = cv2.VideoCapture(dir)

    if not os.path.exists(configs['output_dir']):
        os.makedirs(configs['output_dir'])

    mtx = np.array(intrinsics[camera]['mtx'], np.float32)
    dist = np.array(intrinsics[camera]['dist'], np.float32)

    writer = get_video_writer(camera, configs['output_dir'],
                              configs['fps'], configs['size'])
    for _, t in enumerate(tqdm(poses.reshape(poses.shape[0], -1, 2))):
        _, img_rgb = cap.read()

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


if __name__ == "__main__":
    args = _get_arguments()
    configs = _load_configs(args.config)

    print(f"Config loaded: {configs}")
    
    with open(configs['skeletons']) as handler:
        poses = yaml.safe_load(handler)

    cameras = poses.keys()
    for idx_cam, camera in enumerate(tqdm(cameras)):
        with open(configs['intrinsics']) as handler:
            intrinsics = yaml.safe_load(handler)
        
        poses_cam = np.array(poses[camera]['pose'])

        # Remove camname or camidx
        write_video(poses_cam, camera, idx_cam, intrinsics, configs)
