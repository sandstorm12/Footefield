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
        default='configs/visualizer_skeleton_3d_video.yml',
    )

    args = parser.parse_args()

    return args


def _load_configs(path):
    with open(path, 'r') as yaml_file:
        configs = yaml.safe_load(yaml_file)

    return configs


# By Gemini
def project_3d_to_2d(camera_matrix, dist_coeffs, rvec, tvec, object_points):
    image_points, _ = cv2.projectPoints(object_points, rvec, tvec,
                                        camera_matrix, dist_coeffs)

    image_points = image_points.squeeze()

    return image_points


def get_video_writer(camera, configs):
    if not os.path.exists(configs['output_dir']):
        os.makedirs(configs['output_dir'])

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    writer = cv2.VideoWriter(
        os.path.join(
            configs['output_dir'],
            f'visualizer_skeleton_3d_video_{camera}.avi'
        ),
        fourcc,
        configs['fps'],
        configs['size']
    )
    
    return writer


def write_video(poses_2d, camera, params, configs):
    dir = configs['calibration_folders'][camera]['path']
    cap = cv2.VideoCapture(dir)
    offset = configs['calibration_folders'][camera]['offset']
    for _ in range(offset):
        cap.grab()

    mtx = np.array(params['mtx'], np.float32)
    dist = np.array(params['dist'], np.float32)

    writer = get_video_writer(camera, configs)
    for _, t in enumerate(tqdm(poses_2d.reshape(poses_2d.shape[0], -1, 2))):
        _, img_rgb = cap.read()

        img_rgb = cv2.undistort(img_rgb, mtx, dist, None, None)
        for point in t:
            cv2.circle(img_rgb, (int(point[0]), int(point[1])),
                       3, (0, 255, 0), -1)

        connections = np.array(data_loader.HALPE_EDGES)
        for connection in connections:
            cv2.line(img_rgb,
                    (int(t[connection[0]][0]), int(t[connection[0]][1])),
                    (int(t[connection[1]][0]), int(t[connection[1]][1])),
                    (255, 255, 255), 1)

        writer.write(img_rgb)


def poses_3d_2_2d(poses_3d, params):
    poses_shape = list(poses_3d.shape)
    poses_shape[-1] = 2
    
    mtx = np.array(params['mtx'], np.float64)
    # dist = np.array(params['dist'], np.float64)
    rotation = np.array(params['rotation'], np.float64)
    translation = np.array(params['translation'], np.float64)
    poses_2d = project_3d_to_2d(
        mtx, None,
        rotation,
        translation,
        poses_3d.reshape(-1, 3).astype(np.float32))
    poses_2d[:, 1] = poses_2d[:, 1]
    poses_2d = poses_2d.reshape(poses_shape)

    return poses_2d


if __name__ == "__main__":
    args = _get_arguments()
    configs = _load_configs(args.config)

    print(f"Config loaded: {configs}")
    
    with open(configs['skeletons']) as handler:
        poses = yaml.safe_load(handler)

    with open(configs['params']) as handler:
        params = yaml.safe_load(handler)

    poses = np.array(poses)

    cameras = list(configs['calibration_folders'].keys())
    for idx_cam, camera in enumerate(tqdm(cameras)):
        dir = configs['calibration_folders'][camera]

        poses_2d = poses_3d_2_2d(
            poses,
            params[camera])

        write_video(poses_2d, camera, params[camera], configs)
