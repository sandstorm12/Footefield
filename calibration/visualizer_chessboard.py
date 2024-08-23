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
            f'visualizer_chessboard_{camera}.avi'
        ),
        fourcc,
        fps,
        size,
    )
    
    return writer


def write_video(chessboards, camera, configs):
    dir = configs['calibration_folders'][camera]['path']
    cap = cv2.VideoCapture(dir)
    for _ in range(configs['calibration_folders'][camera]['offset']):
        cap.grab()

    if not os.path.exists(configs['output_dir']):
        os.makedirs(configs['output_dir'])

    writer = get_video_writer(camera, configs['output_dir'], configs['fps'], configs['size'])
    for ret, points in chessboards:
        _, img_rgb = cap.read()

        if ret:
            for point in points:
                cv2.circle(img_rgb, (int(point[0][0]), int(point[0][1])),
                        3, (0, 255, 0), -1)

        writer.write(img_rgb)


if __name__ == "__main__":
    args = _get_arguments()
    configs = _load_configs(args.config)

    print(f"Config loaded: {configs}")
    
    with open(configs['chessboards']) as handler:
        images_info = yaml.safe_load(handler)

    print(images_info.keys())

    cameras = images_info.keys()
    for idx_cam, camera in enumerate(tqdm(cameras)):
        # Remove camname or camidx
        write_video(images_info[camera], camera, configs)
