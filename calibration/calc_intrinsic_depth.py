"""
cache `intrinsics` key: camera_folder value: ret, mtx, and dist of calibrateCamera output
"""

import sys
sys.path.append('../')

import cv2
import yaml
import argparse
import diskcache
import numpy as np

from tqdm import tqdm
from utils import data_loader


def _get_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-c', '--config',
        help='Path to the config file',
        type=str,
        default='configs/calc_intrinsic_depth.yml',
    )

    args = parser.parse_args()

    return args


def _load_configs(path):
    with open(path, 'r') as yaml_file:
        configs = yaml.safe_load(yaml_file)

    print(configs)

    return configs


def load_image_points(configs):
    with open(configs['chessboards']) as handler:
        images_info = yaml.safe_load(handler)

    if len(images_info.keys()) == 0:
        print("No images in images_info. Please run detect_chessboard first.")

    cameras = {}
    for key in images_info.keys():
        camera = key.split("/")[0]

        if not cameras.__contains__(camera):
            cameras[camera] = {'img_points': []}
        camera_points = cameras[camera]

        ret, corners = images_info[key]['findchessboardcorners_infrared']
        if not ret:
            continue
        
        camera_points['img_points'].append(corners)

    return cameras


def calculate_intrinsics(cameras_info, configs):
    cols = data_loader.CHESSBOARD_COLS
    rows = data_loader.CHESSBOARD_ROWS
    square_size = data_loader.CHESSBOARD_SQRS

    obj_points = np.zeros((cols * rows, 3), np.float32)
    obj_points[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square_size

    intrinsics = {}
    for key in tqdm(cameras_info.keys()):
        img_points = np.array(cameras_info[key]['img_points'],
                              dtype=np.float32)

        width = data_loader.IMAGE_INFRARED_WIDTH
        height = data_loader.IMAGE_INFRARED_HEIGHT

        ret, mtx, dist, rvecs, tvecs = \
            cv2.calibrateCamera(
                np.tile(obj_points, (len(img_points), 1, 1)),
                img_points,
                (width, height), None, None, flags=cv2.CALIB_FIX_K3)
        
        mtx = mtx.tolist()
        dist = dist.tolist()
        # rvecs = [item.tolist() for item in rvecs]
        # tvecs = [item.tolist() for item in tvecs]

        intrinsics[key] = {
            "ret": ret,
            "mtx": mtx,
            "dist": dist,
            # "rvecs": rvecs,
            # "tvecs": tvecs,
        }

    _store_artifacts(intrinsics, configs)


def _store_artifacts(artifact, configs):
    with open(configs['output_dir'], 'w') as handle:
        yaml.dump(artifact, handle)


def calc_intrinsic(configs):
    cameras_info = load_image_points(configs)
    calculate_intrinsics(cameras_info, configs)


if __name__ == "__main__":
    args = _get_arguments()
    configs = _load_configs(args.config)

    print(f"Config loaded: {configs}")

    calc_intrinsic(configs)
