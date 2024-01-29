"""
cache `intrinsics` key: camera_folder value: ret, mtx, and dist of calibrateCamera output
"""

import sys
sys.path.append('../')

import cv2
import diskcache
import numpy as np

import detect_chessboard

from tqdm import tqdm
from utils import data_loader


def load_image_points(cache):
    images_info = cache['images_info']

    if not images_info:
        print("'images_info' not found.")

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


def calculate_intrinsics(cameras_info, cache):
    cols = data_loader.CHESSBOARD_COLS
    rows = data_loader.CHESSBOARD_ROWS
    square_size = data_loader.CHESSBOARD_SQRS

    obj_points = np.zeros((cols * rows, 3), np.float32)
    obj_points[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square_size

    intrinsics = {}
    for key in tqdm(cameras_info.keys()):
        img_points = cameras_info[key]['img_points']

        width = data_loader.IMAGE_INFRARED_WIDTH
        height = data_loader.IMAGE_INFRARED_HEIGHT

        ret, mtx, dist, rvecs, tvecs = \
            cv2.calibrateCamera(
                np.tile(obj_points, (len(img_points), 1, 1)),
                img_points,
                (width, height), None, None, flags=cv2.CALIB_RATIONAL_MODEL)

        intrinsics[key + 'infrared'] = {
            "ret": ret,
            "mtx": mtx,
            "dist": dist,
            "rvecs": rvecs,
            "tvecs": tvecs,
        }

    cache['intrinsics'] = intrinsics


if __name__ == "__main__":
    cache = diskcache.Cache('cache')

    cameras_info = load_image_points(cache)
    calculate_intrinsics(cameras_info, cache)
