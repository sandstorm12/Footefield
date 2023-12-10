"""
cache `intrinsics` key: camera_folder value: ret, mtx, and dist of calibrateCamera output
"""

import cv2
import diskcache
import numpy as np

import detect_chessboard

from tqdm import tqdm


def load_image_points(cache):
    images_info = cache.get("images_info", None)

    criteria = (cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, 30, 0.001)
    
    if not images_info:
        print("'images_info' not found.")

    cameras = {}
    for key in images_info.keys():
        camera = key.split("/")[0]

        if not cameras.__contains__(camera):
            cameras[camera] = {'img_points': [], 'width': 0, 'height': 0}
        camera_points = cameras[camera]

        ret, corners = images_info[key]['findchessboardcorners_rgb']
        if not ret:
            continue
        
        image_gray = cv2.imread(images_info[key]['fullpath'], cv2.IMREAD_GRAYSCALE)
        corners_refined = cv2.cornerSubPix(image_gray, corners, (5, 5), (-1, -1), criteria)

        camera_points['img_points'].append(corners_refined)
        camera_points['width'] = image_gray.shape[1] # images_info[key]['width']
        camera_points['height'] = image_gray.shape[0] # ['height']

    return cameras


def calculate_intrinsics(cameras_info, cache):
    cols = detect_chessboard.CHESSBOARD_COLS
    rows = detect_chessboard.CHESSBOARD_ROWS
    square_size = detect_chessboard.CHESSBOARD_SQRS

    obj_points = np.zeros((cols * rows, 3), np.float32)
    obj_points[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square_size

    intrinsics = {}
    for key in tqdm(cameras_info.keys()):
        img_points = cameras_info[key]['img_points']

        width = cameras_info[key]['width']
        height = cameras_info[key]['height']

        ret, mtx, dist, rvecs, tvecs = \
            cv2.calibrateCamera(
                np.tile(obj_points, (len(img_points), 1, 1)),
                img_points,
                (width, height), None, None, flags=cv2.CALIB_RATIONAL_MODEL)

        intrinsics[key] = {
            "ret": ret,
            "mtx": mtx,
            "dist": dist,
        }

    cache['intrinsics'] = intrinsics


if __name__ == "__main__":
    cache = diskcache.Cache('storage')

    cameras_info = load_image_points(cache)
    calculate_intrinsics(cameras_info, cache)
