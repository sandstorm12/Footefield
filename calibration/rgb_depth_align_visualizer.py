import sys
sys.path.append('../')

import cv2
import diskcache
import numpy as np

from tqdm import tqdm
from utils import data_loader


STEREO_CALIBRATION_CRITERIA = (
    cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS,
    1000, 1e-6)

DISPARITY = -18


def get_obj_points():
    cols = data_loader.CHESSBOARD_COLS
    rows = data_loader.CHESSBOARD_ROWS
    square_size = data_loader.CHESSBOARD_SQRS

    obj_points = np.zeros((cols * rows, 3), np.float32)
    obj_points[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square_size

    return obj_points


def load_image_points(cache, images):
    images_info = cache['images_info']

    criteria = (cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, 30, 0.001)
    
    if not images_info:
        print("'images_info' not found.")

    if len(images_info.keys()) == 0:
        print("No images in images_info. Please run detect_chessboard first.")

    img_points = []
    for key in tqdm(images):
        ret, corners = images_info[key]['findchessboardcorners_rgb']
        if not ret:
            continue
        
        image_gray = cv2.imread(
            images_info[key]['fullpath'], cv2.IMREAD_GRAYSCALE)
        corners_refined = cv2.cornerSubPix(
            image_gray, corners, (5, 5), (-1, -1), criteria)

        img_points.append(corners_refined)
        width = image_gray.shape[1] # images_info[key]['width']
        height = image_gray.shape[0] # ['height']

    return img_points, width, height


def find_rgb_depth_images(images_info, cam_1):
    images_info = cache['images_info']

    image_paths_rgb = []
    image_paths_infrared = []
    image_points_rgb = []
    image_points_infrared = []
    width = None
    height = None
    for key in images_info.keys():
        if key.split("/")[0] == cam_1:
            points_found_rgb, points_rgb = \
                images_info[key]['findchessboardcorners_rgb']
            points_found_infrared, points_infrared = \
                images_info[key]['findchessboardcorners_infrared']
            if points_found_rgb and points_found_infrared:
                image_paths_rgb.append(images_info[key]['fullpath_rgb'])
                image_paths_infrared.append(images_info[key]['fullpath_infrared'])
                image_points_rgb.append(points_rgb)
                image_points_infrared.append(points_infrared)
                width = images_info[key]['width']
                height = images_info[key]['height']

    rgb_depth_pairs = {
        "image_paths_rgb": image_paths_rgb,
        "image_paths_infrared": image_paths_infrared,
        "image_points_rgb": image_points_rgb,
        "image_points_infrared": image_points_infrared,
        "width": width,
        "height": height,
    }
            
    return rgb_depth_pairs


def calc_reprojection_error(cam_1, obj_points, cache):
    print(f"Calibrating... {cam_1}")

    rgb_depth_pairs = find_rgb_depth_images(cache['images_info'], cam_1)

    print(f"Matching pairs: {len(rgb_depth_pairs['image_points_rgb'])}")

    if not cache.__contains__('depth_matching'):
        raise Exception('Depth matching not cached. '
                        'Run rgb_depth_calibration script.')
    
    map1x = cache['depth_matching'][cam_1]['map_rgb_x']
    map1y = cache['depth_matching'][cam_1]['map_rgb_y']
    map2x = cache['depth_matching'][cam_1]['map_infrared_x']
    map2y = cache['depth_matching'][cam_1]['map_infrared_y']
    
    for idx in range(len(rgb_depth_pairs['image_points_rgb'])):
        img_rgb = cv2.imread(rgb_depth_pairs['image_paths_rgb'][idx],
                             cv2.IMREAD_GRAYSCALE)
        img_rgb = cv2.remap(img_rgb, map1x, map1y, cv2.INTER_LANCZOS4)

        img_inf = cv2.imread(rgb_depth_pairs['image_paths_infrared'][idx], -1)

        # Remove magic number .8
        img_inf = np.clip(
            img_inf.astype(np.float32) * .8, 0, 255).astype('uint8')
        img_inf = cv2.resize(
            img_inf,
            (data_loader.IMAGE_RGB_WIDTH, data_loader.IMAGE_RGB_HEIGHT))
        img_inf = cv2.remap(img_inf, map2x, map2y, cv2.INTER_LANCZOS4)

        # Add the dispartiy between RGB and INFRARED cameras
        img_inf = np.roll(img_inf, DISPARITY, axis=1)

        img_cmb = (img_rgb * .5 + img_inf * .5).astype(np.uint8)

        cv2.imshow("CMB", img_cmb)
        if cv2.waitKey(0) == ord('q'):
            break


if __name__ == "__main__":
    cache = diskcache.Cache('cache')

    obj_points = get_obj_points()
    intrinsics = cache['intrinsics']

    cameras = list(intrinsics.keys())
    for cam1_idx in range(len(cameras)):
        cam_1 = cameras[cam1_idx]

        calc_reprojection_error(cam_1, obj_points, cache)
