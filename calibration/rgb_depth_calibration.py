import sys
sys.path.append('../')

import cv2
import diskcache
import numpy as np

from utils import data_loader


STEREO_CALIBRATION_CRITERIA = (
    cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS,
    1000, 1e-6)


def get_obj_points():
    cols = data_loader.CHESSBOARD_COLS
    rows = data_loader.CHESSBOARD_ROWS
    square_size = data_loader.CHESSBOARD_SQRS

    obj_points = np.zeros((cols * rows, 3), np.float32)
    obj_points[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square_size

    return obj_points


def find_rgb_depth_images(images_info, camera):
    images_info = cache['images_info']

    image_points_rgb = []
    image_points_infrared = []
    for key in images_info.keys():
        if key.split("/")[0] == camera:
            points_found_rgb, points_rgb = \
                images_info[key]['findchessboardcorners_rgb']
            points_found_infrared, points_infrared = \
                images_info[key]['findchessboardcorners_infrared']
            if points_found_rgb and points_found_infrared:
                image_points_rgb.append(points_rgb)
                image_points_infrared.append(points_infrared)

    rgb_depth_pairs = {
        "image_points_rgb": image_points_rgb,
        "image_points_infrared": image_points_infrared,
    }
            
    return rgb_depth_pairs


def calc_depth_rgb_match(camera, obj_points, cache):
    print(f"Calibrating... {camera}")

    rgb_depth_pairs = find_rgb_depth_images(cache['images_info'], camera)

    print(f"Matching pairs: {len(rgb_depth_pairs['image_points_rgb'])}")

    _, mtx_1, dist_1, mtx_2, dist_2, R, T, _, _ = cv2.stereoCalibrate(
        np.tile(obj_points, (len(rgb_depth_pairs['image_points_rgb']), 1, 1)),
        rgb_depth_pairs['image_points_rgb'],
        rgb_depth_pairs['image_points_infrared'],
        None, None, None, None,
        (data_loader.IMAGE_RGB_WIDTH,
         data_loader.IMAGE_RGB_HEIGHT),
        criteria=STEREO_CALIBRATION_CRITERIA, flags=0)
    
    if not cache.__contains__('extrinsics'):
        cache['extrinsics'] = {}

    depth_matching = cache.get('depth_matching', {})
    depth_matching[camera] = {
        'mtx_l': mtx_1,
        'dist_l': dist_1,
        'mtx_r': mtx_2,
        'dist_r': dist_2,
        'rotation': R,
        'transition': T,
    }
    cache['depth_matching'] = depth_matching


def calc_reprojection_error(camera, obj_points, cache):
    print(f"Calculating reprojection error... {camera}")

    rgb_depth_pairs = find_rgb_depth_images(cache['images_info'], camera)

    if not cache.__contains__('extrinsics'):
        raise Exception('Extrinsics not cached.')
    
    mtx_1 = cache['depth_matching'][camera]['mtx_l']
    dist_1 = cache['depth_matching'][camera]['dist_l']
    mtx_2 = cache['depth_matching'][camera]['mtx_r']
    dist_2 = cache['depth_matching'][camera]['dist_r']
    R = cache['depth_matching'][camera]['rotation']
    T = cache['depth_matching'][camera]['transition']
    
    total_error = 0
    for i in range(len(rgb_depth_pairs['image_points_rgb'])):
        _, rvec_l, tvec_l = cv2.solvePnP(
            obj_points, rgb_depth_pairs['image_points_rgb'][i], mtx_1, dist_1)
        rvec_r, tvec_r = cv2.composeRT(
            rvec_l, tvec_l, cv2.Rodrigues(R)[0], T)[:2]

        imgpoints1_projected, _ = cv2.projectPoints(
            obj_points, rvec_l, tvec_l, mtx_1, dist_1)
        imgpoints2_projected, _ = cv2.projectPoints(
            obj_points, rvec_r, tvec_r, mtx_2, dist_2)

        error1 = cv2.norm(
            rgb_depth_pairs['image_points_rgb'][i],
            imgpoints1_projected, cv2.NORM_L2) \
                / len(imgpoints1_projected)
        error2 = cv2.norm(
            rgb_depth_pairs['image_points_infrared'][i],
            imgpoints2_projected, cv2.NORM_L2) \
                / len(imgpoints2_projected)
        total_error += (error1 + error2) / 2

    # Print the average projection error
    print("Average projection error: ",
          total_error / len(rgb_depth_pairs['image_points_rgb']))


if __name__ == "__main__":
    cache = diskcache.Cache('cache')

    obj_points = get_obj_points()
    intrinsics = cache['intrinsics']

    # TODO: Not very clean
    cameras = [camera for camera in intrinsics.keys()
               if '_infrared' not in camera]
    for camera in cameras:
        calc_depth_rgb_match(camera, obj_points, cache)
        calc_reprojection_error(camera, obj_points, cache)
