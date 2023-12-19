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


def calc_depth_rgb_match(cam_1, obj_points, cache):
    print(f"Calibrating... {cam_1}")

    rgb_depth_pairs = find_rgb_depth_images(cache['images_info'], cam_1)

    print(f"Matching pairs: {len(rgb_depth_pairs['image_points_rgb'])}")

    _, mtx_1, dist_1, mtx_2, dist_2, R, T, _, _ = cv2.stereoCalibrate(
        np.tile(obj_points, (len(rgb_depth_pairs['image_points_rgb']), 1, 1)),
        rgb_depth_pairs['image_points_rgb'],
        rgb_depth_pairs['image_points_infrared'],
        None, None, None, None,
        (rgb_depth_pairs['width'], rgb_depth_pairs['height']),
        criteria=STEREO_CALIBRATION_CRITERIA, flags=0)
    
    if not cache.__contains__('extrinsics'):
        cache['extrinsics'] = {}

    depth_matching = cache.get('depth_matching', {})
    depth_matching[cam_1] = {
        'mtx_l': mtx_1,
        'dist_l': dist_1,
        'mtx_r': mtx_2,
        'dist_r': dist_2,
        'rotation': R,
        'transition': T,
    }
    cache['depth_matching'] = depth_matching


def calc_reprojection_error(cam_1, obj_points, cache):
    print(f"Calibrating... {cam_1}")

    rgb_depth_pairs = find_rgb_depth_images(cache['images_info'], cam_1)

    print(f"Matching pairs: {len(rgb_depth_pairs['image_points_rgb'])}")

    if not cache.__contains__('extrinsics'):
        raise Exception('Extrinsics not cached.')
    
    mtx_1 = cache['depth_matching'][cam_1]['mtx_l']
    dist_1 = cache['depth_matching'][cam_1]['dist_l']
    mtx_2 = cache['depth_matching'][cam_1]['mtx_r']
    dist_2 = cache['depth_matching'][cam_1]['dist_r']
    R = cache['depth_matching'][cam_1]['rotation']
    T = cache['depth_matching'][cam_1]['transition']

    RL, RR, PL, PR, Q, roiL, roiR = cv2.stereoRectify(
        mtx_1, dist_1, mtx_2, dist_2, (640, 576), R, T)

    map1, map2 = cv2.initUndistortRectifyMap(
        mtx_1, dist_1, RR, PR,
        (640, 576),
        cv2.CV_32FC1)
    
    for idx in range(len(rgb_depth_pairs['image_paths_infrared'])):
        img_rgb = cv2.imread(rgb_depth_pairs['image_paths_rgb'][idx], cv2.IMREAD_GRAYSCALE)
        img_infrared = cv2.imread(rgb_depth_pairs['image_paths_infrared'][idx], cv2.IMREAD_GRAYSCALE)
        img_infrared = cv2.equalizeHist(img_infrared)
        undistorted_img = cv2.remap(img_rgb, map1, map2, interpolation=cv2.INTER_LINEAR)
        # undistorted_img = cv2.resize(undistorted_img, (img_infrared.shape[1], img_infrared.shape[0]))
        # undistorted_img = cv2.cvtColor(undistorted_img, cv2.COLOR_BGR2GRAY)
        undistorted_img = ((img_infrared * 1. + undistorted_img * 1.) / 2).astype(np.uint8)
        cv2.imshow("image", undistorted_img)
        if cv2.waitKey(0) == ord('q'):
            break
    
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
        print(f"Errors: cam1 {error1} cam2 {error2}")
        total_error += (error1 + error2) / 2

    # Print the average projection error
    print("Average projection error: ",
          total_error / len(rgb_depth_pairs['image_points_rgb']))


if __name__ == "__main__":
    cache = diskcache.Cache('cache')

    obj_points = get_obj_points()
    intrinsics = cache['intrinsics']

    cameras = list(intrinsics.keys())
    for cam1_idx in range(len(cameras)):
        cam_1 = cameras[cam1_idx]

        calc_depth_rgb_match(cam_1, obj_points, cache)
        calc_reprojection_error(cam_1, obj_points, cache)
