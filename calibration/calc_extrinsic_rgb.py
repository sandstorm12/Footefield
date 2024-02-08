import sys
sys.path.append('../')

import cv2
import diskcache
import numpy as np

from tqdm import tqdm
from utils import data_loader


ORDER_VALID = (
    'azure_kinect1_5_calib_snap/azure_kinect1_4_calib_snap',
    'azure_kinect1_4_calib_snap/azure_kinect3_4_calib_snap',
    'azure_kinect3_4_calib_snap/azure_kinect3_5_calib_snap',
    'azure_kinect3_5_calib_snap/azure_kinect2_4_calib_snap',
    'azure_kinect2_4_calib_snap/azure_kinect1_5_calib_snap',
)

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


def get_all_keys(cache):
    keys_all = cache._sql('SELECT key FROM Cache').fetchall()
    keys_all = [item[0] for item in keys_all]

    return keys_all


def load_image_points(cache, images):
    images_info = cache['images_info']

    if not images_info:
        print("'images_info' not found.")

    if len(images_info.keys()) == 0:
        print("No images in images_info. Please run detect_chessboard first.")

    img_points = []
    for key in tqdm(images):
        ret, corners = images_info[key]['findchessboardcorners_rgb']
        if not ret:
            continue
        
        img_points.append(corners)

    return img_points


def find_matching_images(images_info, cam_1, cam_2):
    matching_pairs = {}
    for image in images_info.keys():
        img_name = image.split("/")[1]
        cam_1_img = f"{cam_1}/{img_name}"
        cam_2_img = f"{cam_2}/{img_name}"
        if images_info.__contains__(cam_1_img) \
            and images_info.__contains__(cam_2_img):
            points_found_1, _ = \
                images_info[cam_1_img]['findchessboardcorners_rgb']
            points_found_2, _ = \
                images_info[cam_2_img]['findchessboardcorners_rgb']
            if points_found_1 and points_found_2:
                matching_pairs[img_name] = {
                    "cam_1_img": cam_1_img,
                    "cam_2_img": cam_2_img,
                }
            
    return matching_pairs


def calc_extrinsics(cam_1, cam_2, obj_points, cache):
    print(f"Calibrating... {cam_1} vs {cam_2}")

    matching_pairs = find_matching_images(cache['images_info'], cam_1, cam_2)

    print(f"Matching pairs: {len(matching_pairs)}")

    img_points_1 = load_image_points(
        cache, images=[item['cam_1_img'] for item in matching_pairs.values()])
    img_points_2 = load_image_points(
        cache, images=[item['cam_2_img'] for item in matching_pairs.values()])

    _, mtx_1, dist_1, mtx_2, dist_2, R, T, _, _ = cv2.stereoCalibrate(
        np.tile(obj_points, (len(img_points_1), 1, 1)),
        img_points_1, img_points_2,
        None, None, None, None,
        (data_loader.IMAGE_INFRARED_WIDTH,
         data_loader.IMAGE_INFRARED_HEIGHT),
        criteria=STEREO_CALIBRATION_CRITERIA, flags=0)
    
    if not cache.__contains__('extrinsics'):
        cache['extrinsics'] = {}

    extrinsics = cache['extrinsics']
    extrinsics[cam_1] = {
        'right_cam': cam_2,
        'mtx_l': mtx_1,
        'dist_l': dist_1,
        'mtx_r': mtx_2,
        'dist_r': dist_2,
        'rotation': R,
        'transition': T,
    }
    cache['extrinsics'] = extrinsics


def calc_reprojection_error(cam_1, cam_2, obj_points, cache):
    print(f"Calibrating... {cam_1} vs {cam_2}")

    matching_pairs = find_matching_images(cache['images_info'], cam_1, cam_2)

    print(f"Matching pairs: {len(matching_pairs)}")

    img_points_1 = load_image_points(
        cache, images=[item['cam_1_img'] for item in matching_pairs.values()])
    img_points_2 = load_image_points(
        cache, images=[item['cam_2_img'] for item in matching_pairs.values()])

    if not cache.__contains__('extrinsics'):
        raise Exception('Extrinsics not cached.')
    
    mtx_1 = cache['extrinsics'][cam_1]['mtx_l']
    dist_1 = cache['extrinsics'][cam_1]['dist_l']
    mtx_2 = cache['extrinsics'][cam_1]['mtx_r']
    dist_2 = cache['extrinsics'][cam_1]['dist_r']
    R = cache['extrinsics'][cam_1]['rotation']
    T = cache['extrinsics'][cam_1]['transition']
    
    total_error = 0
    for i in range(len(img_points_1)):
        _, rvec_l, tvec_l = cv2.solvePnP(obj_points, img_points_1[i], mtx_1, dist_1)
        rvec_r, tvec_r = cv2.composeRT(rvec_l, tvec_l, cv2.Rodrigues(R)[0], T)[:2]

        imgpoints1_projected, _ = cv2.projectPoints(
            obj_points, rvec_l, tvec_l, mtx_1, dist_1)
        imgpoints2_projected, _ = cv2.projectPoints(
            obj_points, rvec_r, tvec_r, mtx_2, dist_2)

        error1 = cv2.norm(
            img_points_1[i], imgpoints1_projected, cv2.NORM_L2) \
                / len(imgpoints1_projected)
        error2 = cv2.norm(
            img_points_2[i], imgpoints2_projected, cv2.NORM_L2) \
                / len(imgpoints2_projected)
        print(f"Errors: cam1 {error1} cam2 {error2}")
        total_error += (error1 + error2) / 2

    # Print the average projection error
    print("Average projection error: ", total_error / len(img_points_1))


def calc_extrinsic():
    cache = diskcache.Cache('cache')

    print("Cache keys:", get_all_keys(cache))

    obj_points = get_obj_points()
    intrinsics = cache['intrinsics']

    cameras = list(intrinsics.keys())
    for cam1_idx in range(len(cameras)):
        for cam2_idx in range(len(cameras)):
            cam_1 = cameras[cam1_idx]
            cam_2 = cameras[cam2_idx]

            if f"{cam_1}/{cam_2}" not in ORDER_VALID:
                continue
                    
            calc_extrinsics(cam_1, cam_2, obj_points, cache)
            calc_reprojection_error(cam_1, cam_2, obj_points, cache)


if __name__ == "__main__":
    calc_extrinsic()    
