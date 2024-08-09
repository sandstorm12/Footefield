import sys
sys.path.append('../')

import cv2
import yaml
import argparse
import numpy as np

from tqdm import tqdm
from utils import data_loader


ORDER_VALID = (
    'cam1_5/cam1_4',
    'cam1_4/cam3_4',
    'cam3_4/cam3_5',
    # 'cam3_5/cam2_4', # No matches found
    'cam2_4/cam1_5',
)

STEREO_CALIBRATION_CRITERIA = (
    cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS,
    1000, 1e-6)


def _get_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-c', '--config',
        help='Path to the config file',
        type=str,
        default='configs/calc_extrinsic_depth.yml',
    )

    args = parser.parse_args()

    return args


def _load_configs(path):
    with open(path, 'r') as yaml_file:
        configs = yaml.safe_load(yaml_file)

    return configs


def get_obj_points():
    cols = data_loader.CHESSBOARD_COLS
    rows = data_loader.CHESSBOARD_ROWS
    square_size = data_loader.CHESSBOARD_SQRS

    obj_points = np.zeros((cols * rows, 3), np.float32)
    obj_points[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square_size

    return obj_points


def load_image_points(images_info, images):
    if len(images_info.keys()) == 0:
        print("No images in images_info. Please run detect_chessboard first.")

    img_points = []
    for key in tqdm(images):
        ret, corners = images_info[key]['findchessboardcorners_infrared']
        if not ret:
            continue
        
        img_points.append(corners)

    img_points = np.array(img_points, dtype=np.float32)

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
                images_info[cam_1_img]['findchessboardcorners_infrared']
            points_found_2, _ = \
                images_info[cam_2_img]['findchessboardcorners_infrared']
            if points_found_1 and points_found_2:
                matching_pairs[img_name] = {
                    "cam_1_img": cam_1_img,
                    "cam_2_img": cam_2_img,
                }
            
    return matching_pairs


def calc_extrinsics(cam_1, cam_2, obj_points, extrinsics, configs):
    print(f"Calibrating... {cam_1} vs {cam_2}")

    with open(configs['chessboards']) as handler:
        images_info = yaml.safe_load(handler)

    matching_pairs = find_matching_images(images_info, cam_1, cam_2)

    print(f"Matching pairs: {len(matching_pairs)}")

    img_points_1 = load_image_points(
        images_info, images=[item['cam_1_img'] for item in matching_pairs.values()])
    img_points_2 = load_image_points(
        images_info, images=[item['cam_2_img'] for item in matching_pairs.values()])

    _, mtx_1, dist_1, mtx_2, dist_2, R, T, _, _ = cv2.stereoCalibrate(
        np.tile(obj_points, (len(img_points_1), 1, 1)),
        img_points_1, img_points_2,
        None, None, None, None,
        (data_loader.IMAGE_INFRARED_WIDTH,
         data_loader.IMAGE_INFRARED_HEIGHT),
        criteria=STEREO_CALIBRATION_CRITERIA, flags=0)

    extrinsics[cam_1] = {
        'left_cam': cam_1,
        'right_cam': cam_2,
        'mtx_l': mtx_1.tolist(),
        'dist_l': dist_1.tolist(),
        'mtx_r': mtx_2.tolist(),
        'dist_r': dist_2.tolist(),
        'rotation': R.tolist(),
        'transition': T.tolist(),
    }


def calc_reprojection_error(cam_1, cam_2, obj_points, extrinsics, configs):
    print(f"Calculating error... {cam_1} vs {cam_2}")

    with open(configs['chessboards']) as handler:
        images_info = yaml.safe_load(handler)

    matching_pairs = find_matching_images(images_info, cam_1, cam_2)

    print(f"Matching pairs: {len(matching_pairs)}")

    img_points_1 = load_image_points(
        images_info, images=[item['cam_1_img'] for item in matching_pairs.values()])
    img_points_2 = load_image_points(
        images_info, images=[item['cam_2_img'] for item in matching_pairs.values()])
    
    mtx_1 = np.array(extrinsics[cam_1]['mtx_l'], dtype=np.float32)
    dist_1 = np.array(extrinsics[cam_1]['dist_l'], dtype=np.float32)
    mtx_2 = np.array(extrinsics[cam_1]['mtx_r'], dtype=np.float32)
    dist_2 = np.array(extrinsics[cam_1]['dist_r'], dtype=np.float32)
    R = np.array(extrinsics[cam_1]['rotation'], dtype=np.float32)
    T = np.array(extrinsics[cam_1]['transition'], dtype=np.float32)
    
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
        total_error += (error1 + error2) / 2

    # Print the average projection error
    print("Average projection error: ", total_error / len(img_points_1))


def calc_extrinsic(configs):
    obj_points = get_obj_points()

    extrinsics = {}

    cameras = configs['cameras']
    for cam1_idx in range(len(cameras)):
        for cam2_idx in range(len(cameras)):
            cam_1 = cameras[cam1_idx]
            cam_2 = cameras[cam2_idx]

            if f"{cam_1}/{cam_2}" not in ORDER_VALID:
                continue
                    
            calc_extrinsics(cam_1, cam_2, obj_points, extrinsics, configs)
            calc_reprojection_error(cam_1, cam_2, obj_points, extrinsics, configs)

    _store_artifacts(extrinsics, configs)


def _store_artifacts(artifact, configs):
    with open(configs['output_dir'], 'w') as handle:
        yaml.dump(artifact, handle)


if __name__ == "__main__":
    args = _get_arguments()
    configs = _load_configs(args.config)

    print(f"Config loaded: {configs}")

    calc_extrinsic(configs)   
