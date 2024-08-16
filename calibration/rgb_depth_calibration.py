import sys
sys.path.append('../')

import cv2
import yaml
import argparse
import numpy as np

from utils import data_loader


STEREO_CALIBRATION_CRITERIA = (
    cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS,
    1000, 1e-6)


def _get_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-c', '--config',
        help='Path to the config file',
        type=str,
        default='configs/rgb_depth_calibration.yml',
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


def find_rgb_depth_images(images_info_rgb, images_info_depth, camera):
    image_points_rgb = []
    image_points_infrared = []
    for key in images_info_rgb.keys():
        if key.split("/")[0] == camera:
            points_found_rgb, points_rgb = \
                images_info_rgb[key]['findchessboardcorners_rgb']
            points_found_infrared, points_infrared = \
                images_info_depth[key]['findchessboardcorners_infrared']
            if points_found_rgb and points_found_infrared:
                image_points_rgb.append(points_rgb)
                image_points_infrared.append(points_infrared)

    rgb_depth_pairs = {
        "image_points_rgb": np.array(image_points_rgb, np.float32),
        "image_points_infrared": np.array(image_points_infrared, np.float32),
    }
            
    return rgb_depth_pairs


def calc_depth_rgb_match(camera, obj_points, depth_extrinsics, configs):
    print(f"Calibrating... {camera}")

    with open(configs['chessboards_rgb']) as handler:
        images_info_rgb = yaml.safe_load(handler)
    with open(configs['chessboards_depth']) as handler:
        images_info_depth = yaml.safe_load(handler)

    rgb_depth_pairs = find_rgb_depth_images(
        images_info_rgb, images_info_depth, camera)

    print(f"Matching pairs: {len(rgb_depth_pairs['image_points_rgb'])}")

    _, mtx_1, dist_1, mtx_2, dist_2, R, T, _, _ = cv2.stereoCalibrate(
        np.tile(obj_points, (len(rgb_depth_pairs['image_points_rgb']), 1, 1)),
        rgb_depth_pairs['image_points_rgb'],
        rgb_depth_pairs['image_points_infrared'],
        None, None, None, None,
        (data_loader.IMAGE_RGB_WIDTH,
         data_loader.IMAGE_RGB_HEIGHT),
        criteria=STEREO_CALIBRATION_CRITERIA, flags=0)

    depth_extrinsics[camera] = {
        'mtx_l_rgb': mtx_1.tolist(),
        'dist_l_rgb': dist_1.tolist(),
        'mtx_r_depth': mtx_2.tolist(),
        'dist_r_depth': dist_2.tolist(),
        'rotation': R.tolist(),
        'transition': T.tolist(),
    }


def calc_reprojection_error(camera, obj_points, depth_extrinsics, configs):
    print(f"Calculating reprojection error... {camera}")

    with open(configs['chessboards_rgb']) as handler:
        images_info_rgb = yaml.safe_load(handler)
    with open(configs['chessboards_depth']) as handler:
        images_info_depth = yaml.safe_load(handler)

    rgb_depth_pairs = find_rgb_depth_images(
        images_info_rgb, images_info_depth, camera)
    
    mtx_1 = np.array(depth_extrinsics[camera]['mtx_l_rgb'], np.float32)
    dist_1 = np.array(depth_extrinsics[camera]['dist_l_rgb'], np.float32)
    mtx_2 = np.array(depth_extrinsics[camera]['mtx_r_depth'], np.float32)
    dist_2 = np.array(depth_extrinsics[camera]['dist_r_depth'], np.float32)
    R = np.array(depth_extrinsics[camera]['rotation'], np.float32)
    T = np.array(depth_extrinsics[camera]['transition'], np.float32)
    
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


def _store_artifacts(artifact, configs):
    with open(configs['output_dir'], 'w') as handle:
        yaml.dump(artifact, handle)


if __name__ == "__main__":
    args = _get_arguments()
    configs = _load_configs(args.config)

    print(f"Config loaded: {configs}")

    obj_points = get_obj_points()
    depth_extrinsics = {}

    cameras = configs['cameras']
    for camera in cameras:
        calc_depth_rgb_match(camera, obj_points, depth_extrinsics, configs)
        calc_reprojection_error(camera, obj_points, depth_extrinsics, configs)

    _store_artifacts(depth_extrinsics, configs)
