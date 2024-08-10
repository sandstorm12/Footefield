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
        default='configs/rgb_depth_error_visualizer.yml',
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


def find_rgb_depth_images(images_info_rgb, images_info_depth, cam_1):
    image_paths_rgb = []
    image_paths_infrared = []
    image_points_rgb = []
    image_points_infrared = []
    for key in images_info_rgb.keys():
        if key.split("/")[0] == cam_1:
            points_found_rgb, points_rgb = \
                images_info_rgb[key]['findchessboardcorners_rgb']
            points_found_infrared, points_infrared = \
                images_info_depth[key]['findchessboardcorners_infrared']
            if points_found_rgb and points_found_infrared:
                image_paths_rgb.append(images_info_rgb[key]['fullpath_rgb'])
                image_paths_infrared.append(
                    images_info_depth[key]['fullpath_infrared'])
                image_points_rgb.append(points_rgb)
                image_points_infrared.append(points_infrared)

    rgb_depth_pairs = {
        "image_paths_rgb": image_paths_rgb,
        "image_paths_infrared": image_paths_infrared,
        "image_points_rgb": np.array(image_points_rgb, np.float32),
        "image_points_infrared": np.array(image_points_infrared, np.float32),
    }
            
    return rgb_depth_pairs


def calc_reprojection_error(cam_1, obj_points, depth_extrinsics, configs):
    print(f"Calibrating... {cam_1}")

    with open(configs['chessboards_rgb']) as handler:
        images_info_rgb = yaml.safe_load(handler)
    with open(configs['chessboards_depth']) as handler:
        images_info_depth = yaml.safe_load(handler)

    rgb_depth_pairs = find_rgb_depth_images(
        images_info_rgb, images_info_depth, cam_1)

    print(f"Matching pairs: {len(rgb_depth_pairs['image_points_rgb'])}")
    
    mtx_1 = np.array(depth_extrinsics[cam_1]['mtx_l_rgb'], np.float32)
    dist_1 = np.array(depth_extrinsics[cam_1]['dist_l_rgb'], np.float32)
    mtx_2 = np.array(depth_extrinsics[cam_1]['mtx_r_depth'], np.float32)
    dist_2 = np.array(depth_extrinsics[cam_1]['dist_r_depth'], np.float32)
    R = np.array(depth_extrinsics[cam_1]['rotation'], np.float32)
    T = np.array(depth_extrinsics[cam_1]['transition'], np.float32)
    
    for idx in range(len(rgb_depth_pairs['image_points_rgb'])):
        _, rvec_l, tvec_l = cv2.solvePnP(
            obj_points, rgb_depth_pairs['image_points_rgb'][idx],
            mtx_1, dist_1)
        rvec_r, tvec_r = cv2.composeRT(
            rvec_l, tvec_l, cv2.Rodrigues(R)[0], T)[:2]

        imgpoints2_projected, _ = cv2.projectPoints(
            obj_points, rvec_r, tvec_r, mtx_2, dist_2)
        
        img_inf = cv2.imread(rgb_depth_pairs['image_paths_infrared'][idx], -1)

        # Remove magic number .8
        img_inf = np.clip(
            img_inf.astype(np.float32) * .8, 0, 255).astype('uint8')
        img_inf = cv2.resize(
            img_inf,
            (data_loader.IMAGE_INFRARED_WIDTH,
             data_loader.IMAGE_INFRARED_HEIGHT))
        for idx_point, point in enumerate(imgpoints2_projected):
            x = int(point[0][0])
            y = int(point[0][1])

            cv2.circle(
                img_inf, (x, y), 10, (0, 0, 0),
                thickness=-1, lineType=8)

            cv2.putText(
                img_inf, str(idx_point), (x - 5, y + 5),
                cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), thickness=2)
            
        cv2.imshow('Infrared', img_inf)
        if cv2.waitKey(0) == ord('q'):
            break


if __name__ == "__main__":
    args = _get_arguments()
    configs = _load_configs(args.config)

    print(f"Config loaded: {configs}")
    
    obj_points = get_obj_points()
    with open(configs['rgb_depth_extrinsic']) as handler:
        depth_extrinsics = yaml.safe_load(handler)

    cameras = configs['cameras']
    for cam1_idx in range(len(cameras)):
        cam_1 = cameras[cam1_idx]

        calc_reprojection_error(cam_1, obj_points, depth_extrinsics, configs)
