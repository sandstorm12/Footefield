import sys
sys.path.append('../')

import os
import cv2
import glob
import yaml
import argparse
import numpy as np


def _get_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-c', '--config',
        help='Path to the config file',
        type=str,
        default='configs/test_undistort_depth.yml',
    )

    args = parser.parse_args()

    return args


def _load_configs(path):
    with open(path, 'r') as yaml_file:
        configs = yaml.safe_load(yaml_file)

    return configs


if __name__ == "__main__":
    args = _get_arguments()
    configs = _load_configs(args.config)

    with open(configs['intrinsics']) as handler:
        intrinsics = yaml.safe_load(handler)

    for calib_info in configs['calibration_folders']:
        key = calib_info['camera_name']

        ret = np.array(intrinsics[key]['ret'], np.float32)
        mtx = np.array(intrinsics[key]['mtx'], np.float32)
        dist = np.array(intrinsics[key]['dist'], np.float32)

        file_paths = glob.glob(os.path.join(calib_info['path'], "infrared*"))
        for file_path in file_paths:
            img = cv2.imread(file_path, -1)
            img = np.clip(
                img.astype(np.float32) * .8, 0, 255).astype('uint8')

            undistorted_img = cv2.undistort(img, mtx, dist, None, mtx)

            cv2.imshow("Undistorted Image", undistorted_img)
            cv2.imshow("Original Image", img)
            
            if cv2.waitKey(0) == ord('q'):
                break
        
