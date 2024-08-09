"""
cache `images_info` key: camera_folder value: findChessboardCorners_output
"""

import sys
sys.path.append('../')

import cv2
import yaml
import argparse
import diskcache
import numpy as np

from utils import data_loader

from tqdm import tqdm
from threading import Thread


criteria = (cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, 30, 0.001)


def _get_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-c', '--config',
        help='Path to the config file',
        type=str,
        default='configs/detect_chessboard_depth.yml',
    )

    args = parser.parse_args()

    return args


def _load_configs(path):
    with open(path, 'r') as yaml_file:
        configs = yaml.safe_load(yaml_file)

    print(configs)

    return configs


def extract_chessboardcorners(image_paths, images_info, camera_name, display=False):
    success_count = 0

    bar = tqdm(image_paths)
    bar.set_description(camera_name)
    for image_path in bar:
        image_name = data_loader.image_name_from_fullpath(image_path)
        image_name = f"{camera_name}/{image_name}"

        gray_org = cv2.imread(image_path, -1)

        for thr in reversed([1., .8, .5, .2]):
            gray = np.clip(
                gray_org.astype(np.float32) * thr, 0, 255).astype('uint8')
            # TODO: Move chessboard dimensions into the config file
            ret, corners = cv2.findChessboardCorners(
                gray, (data_loader.CHESSBOARD_COLS,
                        data_loader.CHESSBOARD_ROWS))
            
            if ret:
                break
        
        if ret:
            corners = cv2.cornerSubPix(
                gray, corners, (3, 3), (-1, -1), criteria)
            corners = corners.tolist()
        else:
            corners = []

        images_info[image_name] = {
            "fullpath_infrared": image_path,
            "findchessboardcorners_infrared": [ret, corners],
        }

        if display:
            if not _display(gray_org, corners):
                break

        if ret:
            success_count += 1


    print(f"Found {success_count} chessboards from " +
        f"{len(image_paths)} image for {camera_name}")
    
    return images_info


def _display(image, corners):
    for point in corners:
        x = int(point[0][0])
        y = int(point[0][1])

        cv2.circle(
            image, (x, y), 5, (123, 105, 34),
            thickness=-1, lineType=8) 

    cv2.imshow("image", image)
    key = cv2.waitKey(0)
    if key == ord('q'):
        return False
    
    return True


def calculate_total_success_dets(images_info):
    total_success_counter = 0
    for key in images_info.keys():
        if images_info[key]['findchessboardcorners_infrared'][0]:
            total_success_counter += 1
    print(f"Grand num of found chessboards: {total_success_counter}")


def _store_artifacts(images_info, configs):
    with open(configs['output_dir'], 'w') as handle:
        yaml.dump(images_info, handle)


def detect_chessboards(configs):
    images_info = {}

    processes = []
    for calib_info in configs['calibration_folders']:
        calibration_images = data_loader.list_calibration_images(calib_info['path'])
        process = Thread(
            target=extract_chessboardcorners,
            args=(calibration_images['data'][data_loader.TYPE_INFRARED],
                  images_info, calib_info['camera_name'], configs['display']))
        process.start()
        processes.append(process)

        if not configs['parallel']:
            process.join()

    for process in processes:
        process.join()
    
    _store_artifacts(images_info, configs)

    calculate_total_success_dets(images_info)


if __name__ == "__main__":
    args = _get_arguments()
    configs = _load_configs(args.config)

    print(f"Config loaded: {configs}")

    detect_chessboards(configs)
