"""
cache `images_info` key: camera_folder value: findChessboardCorners_output
"""

import sys
sys.path.append('../')

import cv2
import yaml
import argparse

from tqdm import tqdm
from threading import Thread

from utils import data_loader


criteria = (cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, 30, 0.001)


def _get_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-c', '--config',
        help='Path to the config file',
        type=str,
        required=True,
    )

    args = parser.parse_args()

    return args


def _load_configs(path):
    with open(path, 'r') as yaml_file:
        configs = yaml.safe_load(yaml_file)

    return configs


def extract_chessboardcorners(path_video, images_info, camera_name, configs):
    cap = cv2.VideoCapture(path_video)

    if not images_info.__contains__(camera_name):
        images_info[camera_name] = []

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    success_count = 0

    bar = tqdm(range(frame_count))
    bar.set_description(camera_name)
    for frame_idx in bar:
        _, image = cap.read()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        ret, corners = cv2.findChessboardCorners(
            gray, (data_loader.CHESSBOARD_COLS, data_loader.CHESSBOARD_ROWS),
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)
        
        if ret:
            corners = cv2.cornerSubPix(gray, corners, (6, 6), (-1, -1), criteria)
            corners = corners.tolist()
        else:
            corners = []

        
        images_info[camera_name].append(
            [ret, corners]
        )

        if configs['display']:
            if not _display(image, corners):
                break

        if ret:
            success_count += 1

    print(f"Found {success_count} chessboards from " +
        f"{frame_count} image for {camera_name}")
    

def _display(image, corners):
    for point in corners:
        x = int(point[0][0])
        y = int(point[0][1])

        cv2.circle(
            image, (x, y), 5, (123, 105, 34),
            thickness=-1, lineType=8) 

    cv2.imshow("image", image)
    key = cv2.waitKey(1)
    if key == ord('q'):
        return False
    
    return True


def calculate_total_success_dets(images_info):
    total_success_counter = 0
    for key in images_info.keys():
        for item in images_info[key]:
            if item[0]:
                total_success_counter += 1
    print(f"Grand num of found chessboards: {total_success_counter}")


def _store_artifacts(images_info, configs):
    with open(configs['output_dir'], 'w') as handle:
        yaml.dump(images_info, handle)


def detect_chessboards(configs):
    images_info = {}

    processes = []
    for camera in configs['calibration_videos']:
        process = Thread(
            target=extract_chessboardcorners,
            args=(configs['calibration_videos'][camera]['path'], images_info,
                  camera, configs))
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
    