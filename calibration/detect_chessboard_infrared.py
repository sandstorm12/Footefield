"""
cache `images_info` key: camera_folder value: findChessboardCorners_output
"""

import sys
sys.path.append('../')

import os
import re
import cv2
import diskcache
import numpy as np
import matplotlib.pyplot as plt
from utils import data_loader

from tqdm import tqdm
from threading import Thread


_PARALLEL = True
_DISPLAY = False

criteria = (cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, 30, 0.001)


def extract_chessboardcorners(image_paths, images_info, display=False):
    # print(f"Processing --> {os.path.dirname(image_paths[0])}")
    
    camera_name = image_paths[0].split("/")[-2]

    success_count = 0

    bar = tqdm(image_paths)
    bar.set_description(camera_name)
    for image_path in bar:
        image_name = data_loader.image_name_from_fullpath(image_path)

        if not images_info.__contains__(image_name) or (images_info.__contains__(image_name) and images_info[image_name]['findchessboardcorners_rgb'][0]): 
            gray_org = cv2.imread(image_path, -1)

            for thr in [.8, .5, .2, .1]:
                gray = np.clip(gray_org.astype(np.float32) * thr, 0, 255).astype('uint8')
                gray = cv2.resize(gray, (1920, 1080))
                ret, corners = cv2.findChessboardCorners(
                    gray, (data_loader.CHESSBOARD_COLS, data_loader.CHESSBOARD_ROWS))
                
                if ret:
                    break
            
            if ret:
                corners = cv2.cornerSubPix(
                    gray, corners, (11, 11), (-1, -1), criteria)
        else:
            ret = False
            corners = None

        if images_info.__contains__(image_name):
            images_info[image_name]['findchessboardcorners_infrared'] = \
                (ret, corners)
            images_info[image_name]['fullpath_infrared'] = image_path
        else:
            images_info[image_name] = {
                "fullpath_infrared": image_path,
                "findchessboardcorners_infrared": (ret, corners),
                "width": gray.shape[1],
                "height": gray.shape[0],
            }

        # print(f"{chessboard[0]} --> {image_path}")

        if display:
            print("ret", ret)
            if ret:
                for point in corners:
                    x = int(point[0][0])
                    y = int(point[0][1])

                    cv2.circle(
                        gray, (x, y), 10, (123, 105, 34),
                        thickness=-1, lineType=8)

                cv2.imshow("gray", gray)
                key = cv2.waitKey(0)
                if key == ord('q'):
                    break

        if ret:
            success_count += 1


    print(f"Found {success_count} chessboards from " +
        f"{len(image_paths)} image for {camera_name}")
    
    return images_info


def calculate_total_success_dets(cache):
    total_success_counter = 0
    for key in cache['images_info'].keys():
        if cache['images_info'][key]['findchessboardcorners_infrared'][0]:
            total_success_counter += 1
    print(f"Grand num of found chessboards: {total_success_counter}")


def detect_chessboards():
    cache = diskcache.Cache('cache')
    
    cache_available = cache.__contains__('images_info')
    print(
        f"Images_info available: {cache_available}")
    
    if cache_available:
        images_info = cache['images_info']
    else:
        images_info = {}

    processes = []
    for path_calib in data_loader.PATH_CALIBS:
        calibration_images = data_loader.list_calibration_images(path_calib)
        process = Thread(
            target=extract_chessboardcorners,
            args=(calibration_images['data'][data_loader.TYPE_INFRARED],
                  images_info, _DISPLAY))
        process.start()
        processes.append(process)

        if not _PARALLEL:
            process.join()

    for process in processes:
        process.join()

    cache['images_info'] = images_info

    calculate_total_success_dets(cache)


if __name__ == "__main__":
    detect_chessboards()
