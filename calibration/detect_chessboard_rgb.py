"""
cache `images_info` key: camera_folder value: findChessboardCorners_output
"""

import sys
sys.path.append('../')

import re
import cv2
import diskcache
from utils import data_loader

from tqdm import tqdm
from threading import Thread


_PARALLEL = True
_DISPLAY = False

criteria = (cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, 30, 0.001)

# TODO: Shorten
def extract_chessboardcorners(image_paths, images_info, display=False):
    camera_name = image_paths[0].split("/")[-2]

    success_count = 0

    bar = tqdm(image_paths)
    bar.set_description(camera_name)
    for image_path in bar:
        image_name = data_loader.image_name_from_fullpath(image_path)
        image = cv2.imread(image_path)
        image = data_loader.downsample_keep_aspect_ratio(
            image,
            (data_loader.IMAGE_INFRARED_WIDTH,
             data_loader.IMAGE_INFRARED_HEIGHT))
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        ret, corners = cv2.findChessboardCorners(
            gray, (data_loader.CHESSBOARD_COLS, data_loader.CHESSBOARD_ROWS),
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)
        
        if ret:
            corners = cv2.cornerSubPix(gray, corners, (3, 3), (-1, -1), criteria)

        if images_info.__contains__(image_name):
            images_info[image_name]['findchessboardcorners_rgb'] = \
                (ret, corners)
            images_info[image_name]['fullpath_rgb'] = image_path
        else:
            images_info[image_name] = {
                "fullpath_rgb": image_path,
                "findchessboardcorners_rgb": (ret, corners),
            }

        if display:
            if ret:
                for point in corners:
                    x = int(point[0][0])
                    y = int(point[0][1])

                    cv2.circle(
                        image, (x, y), 5, (123, 105, 34),
                        thickness=-1, lineType=8) 

            cv2.imshow("image", image)
            key = cv2.waitKey(0)
            if key == ord('q'):
                break

        if ret:
            success_count += 1

    print(f"Found {success_count} chessboards from " +
        f"{len(image_paths)} image for {camera_name}")


def calculate_total_success_dets(cache):
    total_success_counter = 0
    for key in cache['images_info'].keys():
        if cache['images_info'][key]['findchessboardcorners_rgb'][0]:
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
            args=(calibration_images['data'][data_loader.TYPE_RGB],
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
    