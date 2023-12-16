import sys
sys.path.append('../')

import os
import cv2
import diskcache
import numpy as np
from utils import data_loader


CHESSBOARD_COLS = 8
CHESSBOARD_ROWS = 11
CHESSBOARD_SQRS = 60.


for path_calib in data_loader.PATH_CALIBS:
    calibration_images = data_loader.list_calibration_images(path_calib)

    paths_rgb_sorted = sorted(calibration_images['data']['TYPE_RGB'])
    paths_dpt_sorted = sorted(calibration_images['data']['TYPE_DEPTH'])

    for idx in range(len(calibration_images['data']['TYPE_RGB'])):
        print(paths_rgb_sorted[idx])
        print(paths_dpt_sorted[idx])
        image_rgb = cv2.imread(paths_rgb_sorted[idx])
        image_depth = cv2.imread(paths_dpt_sorted[idx],
                                 cv2.IMREAD_GRAYSCALE)
        
        image_depth = (image_depth - np.min(image_depth)) / (np.max(image_depth) - np.min(image_depth))
        image_depth *= 255
        
        print(image_rgb.shape)
        print(image_depth.shape)

        image_depth = cv2.resize(image_depth, (1920, 1080))
        image_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2GRAY)

        image_rgb = (image_rgb * .5 + image_depth * .5).astype(np.uint8)

        print(np.min(image_rgb), np.mean(image_rgb), np.max(image_rgb))

        # img = cv2.imread(image_depth, cv2.IMREAD_GRAYSCALE)
        # img = cv2.imread(image_depth)

        cv2.imshow('rgb', image_rgb)
        # cv2.imshow('depth', image_depth)
        if cv2.waitKey(0) == ord('q'):
            break