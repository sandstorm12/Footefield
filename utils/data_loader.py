import os
import re
import glob


PATH_CALIBS = [
    "/home/hamid/Documents/footefield/data/AzureKinectRecord_0729/azure_kinect1_4_calib_snap",
    "/home/hamid/Documents/footefield/data/AzureKinectRecord_0729/azure_kinect1_5_calib_snap",
    "/home/hamid/Documents/footefield/data/AzureKinectRecord_0729/azure_kinect2_4_calib_snap",
    "/home/hamid/Documents/footefield/data/AzureKinectRecord_0729/azure_kinect3_4_calib_snap",
    "/home/hamid/Documents/footefield/data/AzureKinectRecord_0729/azure_kinect3_5_calib_snap",
]

EXPERIMENTS = {
    'a1': [
        "/home/hamid/Documents/footefield/data/AzureKinectRecord_0729/a1/azure_kinect1_4",
        "/home/hamid/Documents/footefield/data/AzureKinectRecord_0729/a1/azure_kinect1_5",
        "/home/hamid/Documents/footefield/data/AzureKinectRecord_0729/a1/azure_kinect2_4",
        "/home/hamid/Documents/footefield/data/AzureKinectRecord_0729/a1/azure_kinect3_4",
        "/home/hamid/Documents/footefield/data/AzureKinectRecord_0729/a1/azure_kinect3_5",
    ],
    'a2': [
        "/home/hamid/Documents/footefield/data/AzureKinectRecord_0729/a2/azure_kinect1_4",
        "/home/hamid/Documents/footefield/data/AzureKinectRecord_0729/a2/azure_kinect1_5",
        "/home/hamid/Documents/footefield/data/AzureKinectRecord_0729/a2/azure_kinect2_4",
        "/home/hamid/Documents/footefield/data/AzureKinectRecord_0729/a2/azure_kinect3_4",
        "/home/hamid/Documents/footefield/data/AzureKinectRecord_0729/a2/azure_kinect3_5",
    ]
}

MMPOSE_EDGES = [
    (0, 1),
    (0, 2),
    (1, 2),
    (1, 3),
    (2, 4),
    (3, 5),
    (4, 6),
    (5, 6),
    (5, 7),
    (5, 11),
    (6, 8),
    (6, 12),
    (7, 9),
    (8, 10),
    (11, 12),
    (11, 13),
    (12, 14),
    (13, 15),
    (14, 16),
]

TYPE_RGB = "TYPE_RGB"
TYPE_DEPTH = "TYPE_DEPTH"
TYPE_INFRARED = "TYPE_INFRARED"

CHESSBOARD_COLS = 8
CHESSBOARD_ROWS = 11
CHESSBOARD_SQRS = 60.

IMAGE_RGB_WIDTH = 1920
IMAGE_RGB_HEIGHT = 1080
IMAGE_INFRARED_WIDTH = 640
IMAGE_INFRARED_HEIGH = 576

PATTERN_NAME = r"(color|depth|infrared)"


def list_rgb_images(dir):
    file_paths = sorted(glob.glob(os.path.join(dir, "color*")))

    return file_paths


def list_depth_images(dir):
    file_paths = sorted(glob.glob(os.path.join(dir, "depth*")))

    return file_paths


def list_infrared_images(dir):
    file_paths = sorted(glob.glob(os.path.join(dir, "infrared*")))

    return file_paths


def list_calibration_images(dir):
    calibration_images = {"data": {}}

    calibration_images["data"][TYPE_RGB] = list_rgb_images(dir)
    calibration_images["data"][TYPE_DEPTH] = list_depth_images(dir)
    calibration_images["data"][TYPE_INFRARED] = list_infrared_images(dir)

    return calibration_images


def image_name_from_fullpath(fullpath):
    dir, name = fullpath.split("/")[-2:]

    name = name.split(".")[0]
    name = re.sub(PATTERN_NAME, "", name)

    image_name = f"{dir}/{name}"
    
    return image_name