import os
import re
import cv2
import glob


PATH_CALIBS = [
    "/home/hamid/Documents/phd/footefield/data/AzureKinectRecord_0729/azure_kinect1_4_calib_snap",
    "/home/hamid/Documents/phd/footefield/data/AzureKinectRecord_0729/azure_kinect1_5_calib_snap",
    "/home/hamid/Documents/phd/footefield/data/AzureKinectRecord_0729/azure_kinect2_4_calib_snap",
    "/home/hamid/Documents/phd/footefield/data/AzureKinectRecord_0729/azure_kinect3_4_calib_snap",
    "/home/hamid/Documents/phd/footefield/data/AzureKinectRecord_0729/azure_kinect3_5_calib_snap",
]

EXPERIMENTS = {
    'a1': {
        "azure_kinect1_4_calib_snap": "/home/hamid/Documents/phd/footefield/data/AzureKinectRecord_0729/a1/azure_kinect1_4",
        "azure_kinect1_5_calib_snap": "/home/hamid/Documents/phd/footefield/data/AzureKinectRecord_0729/a1/azure_kinect1_5",
        "azure_kinect2_4_calib_snap": "/home/hamid/Documents/phd/footefield/data/AzureKinectRecord_0729/a1/azure_kinect2_4",
        "azure_kinect3_4_calib_snap": "/home/hamid/Documents/phd/footefield/data/AzureKinectRecord_0729/a1/azure_kinect3_4",
        "azure_kinect3_5_calib_snap": "/home/hamid/Documents/phd/footefield/data/AzureKinectRecord_0729/a1/azure_kinect3_5",
    },
    'a2': {
        "azure_kinect1_4_calib_snap": "/home/hamid/Documents/phd/footefield/data/AzureKinectRecord_0729/a2/azure_kinect1_4",
        "azure_kinect1_5_calib_snap": "/home/hamid/Documents/phd/footefield/data/AzureKinectRecord_0729/a2/azure_kinect1_5",
        "azure_kinect2_4_calib_snap": "/home/hamid/Documents/phd/footefield/data/AzureKinectRecord_0729/a2/azure_kinect2_4",
        "azure_kinect3_4_calib_snap": "/home/hamid/Documents/phd/footefield/data/AzureKinectRecord_0729/a2/azure_kinect3_4",
        "azure_kinect3_5_calib_snap": "/home/hamid/Documents/phd/footefield/data/AzureKinectRecord_0729/a2/azure_kinect3_5",
    }
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
IMAGE_INFRARED_HEIGHT = 576

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


def downsample_keep_aspect_ratio(img, size):
    aspect_ratio_org = img.shape[0] / img.shape[1]
    aspect_ratio_new = size[1] / size[0]

    if aspect_ratio_new > aspect_ratio_org:
        width_resize = int(img.shape[1] * (size[1] / img.shape[0]))
        height_resize = size[1]
        center = width_resize // 2

        img_resize = cv2.resize(img, (width_resize, height_resize))
        img_resize = img_resize[
            :, center - size[0] // 2:center + size[0] // 2]
    else:
        width_resize = size[0]
        height_resize = int(img.shape[0] * (size[0] / img.shape[1]))
        center = height_resize // 2

        img_resize = cv2.resize(img, (width_resize, height_resize))
        img_resize = img_resize[
            center - size[0] // 2:center + size[0] // 2, :]

    return img_resize
