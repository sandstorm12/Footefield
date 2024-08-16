import sys
sys.path.append('../')

import cv2
import yaml
import argparse
import numpy as np

from tqdm import tqdm
from utils import data_loader


ORDER_VALID = (
    'cam1_5/cam1_4',
    'cam1_4/cam3_4',
    'cam3_4/cam3_5',
    'cam3_5/cam2_4',
    'cam2_4/cam1_5',
)

def _get_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-c', '--config',
        help='Path to the config file',
        type=str,
        default='configs/stereo_chessboard_visualizer.yml',
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


def get_all_keys(cache):
    keys_all = cache._sql('SELECT key FROM Cache').fetchall()
    keys_all = [item[0] for item in keys_all]

    return keys_all


def load_image_points(images_info, images):
    if len(images_info.keys()) == 0:
        print("No images in images_info. Please run detect_chessboard first.")

    img_points = []
    for key in tqdm(images):
        ret, corners = images_info[key]['findchessboardcorners_rgb']
        if not ret:
            continue
        
        img_points.append(corners)

    return img_points


def find_matching_images(images_info, cam_1, cam_2):
    matching_pairs = {}
    for image in images_info.keys():
        img_name = image.split("/")[1]
        cam_1_img = f"{cam_1}/{img_name}"
        cam_2_img = f"{cam_2}/{img_name}"
        if images_info.__contains__(cam_1_img) \
            and images_info.__contains__(cam_2_img):
            points_found_1, _ = images_info[cam_1_img]['findchessboardcorners_rgb']
            points_found_2, _ = images_info[cam_2_img]['findchessboardcorners_rgb']
            if points_found_1 and points_found_2:
                matching_pairs[img_name] = {
                    "cam_1_img": cam_1_img,
                    "cam_2_img": cam_2_img,
                }
            
    return matching_pairs


# This is too long. trim this function.
if __name__ == "__main__":
    args = _get_arguments()
    configs = _load_configs(args.config)

    print(f"Config loaded: {configs}")

    obj_points = get_obj_points()
    with open(configs['chessboards_rgb']) as handler:
        images_info = yaml.safe_load(handler)

    cameras = configs['cameras']
    for cam1_idx in range(len(cameras)):
        for cam2_idx in range(len(cameras)):
            cam_1 = cameras[cam1_idx]
            cam_2 = cameras[cam2_idx]

            if f"{cam_1}/{cam_2}" not in ORDER_VALID:
                continue
            
            print(f"Calibrating... {cameras[cam1_idx]}"
                  f" vs {cameras[cam2_idx]}")

            matching_pairs = find_matching_images(images_info, cam_1, cam_2)

            print(f"Matching pairs: {len(matching_pairs)}")
            if len(matching_pairs) == 0:
                continue
            
            img_points_1 = load_image_points(
                images_info, images=[item['cam_1_img'] for item in matching_pairs.values()])
            img_points_2 = load_image_points(
                images_info, images=[item['cam_2_img'] for item in matching_pairs.values()])

            for idx, key in enumerate(matching_pairs.keys()):
                image_1_addr = images_info[matching_pairs[key]['cam_1_img']]['fullpath_rgb']
                image_1 = cv2.imread(image_1_addr)
                image_2_addr = images_info[matching_pairs[key]['cam_2_img']]['fullpath_rgb']
                image_2 = cv2.imread(image_2_addr)

                image_1 = data_loader.downsample_keep_aspect_ratio(
                    image_1,
                    (data_loader.IMAGE_INFRARED_WIDTH,
                     data_loader.IMAGE_INFRARED_HEIGHT))
                
                image_2 = data_loader.downsample_keep_aspect_ratio(
                    image_2,
                    (data_loader.IMAGE_INFRARED_WIDTH,
                     data_loader.IMAGE_INFRARED_HEIGHT))

                for idx_point, point in enumerate(img_points_1[idx]):
                    x = int(point[0][0])
                    y = int(point[0][1])

                    cv2.circle(
                        image_1, (x, y), 10, (0, 0, 0),
                        thickness=-1, lineType=8)

                    cv2.putText(
                        image_1, str(idx_point), (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), thickness=2)
                
                for idx_point, point in enumerate(img_points_2[idx]):
                    x = int(point[0][0])
                    y = int(point[0][1])

                    cv2.circle(
                        image_2, (x, y), 10, (0, 0, 0),
                        thickness=-1, lineType=8)

                    cv2.putText(
                        image_2, str(idx_point), (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), thickness=2)

                cv2.imshow("image1", cv2.resize(image_1, (960, 1080)))
                cv2.imshow("image2", cv2.resize(image_2, (960, 1080)))
                key = cv2.waitKey(0)
                if key == ord('q'):
                    break

