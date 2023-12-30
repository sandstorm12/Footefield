import sys
sys.path.append('../')

import cv2
import diskcache
import numpy as np

from tqdm import tqdm
from utils import data_loader


ORDER_VALID = (
    'azure_kinect1_5_calib_snap/azure_kinect1_4_calib_snap',
    'azure_kinect1_4_calib_snap/azure_kinect3_4_calib_snap',
    'azure_kinect3_4_calib_snap/azure_kinect3_5_calib_snap',
    'azure_kinect3_5_calib_snap/azure_kinect2_4_calib_snap',
    'azure_kinect2_4_calib_snap/azure_kinect1_5_calib_snap',
)


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


def load_image_points(cache, images):
    images_info = cache['images_info']

    criteria = (cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, 30, 0.001)
    
    if not images_info:
        print("'images_info' not found.")

    if len(images_info.keys()) == 0:
        print("No images in images_info. Please run detect_chessboard first.")

    img_points = []
    for key in tqdm(images):
        camera = key.split("/")[0]

        ret, corners = images_info[key]['findchessboardcorners_rgb']
        if not ret:
            continue
        
        image_gray = cv2.imread(images_info[key]['fullpath'], cv2.IMREAD_GRAYSCALE)
        corners_refined = cv2.cornerSubPix(image_gray, corners, (5, 5), (-1, -1), criteria)

        img_points.append(corners_refined)
        width = image_gray.shape[1] # images_info[key]['width']
        height = image_gray.shape[0] # ['height']

    return img_points, width, height


def find_matching_images(images_info, cam_1, cam_2):
    images_info = cache['images_info']

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
    cache = diskcache.Cache('cache')

    print("Cache keys:", get_all_keys(cache))

    obj_points = get_obj_points()
    intrinsics = cache['intrinsics']
    images_info = cache['images_info']

    cameras = list(intrinsics.keys())
    for cam1_idx in range(len(cameras)):
        for cam2_idx in range(len(cameras)):
            cam_1 = cameras[cam1_idx]
            cam_2 = cameras[cam2_idx]

            if f"{cam_1}/{cam_2}" not in ORDER_VALID:
                continue
            
            print(f"Calibrating... {cameras[cam1_idx]}"
                  f" vs {cameras[cam2_idx]}")

            matching_pairs = find_matching_images(cache['images_info'], cam_1, cam_2)

            print(f"Matching pairs: {len(matching_pairs)}")
            if len(matching_pairs) == 0:
                continue
            
            img_points_1, width, height = load_image_points(
                cache, images=[item['cam_1_img'] for item in matching_pairs.values()])
            img_points_2, width, height = load_image_points(
                cache, images=[item['cam_2_img'] for item in matching_pairs.values()])

            for idx, key in enumerate(matching_pairs.keys()):
                image_1_addr = images_info[matching_pairs[key]['cam_1_img']]['fullpath']
                image_1 = cv2.imread(image_1_addr)
                image_2_addr = images_info[matching_pairs[key]['cam_2_img']]['fullpath']
                image_2 = cv2.imread(image_2_addr)

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

