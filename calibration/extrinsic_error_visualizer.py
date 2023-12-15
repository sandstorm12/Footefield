import cv2
import diskcache
import numpy as np

import detect_chessboard

from tqdm import tqdm


ORDER_VALID = (
    'azure_kinect1_5_calib_snap/azure_kinect1_4_calib_snap',
    'azure_kinect1_4_calib_snap/azure_kinect3_4_calib_snap',
    'azure_kinect3_4_calib_snap/azure_kinect3_5_calib_snap',
    'azure_kinect3_5_calib_snap/azure_kinect2_4_calib_snap',
    'azure_kinect2_4_calib_snap/azure_kinect1_5_calib_snap',
)


def get_obj_points():
    cols = detect_chessboard.CHESSBOARD_COLS
    rows = detect_chessboard.CHESSBOARD_ROWS
    square_size = detect_chessboard.CHESSBOARD_SQRS

    obj_points = np.zeros((cols * rows, 3), np.float32)
    obj_points[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square_size

    return obj_points


def get_all_keys(cache):
    keys_all = cache._sql('SELECT key FROM Cache').fetchall()
    keys_all = [item[0] for item in keys_all]

    return keys_all


def load_image_points(cache, images):
    images_info = cache['images_info']

    if not images_info:
        print("'images_info' not found.")

    if len(images_info.keys()) == 0:
        print("No images in images_info. Please run detect_chessboard first.")

    img_points = []
    img_paths = []
    for key in tqdm(images):
        ret, corners = images_info[key]['findchessboardcorners_rgb']
        if not ret:
            continue
        
        img_points.append(corners)
        width = images_info[key]['width']
        height = images_info[key]['height']
        img_paths.append(images_info[key]['fullpath'])

    return img_points, width, height, img_paths


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


def visualize_points(img_paths_1, img_paths_2, imgpoints1_projected, imgpoints2_projected):
    image_1 = cv2.imread(img_paths_1[i])
    image_2 = cv2.imread(img_paths_2[i])
    for idx_point, point in enumerate(imgpoints1_projected):
        x = int(point[0][0])
        y = int(point[0][1])

        cv2.circle(
            image_1, (x, y), 10, (0, 0, 0),
            thickness=-1, lineType=8)

        cv2.putText(
            image_1, str(idx_point), (x - 5, y + 5),
            cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), thickness=2)
    
    for idx_point, point in enumerate(imgpoints2_projected):
        x = int(point[0][0])
        y = int(point[0][1])

        cv2.circle(
            image_2, (x, y), 10, (0, 0, 0),
            thickness=-1, lineType=8)

        cv2.putText(
            image_2, str(idx_point), (x - 5, y + 5),
            cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), thickness=2)
        
    return image_1, image_2


# This is too long
if __name__ == "__main__":
    cache = diskcache.Cache('cache')

    print("Cache keys:", get_all_keys(cache))

    obj_points = get_obj_points()
    intrinsics = cache['intrinsics']

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

            img_points_1, width, height, img_paths_1 = load_image_points(
                cache, images=[item['cam_1_img'] for item in matching_pairs.values()])
            img_points_2, width, height, img_paths_2 = load_image_points(
                cache, images=[item['cam_2_img'] for item in matching_pairs.values()])

            mtx_1 = intrinsics[cam_1]['mtx']
            dist_1 = intrinsics[cam_1]['dist']
            rvecs_1 = intrinsics[cam_1]['rvecs']
            tvecs_1 = intrinsics[cam_1]['tvecs']

            mtx_2 = intrinsics[cam_2]['mtx']
            dist_2 = intrinsics[cam_2]['dist']
            rvecs_2 = intrinsics[cam_2]['rvecs']
            tvecs_2 = intrinsics[cam_2]['tvecs']

            stereocalibration_criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 1000, 1e-6)
            
            # # STERO CALIBRATION
            ret, mtx_1, dist_1, mtx_2, dist_2, R, T, E, F = cv2.stereoCalibrate(
                np.tile(obj_points, (len(img_points_1), 1, 1)),
                img_points_1, img_points_2,
                mtx_1, dist_1, mtx_2, dist_2, (width, height),
                criteria=stereocalibration_criteria, flags=0)

            total_error = 0
            for i in range(len(img_points_1)):
                ret, rvec_l, tvec_l, _ = cv2.solvePnPRansac(obj_points, img_points_1[i], mtx_1, dist_1)
                # ret, rvec_r, tvec_r = cv2.solvePnP(obj_points, img_points_2[i], mtx_2, dist_2)
                rvec_r, tvec_r = cv2.composeRT(rvec_l, tvec_l, cv2.Rodrigues(R)[0], T)[:2]

                imgpoints1_projected, _ = cv2.projectPoints(obj_points, rvec_l, tvec_l, mtx_1, dist_1)
                imgpoints2_projected, _ = cv2.projectPoints(obj_points, rvec_r, tvec_r, mtx_2, dist_2)

                error1 = cv2.norm(img_points_1[i], imgpoints1_projected, cv2.NORM_L2) / len(imgpoints1_projected)
                error2 = cv2.norm(img_points_2[i], imgpoints2_projected, cv2.NORM_L2) / len(imgpoints2_projected)
                print(f"Errors: cam1 {error1} cam2 {error2}")
                total_error += (error1 + error2) / 2

                image_l, image_r = visualize_points(
                    img_paths_1, img_paths_2,
                    imgpoints1_projected, imgpoints2_projected)

                cv2.imshow("image1", cv2.resize(image_l, (960, 1080)))
                cv2.imshow("image2", cv2.resize(image_r, (960, 1080)))
                key = cv2.waitKey(0)
                if key == ord('q'):
                    break

            # Print the average projection error
            print("Average projection error: ", total_error / len(img_points_1))
