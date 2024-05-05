import sys
sys.path.append('../')

import os
import cv2
import pickle
import diskcache
import numpy as np

from tqdm import tqdm
from utils import data_loader


DIR_INPUT = "./keypoints_3d_ba"
DIR_OUTPUT = "./videos_skeleton_ba"
# PARAM_OUTPUT_SIZE = (640, 576)
PARAM_OUTPUT_SIZE = (1920, 1080)
PARAM_OUTPUT_FPS = 5.0


def get_parameters(params):
    mtx = params['mtx']
    dist = params['dist']
    rotation = params['rotation']
    translation = params['translation']

    extrinsics = np.identity(4, dtype=float)
    extrinsics[:3, :3] = rotation
    extrinsics[:3, 3] = translation

    return mtx, dist, extrinsics


# Implemented by Gemini
def project_3d_to_2d(camera_matrix, dist_coeffs, rvec, tvec, object_points):
    image_points, _ = cv2.projectPoints(object_points, rvec, tvec,
                                        camera_matrix, dist_coeffs)

    image_points = image_points.squeeze()

    return image_points


def get_video_writer(experiment, camera):
    if not os.path.exists(DIR_OUTPUT):
        os.mkdir(DIR_OUTPUT)

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    writer = cv2.VideoWriter(
        os.path.join(
            DIR_OUTPUT,
            f'visualizer_skeleton_video_{experiment}_{camera}.avi'
        ),
        fourcc,
        PARAM_OUTPUT_FPS,
        PARAM_OUTPUT_SIZE
    )
    
    return writer


def write_video(poses_2d, experiment, camera, params, cache):
    img_rgb_paths = data_loader.list_rgb_images(os.path.join(dir, "color"))

    mtx, dist, _ = get_parameters(params)

    writer = get_video_writer(experiment, camera)
    for idx, t in enumerate(poses_2d.reshape(poses_2d.shape[0], -1, 2)):
        img_rgb = cv2.imread(img_rgb_paths[idx])

        img_rgb = cv2.undistort(img_rgb, mtx, dist, None, None)
        for point in t:
            cv2.circle(img_rgb, (int(point[0]), int(point[1])),
                       3, (0, 255, 0), -1)

        connections = np.concatenate(
            (np.array(data_loader.HALPE_EDGES),
             np.array(data_loader.HALPE_EDGES) + 26))
        for connection in connections:
            cv2.line(img_rgb,
                    (int(t[connection[0]][0]), int(t[connection[0]][1])),
                    (int(t[connection[1]][0]), int(t[connection[1]][1])),
                    (255, 255, 255), 1)

        writer.write(img_rgb)


def poses_3d_2_2d(poses_3d, params):
    poses_shape = list(poses_3d.shape)
    poses_shape[-1] = 2
    
    mtx = params['mtx']
    dist = params['dist']
    rotation = params['rotation']
    translation = params['translation']
    poses_2d = project_3d_to_2d(
        mtx, None,
        rotation,
        translation,
        poses_3d.reshape(-1, 3))
    poses_2d[:, 1] = poses_2d[:, 1]
    poses_2d = poses_2d.reshape(poses_shape)

    return poses_2d


# TODO: Move the cameras somewhere else
cam24 = 'azure_kinect2_4_calib_snap'
cam15 = 'azure_kinect1_5_calib_snap'
cam14 = 'azure_kinect1_4_calib_snap'
cam34 = 'azure_kinect3_4_calib_snap'
cam35 = 'azure_kinect3_5_calib_snap'
# TODO: Too long
if __name__ == "__main__":
    cache = diskcache.Cache('../calibration/cache')

    cameras = [
        cam24,
        cam15,
        # cam14,
        cam34,
        cam35
    ]

    for file in os.listdir(DIR_INPUT):
        experiment = file.split('.')[-2].split('_')[-2]

        file_path = os.path.join(DIR_INPUT, file)
        print(f"Visualizing {file_path}")
        
        with open(file_path, 'rb') as handle:
            output = pickle.load(handle)

        poses = output['points_3d'].reshape(-1, 2, 26, 3)
        params = output['params']
        for idx_cam, camera in enumerate(tqdm(cameras)):
            dir = data_loader.EXPERIMENTS[experiment][camera]

            poses_2d = poses_3d_2_2d(
                poses,
                params[idx_cam])

            write_video(poses_2d, experiment, camera, params[idx_cam], cache)
