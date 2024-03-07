import sys
sys.path.append('../')

import os
import cv2
import pickle
import diskcache
import numpy as np

from tqdm import tqdm
from utils import data_loader
from calibration import rgb_depth_map


DIR_INPUT = "./keypoints_3d_ba"
DIR_OUTPUT = "./output_videos"
PARAM_OUTPUT_SIZE = (640, 576)
PARAM_OUTPUT_FPS = 5.0
PARAM_CALIB_SIZE = 16


def get_intrinsics(cam, cache):
    if cam == cam24:
        mtx = cache['extrinsics'][cam24 + 'infrared']['mtx_l']
        dist = cache['extrinsics'][cam24 + 'infrared']['dist_l']
    elif cam == cam15:
        mtx = cache['extrinsics'][cam24 + 'infrared']['mtx_r']
        dist = cache['extrinsics'][cam24 + 'infrared']['dist_r']
    elif cam == cam14:
        mtx = cache['extrinsics'][cam15 + 'infrared']['mtx_r']
        dist = cache['extrinsics'][cam15 + 'infrared']['dist_r']
    elif cam == cam34:
        mtx = cache['extrinsics'][cam14 + 'infrared']['mtx_r']
        dist = cache['extrinsics'][cam14 + 'infrared']['dist_r']
    elif cam == cam35:
        mtx = cache['extrinsics'][cam34 + 'infrared']['mtx_r']
        dist = cache['extrinsics'][cam34 + 'infrared']['dist_r']

    return mtx, dist


def get_extrinsics(cam, cache):
    R = cache['extrinsics'][cam24 + 'infrared']['rotation']
    T = cache['extrinsics'][cam24 + 'infrared']['transition']
    R2 = cache['extrinsics'][cam15 + 'infrared']['rotation']
    T2 = cache['extrinsics'][cam15 + 'infrared']['transition']
    R3 = cache['extrinsics'][cam14 + 'infrared']['rotation']
    T3 = cache['extrinsics'][cam14 + 'infrared']['transition']
    R4 = cache['extrinsics'][cam34 + 'infrared']['rotation']
    T4 = cache['extrinsics'][cam34 + 'infrared']['transition']
    
    extrinsics = np.zeros((3, 4), dtype=float)
    if cam == cam24:
        r = np.array([[1, 0, 0],
                     [0, 1, 0],
                     [0, 0, 1]])
        t = np.array([0, 0, 0])
        extrinsics[:3, :3] = r
        extrinsics[:3, 3] = t.reshape(3)
    elif cam == cam15:
        extrinsics[:3, :3] = R
        extrinsics[:3, 3] = T.reshape(3)
    elif cam == cam14:
        R2_com = np.dot(R2, R)
        T2_com = (np.dot(R2, T).reshape(3, 1) + T2).reshape(3,)
        extrinsics[:3, :3] = R2_com
        extrinsics[:3, 3] = T2_com
    elif cam == cam34:
        R2_com = np.dot(R2, R)
        T2_com = (np.dot(R2, T).reshape(3, 1) + T2).reshape(3,)
        R3_com = np.dot(R3, R2_com)
        T3_com = (np.dot(R3, T2_com).reshape(3, 1) + T3).reshape(3,)
        extrinsics[:3, :3] = R3_com
        extrinsics[:3, 3] = T3_com
    elif cam == cam35:
        R2_com = np.dot(R2, R)
        T2_com = (np.dot(R2, T).reshape(3, 1) + T2).reshape(3,)
        R3_com = np.dot(R3, R2_com)
        T3_com = (np.dot(R3, T2_com).reshape(3, 1) + T3).reshape(3,)
        R4_com = np.dot(R4, R3_com)
        T4_com = (np.dot(R4, T3_com).reshape(3, 1) + T4).reshape(3,)
        extrinsics[:3, :3] = R4_com
        extrinsics[:3, 3] = T4_com

    return extrinsics


def get_parameters(cam, cache):
    mtx, dist = get_intrinsics(cam, cache)

    extrinsics = get_extrinsics(cam, cache)

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


def write_video(poses_2d, experiment, camera, cache):
    img_rgb_paths = data_loader.list_rgb_images(os.path.join(dir, "color"))

    writer = get_video_writer(experiment, camera)
    for idx, t in enumerate(poses_2d.reshape(poses_2d.shape[0], -1, 2)):
        image = cv2.imread(img_rgb_paths[idx])
        image = data_loader.downsample_keep_aspect_ratio(
            image,
            (data_loader.IMAGE_INFRARED_WIDTH,
             data_loader.IMAGE_INFRARED_HEIGHT))

        image = rgb_depth_map.align_image_rgb(image, camera, cache)
        for point in t:
            cv2.circle(image, (int(point[0]), int(point[1])),
                       3, (0, 255, 0), -1)

        connections = np.concatenate(
            (np.array(data_loader.HALPE_EDGES),
             np.array(data_loader.HALPE_EDGES) + 26))
        for connection in connections:
            cv2.line(image,
                    (int(t[connection[0]][0]), int(t[connection[0]][1])),
                    (int(t[connection[1]][0]), int(t[connection[1]][1])),
                    (255, 255, 255), 1)

        writer.write(image)


def poses_3d_2_2d(poses_3d, params):
    poses_shape = list(poses_3d.shape)
    poses_shape[-1] = 2
    
    mtx = np.zeros((3, 3), dtype=float)
    mtx[0, 0] = params[12]
    mtx[1, 1] = params[13]
    mtx[0, 2] = params[14]
    mtx[1, 2] = params[15]
    dist = params[16:]
    rotation = params[:9].reshape(3, 3)
    translation = params[9:12]
    poses_2d = project_3d_to_2d(
        mtx, dist,
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
                params.reshape(-1, PARAM_CALIB_SIZE)[idx_cam])

            write_video(poses_2d, experiment, camera, cache)
