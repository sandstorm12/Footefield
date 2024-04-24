from ast import Tuple
import os
import sys
sys.path.append('../')

import cv2
import time
import glob
import pickle
import diskcache
import numpy as np
import open3d as o3d

from tqdm import tqdm
from utils import data_loader
from sklearn.cluster import KMeans


VIS_MESH = True

STORE_DIR = '../pose_estimation/keypoints_3d_ba'
LENGTH = 100
DIR_PARMAS_GLOBAL = "./extrinsics_global"
DIR_STORE = '/home/hamid/Documents/phd/footefield/Pose_to_SMPL/fit/output/HALPE/'
DIR_PARAMS = '../pose_estimation/keypoints_3d_pose2smpl/'
DIR_OUTPUT = "./output_videos_pc"

PARAM_OUTPUT_SIZE = (1920, 1080)
PARAM_OUTPUT_FPS = 5.0
PARAM_CALIB_SIZE = 16


def get_cam(cam_name):
    return f'azure_kinect{cam_name}_calib_snap'


# TODO: Refactor
def get_depth_image(cam_name, experiment, idx):
    cam_num = cam_name[12:15]
    img_depth = '/home/hamid/Documents/phd/footefield/data/AzureKinectRecord_0729/{}/azure_kinect{}/depth/depth{:05d}.png'.format(experiment, cam_num, idx)

    return img_depth


def get_params(cam, params):
    if cam == cam24:
        idx_cam = 0
    elif cam == cam15:
        idx_cam = 1
    elif cam == cam14:
        raise Exception("Unknown camera.")
    elif cam == cam34:
        idx_cam = 2
    elif cam == cam35:
        idx_cam = 3
    else:
        raise Exception("Unknown camera.")

    mtx = params[idx_cam]['mtx']
    dist = params[idx_cam]['dist']
    rotation = params[idx_cam]['rotation']
    translation = params[idx_cam]['translation']

    extrinsics = np.identity(4, dtype=float)
    extrinsics[:3, :3] = rotation
    extrinsics[:3, 3] = translation / 1000

    return mtx, dist, extrinsics


def get_extrinsics(cam, cache):
    R = cache['extrinsics'][cam24 + 'infrared']['rotation']
    T = cache['extrinsics'][cam24 + 'infrared']['transition']
    R2 = cache['extrinsics'][cam15 + 'infrared']['rotation']
    T2 = cache['extrinsics'][cam15 + 'infrared']['transition']
    R3 = cache['extrinsics'][cam14 + 'infrared']['rotation']
    T3 = cache['extrinsics'][cam14 + 'infrared']['transition']
    R4 = cache['extrinsics'][cam34 + 'infrared']['rotation']
    T4 = cache['extrinsics'][cam34 + 'infrared']['transition']
    
    extrinsics = np.identity(4, dtype=float)
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


def get_params_depth(cam, cache):
    mtx = cache['depth_matching'][cam]['mtx_r']
    dist = cache['depth_matching'][cam]['dist_r']
    R = cache['depth_matching'][cam]['rotation']
    T = cache['depth_matching'][cam]['transition']

    extrinsics = np.identity(4, dtype=float)
    extrinsics[:3, :3] = R
    extrinsics[:3, 3] = T.ravel() / 1000

    return mtx, dist, extrinsics


def get_params_rgb(cam, cache):
    intrinsics = cache.get("intrinsics", None)
    mtx = intrinsics[cam]['mtx']
    dist = intrinsics[cam]['dist']

    return mtx, dist


def get_pcd(subject, cam, experiment, idx, extrinsics, cache):
    img_depth = get_depth_image(cam, experiment, idx)
    mtx, dist, _ = get_params_depth(cam, cache)

    img_depth = cv2.imread(img_depth, -1)
    mapx, mapy = cv2.initUndistortRectifyMap(
        mtx, dist, None, mtx, (640, 576), cv2.CV_32FC2)
    img_depth = cv2.remap(img_depth, mapx, mapy, cv2.INTER_NEAREST)

    depth = o3d.geometry.Image(img_depth)

    intrinsics = o3d.camera.PinholeCameraIntrinsic(
        640, 576, mtx[0, 0], mtx[1, 1], mtx[0, 2], mtx[1, 2])
    pcd = o3d.geometry.PointCloud.create_from_depth_image(
        depth, intrinsics, extrinsics['extrinsics'][cam]['base'])
    pcd = pcd.transform(extrinsics['extrinsics'][cam]['offset'])

    pcd_np = np.asarray(pcd.points)

    start_pts = np.array([[0, 0, 0], [1, -1, 3]])
    pcd_np = np.asarray(pcd.points)
    kmeans = KMeans(n_clusters=2, random_state=47,
                    init=start_pts, n_init=1).fit(pcd_np)
    # TODO: Explain what (subject + 1 % 2) is
    pcd.points = o3d.utility.Vector3dVector(
        pcd_np[kmeans.labels_ == (subject + 1) % 2])

    return pcd


def remove_outliers(pointcloud):
    _, ind = pointcloud.remove_statistical_outlier(
        nb_neighbors=16,
        std_ratio=.05)
    pointcloud = pointcloud.select_by_index(ind)
    
    return pointcloud


def preprocess(pointcloud):
    pointcloud = remove_outliers(pointcloud)

    # pointcloud = pointcloud.voxel_down_sample(voxel_size=0.005)

    return pointcloud


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


# Implemented by Gemini
def project_3d_to_2d(camera_matrix, dist_coeffs, rvec, tvec, object_points):
    image_points, _ = cv2.projectPoints(object_points, rvec, tvec,
                                        camera_matrix, dist_coeffs)

    image_points = image_points.squeeze()

    return image_points


# TODO: Shorten
def write_video(img_paths, experiment,
                extrinsics, camera, params, cache):
    writer = get_video_writer(experiment, camera)
    for idx in range(min(len(img_paths), LENGTH)):
        img_rgb = cv2.imread(img_rgb_paths[idx])
        mtx, dist = get_params_rgb(camera, cache)
        img_rgb = cv2.undistort(img_rgb, mtx, dist, None, None)

        for subject in [0, 1]:
            pcd = get_pcd(subject, cam24, experiment, idx,
                          extrinsics[experiment + '_' + str(subject)], cache)
            pcd += get_pcd(subject, cam15, experiment, idx, 
                           extrinsics[experiment + '_' + str(subject)], cache)
            # pcd14 = get_pcd(subject, cam14, experiment, idx, 
            #                 extrinsics[experiment + '_' + str(subject)], cache) 
            pcd += get_pcd(subject, cam34, experiment, idx, 
                           extrinsics[experiment + '_' + str(subject)], cache)
            pcd += get_pcd(subject, cam35, experiment, idx, 
                           extrinsics[experiment + '_' + str(subject)], cache)

            pcd_combined = pcd
            pcd = pcd_combined.transform(
                extrinsics[experiment + '_' + str(subject)]['global'])

            pcd_np = np.asarray(pcd.points) * 1000

            mtx = params['mtx']
            dist = params['dist']
            rotation = params['rotation']
            translation = params['translation']

            pcd_2d = project_3d_to_2d(mtx, dist, rotation, translation, pcd_np)
            for point in pcd_2d:
                x = int(point[0])
                y = int(point[1])
                if 0 < x < img_rgb.shape[1] and 0 < y < img_rgb.shape[0]:
                    color = tuple(((img_rgb[y, x] + [0, 255, 0]) // 2).tolist())
                    cv2.circle(img_rgb, (int(point[0]), int(point[1])),
                            1, color, -1)

        writer.write(img_rgb)


def load_global_extrinsics():
    extrinsics_global = {}
    for path in glob.glob(os.path.join(DIR_PARMAS_GLOBAL, '*')):
        experiment = path.split('.')[-2].split('_')[-2]
        subject = path.split('.')[-2].split('_')[-1]
        with open(path, 'rb') as handle:
            params = pickle.load(handle)

        extrinsics_global[experiment + '_' + subject] = params

    return extrinsics_global


# TODO: Move the cameras somewhere else
cam24 = 'azure_kinect2_4_calib_snap'
cam15 = 'azure_kinect1_5_calib_snap'
cam14 = 'azure_kinect1_4_calib_snap'
cam34 = 'azure_kinect3_4_calib_snap'
cam35 = 'azure_kinect3_5_calib_snap'
if __name__ == "__main__":
    cache = diskcache.Cache('../calibration/cache')

    cameras = [
        cam24,
        cam15,
        # cam14,
        cam34,
        cam35,   
    ]

    extrinsics_global = load_global_extrinsics()

    for file in sorted(os.listdir(STORE_DIR), reverse=False):
        for idx_cam, camera in enumerate(tqdm(cameras)):
            experiment = file.split('.')[0].split('_')[1]
            dir = data_loader.EXPERIMENTS[experiment][camera]

            file_path = os.path.join(STORE_DIR, file)
            print(f"Visualizing {file_path}")
            
            with open(file_path, 'rb') as handle:
                output = pickle.load(handle)

            params = output['params']
            
            img_rgb_paths = data_loader.list_rgb_images(os.path.join(dir, "color"))

            write_video(
                img_rgb_paths, 
                experiment, extrinsics_global, camera,
                params[idx_cam], cache)
