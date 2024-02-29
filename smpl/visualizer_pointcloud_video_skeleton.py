import os
import sys
sys.path.append('../')

import cv2
import time
import pickle
import diskcache
import numpy as np
import open3d as o3d

from utils import data_loader
from calibration import rgb_depth_map


STORE_DIR = '../pose_estimation/keypoints_3d'
HALPE_LINES = np.array(
    [(0, 1), (0, 2), (1, 3), (2, 4), (5, 18), (6, 18), (5, 7),
     (7, 9), (6, 8), (8, 10), (17, 18), (18, 19), (19, 11),
     (19, 12), (11, 13), (12, 14), (13, 15), (14, 16), (20, 24),
     (21, 25), (23, 25), (22, 24), (15, 24), (16, 25)])


def get_cam(cam_name):
    return f'azure_kinect{cam_name}_calib_snap'


# TODO: Refactor
def get_depth_image(cam_name, experiment, idx):
    cam_num = cam_name[12:15]
    img_depth = '/home/hamid/Documents/phd/footefield/data/AzureKinectRecord_0729/{}/azure_kinect{}/depth/depth{:05d}.png'.format(experiment, cam_num, idx)

    return img_depth


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
        extrinsics[:3, 3] = T.reshape(3) / 1000
    elif cam == cam14:
        R2_com = np.dot(R2, R)
        T2_com = (np.dot(R2, T).reshape(3, 1) + T2).reshape(3,)
        extrinsics[:3, :3] = R2_com
        extrinsics[:3, 3] = T2_com / 1000
    elif cam == cam34:
        R2_com = np.dot(R2, R)
        T2_com = (np.dot(R2, T).reshape(3, 1) + T2).reshape(3,)
        R3_com = np.dot(R3, R2_com)
        T3_com = (np.dot(R3, T2_com).reshape(3, 1) + T3).reshape(3,)
        extrinsics[:3, :3] = R3_com
        extrinsics[:3, 3] = T3_com / 1000
    elif cam == cam35:
        R2_com = np.dot(R2, R)
        T2_com = (np.dot(R2, T).reshape(3, 1) + T2).reshape(3,)
        R3_com = np.dot(R3, R2_com)
        T3_com = (np.dot(R3, T2_com).reshape(3, 1) + T3).reshape(3,)
        R4_com = np.dot(R4, R3_com)
        T4_com = (np.dot(R4, T3_com).reshape(3, 1) + T4).reshape(3,)
        extrinsics[:3, :3] = R4_com
        extrinsics[:3, 3] = T4_com / 1000

    return extrinsics


def get_parameters(cam, cache):
    mtx, dist = get_intrinsics(cam, cache)

    extrinsics = get_extrinsics(cam, cache)

    return mtx, dist, extrinsics


def get_pcd(cam, experiment, idx, params):
    mtx = np.zeros((3, 3), dtype=float)
    mtx[0, 0] = params[12]
    mtx[1, 1] = params[13]
    mtx[0, 2] = params[14]
    mtx[1, 2] = params[15]
    dist = params[16:21]
    rotation = params[:9].reshape(3, 3)
    translation = params[9:12] / 1000
    extrinsics = np.identity(4, dtype=float)
    extrinsics[:3, :3] = rotation
    extrinsics[:3, 3] = translation.reshape(3)
    
    intrinsics = o3d.camera.PinholeCameraIntrinsic(
        640, 576, mtx[0, 0], mtx[1, 1], mtx[0, 2], mtx[1, 2])

    img_depth = get_depth_image(cam, experiment, idx)

    depth = o3d.io.read_image(img_depth)

    pcd = o3d.geometry.PointCloud.create_from_depth_image(
        depth, intrinsics, extrinsics)

    return pcd


def remove_outliers(pointcloud):
    _, ind = pointcloud.remove_statistical_outlier(
        nb_neighbors=20,
        std_ratio=.1)
    pointcloud = pointcloud.select_by_index(ind)
    
    return pointcloud


def preprocess(pointcloud):
    pointcloud = remove_outliers(pointcloud)

    pointcloud = pointcloud.voxel_down_sample(voxel_size=0.005)

    return pointcloud


def visualize_poses(poses, experiment, params):
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window()
    vis.get_render_option().background_color = data_loader.COLOR_SPACE_GRAY
    vis.get_render_option().show_coordinate_frame = True
    
    geometry_combined = o3d.geometry.PointCloud()
    geometry = o3d.geometry.PointCloud()
    lines = o3d.geometry.LineSet()
    for idx in range(len(poses)):
        pcd24 = get_pcd(cam24, experiment, idx, params)
        # pcd15 = get_pcd(cam15, i, cache)
        # pcd14 = get_pcd(cam14, i, cache)
        # pcd34 = get_pcd(cam34, i, cache)
        # pcd35 = get_pcd(cam35, i, cache)

        # pcd = pcd24 + pcd15 + pcd14 + pcd34 + pcd35
        pcd_combined = pcd24
        pcd_combined = preprocess(pcd_combined)

        keypoints = poses[idx].reshape(-1, 3)
        pcd = o3d.geometry.PointCloud()
        keypoints /= 1000
        pcd.points = o3d.utility.Vector3dVector(keypoints)
        pcd.paint_uniform_color([0, 1, 0]) # Blue points

        connections = np.concatenate((HALPE_LINES, HALPE_LINES + 26))
        
        lines.points = o3d.utility.Vector3dVector(keypoints)
        lines.lines = o3d.utility.Vector2iVector(connections)
        lines.paint_uniform_color([1, 1, 1]) # White lines

        geometry.points = pcd.points
        geometry.colors = pcd.colors
        geometry_combined.points = pcd_combined.points
        if idx == 0:
            vis.add_geometry(geometry_combined)
            vis.add_geometry(geometry)
            vis.add_geometry(lines)
        else:
            vis.update_geometry(geometry_combined)
            vis.update_geometry(geometry)
            vis.update_geometry(lines)
            
        delay_ms = 100
        for _ in range(delay_ms // 10):
            vis.poll_events()
            vis.update_renderer()
            time.sleep(.01)

        print(f"Update {idx}: {time.time()}")


# TODO: Move the cameras somewhere else
cam24 = 'azure_kinect2_4_calib_snap'
cam15 = 'azure_kinect1_5_calib_snap'
cam14 = 'azure_kinect1_4_calib_snap'
cam34 = 'azure_kinect3_4_calib_snap'
cam35 = 'azure_kinect3_5_calib_snap'
if __name__ == "__main__":
    cache = diskcache.Cache('../calibration/cache')

    # for file in sorted(os.listdir(STORE_DIR), reverse=False):
    #     experiment = file.split('.')[0].split('_')[1]
    #     file_path = os.path.join(STORE_DIR, file)
    #     print(f"Visualizing {file_path}")
    experiment = 'a1'

    with open('/home/hamid/Documents/phd/footefield/footefield/pose_estimation/output.pkl', 'rb') as handle:
        output = pickle.load(handle)

    poses = output['points_3d'].reshape(-1, 52, 3)
    params = output['params']

    visualize_poses(poses, experiment, params)
