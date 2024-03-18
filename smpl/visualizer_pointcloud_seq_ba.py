import sys
sys.path.append('../')

import os
import cv2
import time
import pickle
import numpy as np
import open3d as o3d


DIR_PARAMS = '../pose_estimation/keypoints_3d_ba'


def get_images(cam, idx):
    cam_name = cam[12:15]
    img_depth = '/home/hamid/Documents/phd/footefield/data/AzureKinectRecord_0729/a1/azure_kinect{}/depth/depth{:05d}.png'.format(cam_name, idx)
    img_color = '/home/hamid/Documents/phd/footefield/data/AzureKinectRecord_0729/a1/azure_kinect{}/color/color{:05d}.jpg'.format(cam_name, idx)

    return img_color, img_depth


def get_pcd(cam, idx, params):
    _, img_depth = get_images(cam, idx)

    mtx, dist, extrinsics = get_params(cam, params)
    img_depth = cv2.imread(img_depth, -1)

    mapx, mapy = cv2.initUndistortRectifyMap(mtx, dist, None, None, (640, 576), cv2.CV_32FC2)
    img_depth = cv2.remap(img_depth, mapx, mapy, cv2.INTER_NEAREST)

    depth = o3d.geometry.Image(img_depth)

    intrinsics = o3d.camera.PinholeCameraIntrinsic(640, 576, mtx[0, 0], mtx[1, 1], mtx[0, 2], mtx[1, 2])

    pcd = o3d.geometry.PointCloud.create_from_depth_image(depth, intrinsics, extrinsics)

    return pcd

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

    extrinsics = np.eye(4, dtype=float)
    extrinsics[:3, :3] = rotation
    extrinsics[:3, 3] = translation / 1000

    return mtx, dist, extrinsics


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


def visualize(cameras, params):
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    geometry = o3d.geometry.PointCloud()

    for i in range(1000):
        pcd = o3d.geometry.PointCloud()
        for cam in cameras:
            pcd += preprocess(get_pcd(cam, i, params))

        geometry.points = pcd.points
        if i == 0:
            vis.add_geometry(geometry)
        else:
            vis.update_geometry(geometry)

        delay_ms = 100
        for _ in range(delay_ms // 10):
            vis.poll_events()
            vis.update_renderer()
            time.sleep(.01)

        print(f"Update {i}: {time.time()}")


# TODO: Move the cameras somewhere else
cam24 = 'azure_kinect2_4_calib_snap'
cam15 = 'azure_kinect1_5_calib_snap'
cam14 = 'azure_kinect1_4_calib_snap'
cam34 = 'azure_kinect3_4_calib_snap'
cam35 = 'azure_kinect3_5_calib_snap'
if __name__ == "__main__":
    cameras = [
        cam24,
        cam15,
        cam34,
        cam35,
    ]

    for file in os.listdir(DIR_PARAMS):
        experiment = file.split('.')[-2].split('_')[-2]

        file_path = os.path.join(DIR_PARAMS, file)
        print(f"Visualizing {file_path}")
        
        with open(file_path, 'rb') as handle:
            output = pickle.load(handle)

        poses = output['points_3d'].reshape(-1, 2, 26, 3)
        params = output['params']

        visualize(cameras, params)
    
    
