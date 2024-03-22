import sys
sys.path.append('../')

import os
import cv2
import time
import pickle
import diskcache
import numpy as np
import open3d as o3d


DIR_PARAMS = '../pose_estimation/keypoints_3d_ba'
COLOR_SPACE_GRAY = [0.203921569, 0.239215686, 0.274509804]


def get_images(cam, idx):
    cam_name = cam[12:15]
    img_depth = '/home/hamid/Documents/phd/footefield/data/AzureKinectRecord_0729/a1/azure_kinect{}/depth/depth{:05d}.png'.format(cam_name, idx)
    img_color = '/home/hamid/Documents/phd/footefield/data/AzureKinectRecord_0729/a1/azure_kinect{}/color/color{:05d}.jpg'.format(cam_name, idx)

    return img_color, img_depth


def get_pcd(cam, idx, params, cache):
    _, img_depth = get_images(cam, idx)

    _, _, extrinsics_rgb = get_params(cam, params)
    mtx, dist, extrinsics = get_params_depth(cam, cache)
    extrinsics_finetuned = get_finetuned_extrinsics(cam)
    extrinsics = np.matmul(extrinsics, extrinsics_rgb)
    # extrinsics = np.matmul(extrinsics_finetuned, extrinsics)
    
    img_depth = cv2.imread(img_depth, -1)

    mapx, mapy = cv2.initUndistortRectifyMap(mtx, dist, None, mtx, (640, 576), cv2.CV_32FC2)
    img_depth = cv2.remap(img_depth, mapx, mapy, cv2.INTER_NEAREST)

    depth = o3d.geometry.Image(img_depth)

    intrinsics = o3d.camera.PinholeCameraIntrinsic(640, 576, mtx[0, 0], mtx[1, 1], mtx[0, 2], mtx[1, 2])

    pcd = o3d.geometry.PointCloud.create_from_depth_image(depth, intrinsics, extrinsics)

    pcd = pcd.transform(extrinsics_finetuned)

    return pcd


def get_finetuned_extrinsics(cam):
    if cam == cam24:
        extrinsics = np.array(
            [
                [1.00000000e00, 3.52090809e-26, 8.27180613e-25, -1.65436123e-24],
                [3.52084522e-26, 1.00000000e00, 0.00000000e00, 0.00000000e00],
                [-8.27180613e-25, 0.00000000e00, 1.00000000e00, 0.00000000e00],
                [0.00000000e00, 0.00000000e00, 0.00000000e00, 1.00000000e00],
            ]
        )
    elif cam == cam15:
        extrinsics = np.array(
            [
                [0.99964627, -0.01086932, 0.02427319, -0.04616381],
                [0.00990623, 0.9991724, 0.03945104, -0.08110659],
                [-0.02468191, -0.03919663, 0.99892664, -0.03160174],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
    elif cam == cam14:
        raise Exception("Unknown camera.")
    elif cam == cam34:
        extrinsics = np.array(
            [
                [0.99749669, -0.01552832, 0.06898719, -0.1396021],
                [0.01539566, 0.99987847, 0.0024543, -0.01443389],
                [-0.06901691, -0.00138605, 0.99761453, -0.01269228],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
    elif cam == cam35:
        extrinsics = np.array(
            [
                [9.93827554e-01, -4.39298306e-03, 1.10848971e-01, -2.92642058e-01],
                [4.51243087e-03, 9.99989477e-01, -8.26722131e-04, -1.38620371e-02],
                [-1.10844173e-01, 1.32181755e-03, 9.93836919e-01, 5.66665409e-03],
                [0.00000000e00, 0.00000000e00, 0.00000000e00, 1.00000000e00],
            ]
        )
    else:
        raise Exception("Unknown camera.")
    
    return extrinsics


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


def get_params_depth(cam, cache):
    mtx = cache['depth_matching'][cam]['mtx_r']
    dist = cache['depth_matching'][cam]['dist_r']
    R = cache['depth_matching'][cam]['rotation']
    T = cache['depth_matching'][cam]['transition']

    extrinsics = np.identity(4, dtype=float)
    extrinsics[:3, :3] = R
    extrinsics[:3, 3] = T.ravel() / 1000

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


def visualize(cameras, params, cache):
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.get_render_option().background_color = COLOR_SPACE_GRAY
    vis.get_render_option().show_coordinate_frame = True

    geometry = o3d.geometry.PointCloud()

    for i in range(1000):
        pcd = o3d.geometry.PointCloud()
        for cam in cameras:
            pcd += preprocess(get_pcd(cam, i, params, cache))

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
    cache = diskcache.Cache('../calibration/cache')

    cameras = [
        cam24,
        cam15,
        cam34,
        cam35,
    ]

    for file in os.listdir(DIR_PARAMS):
        experiment = file.split('.')[-2].split('_')[-2]
        # if experiment == 'a1':
        #     continue

        print(experiment)

        file_path = os.path.join(DIR_PARAMS, file)
        print(f"Visualizing {file_path}")
        
        with open(file_path, 'rb') as handle:
            output = pickle.load(handle)

        poses = output['points_3d'].reshape(-1, 2, 26, 3)
        params = output['params']

        visualize(cameras, params, cache)
    
    
