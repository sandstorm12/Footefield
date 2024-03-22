import sys
sys.path.append('../')

import os
import cv2
import time
import glob
import pickle
import diskcache
import numpy as np
import open3d as o3d


DIR_PARAMS = '../pose_estimation/keypoints_3d_ba'
DIR_PARMAS_FINETUNED = "./extrinsics_finetuned"
COLOR_SPACE_GRAY = [0.203921569, 0.239215686, 0.274509804]


def get_images(cam, idx):
    cam_name = cam[12:15]
    img_depth = '/home/hamid/Documents/phd/footefield/data/AzureKinectRecord_0729/a1/azure_kinect{}/depth/depth{:05d}.png'.format(cam_name, idx)
    img_color = '/home/hamid/Documents/phd/footefield/data/AzureKinectRecord_0729/a1/azure_kinect{}/color/color{:05d}.jpg'.format(cam_name, idx)

    return img_color, img_depth


def get_pcd(cam, idx, extrinsics, cache):
    _, img_depth = get_images(cam, idx)
    mtx, dist, _ = get_params_depth(cam, cache)

    img_depth = cv2.imread(img_depth, -1)
    mapx, mapy = cv2.initUndistortRectifyMap(
        mtx, dist, None, mtx, (640, 576), cv2.CV_32FC2)
    img_depth = cv2.remap(img_depth, mapx, mapy, cv2.INTER_NEAREST)
    depth = o3d.geometry.Image(img_depth)

    intrinsics = o3d.camera.PinholeCameraIntrinsic(
        640, 576, mtx[0, 0], mtx[1, 1], mtx[0, 2], mtx[1, 2])
    pcd = o3d.geometry.PointCloud.create_from_depth_image(
        depth, intrinsics, extrinsics['base'])
    pcd = pcd.transform(extrinsics['offset'])

    return pcd


def load_finetuned_extrinsics():
    extrinsics_finetuned = {}
    for path in glob.glob(os.path.join(DIR_PARMAS_FINETUNED, '*')):
        experiment = path.split('.')[-2].split('_')[-1]
        with open(path, 'rb') as handle:
            params = pickle.load(handle)

        extrinsics_finetuned[experiment] = params

    return extrinsics_finetuned


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
        nb_neighbors=5,
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
        for idx_cam, cam in enumerate(cameras):
            pcd += preprocess(get_pcd(cam, i, params[idx_cam], cache))

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
        # cam14,
        cam34,
        cam35,
    ]

    finetuned_extrinsics = load_finetuned_extrinsics()

    for file in os.listdir(DIR_PARAMS):
        print(file)
        experiment = file.split('.')[-2].split('_')[-2]

        file_path = os.path.join(DIR_PARAMS, file)
        print(f"Visualizing {file_path}")

        visualize(cameras, finetuned_extrinsics[experiment], cache)
