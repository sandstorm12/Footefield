import sys
sys.path.append('../')

import copy
import cv2
import diskcache
import numpy as np
import open3d as o3d

from probreg import cpd
from utils import data_loader
from calibration import rgb_depth_map


def load_pcd(cam, cache):
    cache = diskcache.Cache('../calibration/cache')

    camera = f'azure_kinect{cam}_calib_snap'
    img_depth_path = f'/home/hamid/Documents/footefield/data/AzureKinectRecord_0729/a1/azure_kinect{cam}/depth/depth00000.png'
    img_color_path = f'/home/hamid/Documents/footefield/data/AzureKinectRecord_0729/a1/azure_kinect{cam}/color/color00000.jpg'

    img_depth = cv2.imread(img_depth_path, -1)
    img_color = cv2.imread(img_color_path)
    img_color = data_loader.downsample_keep_aspect_ratio(
        img_color,
        (
            data_loader.IMAGE_INFRARED_WIDTH,
            data_loader.IMAGE_INFRARED_HEIGHT
        )
    )

    img_color = rgb_depth_map.align_image_rgb(img_color, camera, cache)
    img_depth = rgb_depth_map.align_image_depth(img_depth, camera, cache)

    img_depth = np.clip(img_depth, 0, 3000)

    point_cloud = []
    colors = []
    for i in range(img_depth.shape[1]):
        for j in range(img_depth.shape[0]):
            if img_depth[j, i] != 0:
                point_cloud.append((i / 640., 1 - (j / 576.), 1 - (img_depth[j, i] / 5000.)))
                colors.append((img_color[j, i, 0] / 255.,
                                img_color[j, i, 1] / 255.,
                                img_color[j, i, 2] / 255.))

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    return pcd


if __name__ == "__main__":
    cache = diskcache.Cache('../calibration/cache')

    source = load_pcd(cam='2_4', cache=cache)
    target = load_pcd(cam='1_5', cache=cache)

    source.estimate_normals()
    target.estimate_normals()

    # load source and target point cloud
    source.remove_non_finite_points()
    
    # transform target point cloud
    th = np.deg2rad(30.0)
    target.transform(np.array([[np.cos(th), -np.sin(th), 0.0, 0.0],
                            [np.sin(th), np.cos(th), 0.0, 0.0],
                            [0.0, 0.0, 1.0, 0.0],
                            [0.0, 0.0, 0.0, 1.0]]))
    source = source.voxel_down_sample(voxel_size=0.005)
    target = target.voxel_down_sample(voxel_size=0.005)

    # compute cpd registration
    tf_param, _, _ = cpd.registration_cpd(source, target)
    result = copy.deepcopy(source)
    result.points = tf_param.transform(result.points)

    # draw result
    source.paint_uniform_color([1, 0, 0])
    target.paint_uniform_color([0, 1, 0])
    result.paint_uniform_color([0, 0, 1])
    o3d.visualization.draw_geometries([source, target, result])