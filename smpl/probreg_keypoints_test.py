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


def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])


if __name__ == "__main__":
    cache = diskcache.Cache('../calibration/cache')

    poses_1 = cache['process']['a1_azure_kinect3_4_calib_snap_azure_kinect3_5_calib_snap_skeleton_3D']
    poses_2 = cache['process']['a1_azure_kinect2_4_calib_snap_azure_kinect1_5_calib_snap_skeleton_3D']

    poses_1 = np.array(poses_1[0])
    poses_2 = np.array(poses_2[0])

    print(poses_1.shape, poses_2.shape)

    pcd_1 = o3d.geometry.PointCloud()
    pcd_1.points = o3d.utility.Vector3dVector(poses_1.reshape(-1, 3))

    pcd_2 = o3d.geometry.PointCloud()
    pcd_2.points = o3d.utility.Vector3dVector(poses_2.reshape(-1, 3))

    source = pcd_1
    target = pcd_2

    source.paint_uniform_color([1, 0.706, 0])
    target.paint_uniform_color([0, 0.651, 0.929])
    o3d.visualization.draw_geometries([source, target])

    source.estimate_normals()
    target.estimate_normals()

    # load source and target point cloud
    source.remove_non_finite_points()
    
    # transform target point cloud
    th = np.deg2rad(180.0)
    target.transform(np.array([
        [np.cos(th), -np.sin(th), 0.0, 0.0], # 1
        [np.sin(th), np.cos(th), 0.0, 0.0], # 2
        [0.0, 0.0, 1.0, 0.0], # 3
        [0.0, 0.0, 0.0, 1.0]]))
    # source = source.voxel_down_sample(voxel_size=0.005)
    # target = target.voxel_down_sample(voxel_size=0.005)

    o3d.visualization.draw_geometries([source, target])

    # compute cpd registration
    tf_param, _, _ = cpd.registration_cpd(source, target, tf_type_name='rigid')
    result = copy.deepcopy(source)
    result.points = tf_param.transform(result.points)

    print(tf_param)

    # draw result
    source.paint_uniform_color([1, 0, 0])
    target.paint_uniform_color([0, 1, 0])
    result.paint_uniform_color([0, 0, 1])
    o3d.visualization.draw_geometries([target, result])


    # threshold = 0.5
    # trans_init = np.identity (4)

    # draw_registration_result(source, target, trans_init)
    # print("Initial alignment")
    # evaluation = o3d.pipelines.registration.evaluate_registration(source, target,
    #                                                     threshold, trans_init)
    # print(evaluation)

    # print("Apply point-to-point ICP")
    # reg_p2p = o3d.pipelines.registration.registration_icp(
    #     source, target, threshold, trans_init,
    #     o3d.pipelines.registration.TransformationEstimationPointToPoint())
    # print(reg_p2p)
    # print("Transformation is:")
    # print(reg_p2p.transformation)
    # print("")
    # draw_registration_result(source, target, reg_p2p.transformation)

    # print("Apply point-to-plane ICP")
    # reg_p2l = o3d.pipelines.registration.registration_icp(
    #     source, target, threshold, trans_init,
    #     o3d.pipelines.registration.TransformationEstimationPointToPlane())
    # print(reg_p2l)
    # print("Transformation is:")
    # print(reg_p2l.transformation)
    # print("")
    # draw_registration_result(source, target, reg_p2l.transformation)