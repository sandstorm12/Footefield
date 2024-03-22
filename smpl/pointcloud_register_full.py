import sys
sys.path.append('../')

import os
import cv2
import copy
import pickle
import diskcache
import numpy as np
import open3d as o3d

from utils import data_loader


COLOR_SPACE_GRAY = [0.203921569, 0.239215686, 0.274509804]
DIR_INPUT = '../pose_estimation/keypoints_3d_ba'
DIR_OUTPUT = './extrinsics_finetuned'


def load_pointcloud(path, cam, idx, params, cache):
    path_depth = os.path.join(path, 'depth/depth{:05d}.png'.format(idx))

    _, _, extrinsics_rgb = get_params(cam, params)
    mtx, dist, extrinsics = get_params_depth(cam, cache)
    extrinsics = np.matmul(extrinsics, extrinsics_rgb)
    intrinsics = o3d.camera.PinholeCameraIntrinsic(
        data_loader.IMAGE_INFRARED_WIDTH,
        data_loader.IMAGE_INFRARED_HEIGHT,
        mtx[0, 0], mtx[1, 1], mtx[0, 2], mtx[1, 2])

    img_depth = cv2.imread(path_depth, -1)
    mapx, mapy = cv2.initUndistortRectifyMap(
        mtx, dist, None, mtx, (640, 576), cv2.CV_32FC2)
    img_depth = cv2.remap(img_depth, mapx, mapy, cv2.INTER_NEAREST)
    depth = o3d.geometry.Image(img_depth)

    pc = o3d.geometry.PointCloud.create_from_depth_image(
        depth, intrinsics, extrinsics)

    return pc


def count_depth_images(experiment):
    path_first_cam = list(data_loader.EXPERIMENTS[experiment].values())[0]
    path = os.path.join(path_first_cam, 'depth')
    file_count = len(os.listdir(path))

    return file_count


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
    _, ind = pointcloud.remove_radius_outlier(
        nb_points=16, radius=0.05)
    pointcloud = pointcloud.select_by_index(ind)

    return pointcloud


def preprocess(pointcloud, voxel_size):
    pointcloud = remove_outliers(pointcloud)

    pointcloud = pointcloud.voxel_down_sample(voxel_size=voxel_size)

    return pointcloud


def show(pointcloud):
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=True)
    vis.get_render_option().background_color = [
        0.203921569, 0.239215686, 0.274509804]
    vis.get_render_option().show_coordinate_frame = True
    vis.add_geometry(pointcloud)
    vis.run()


def get_subject(cams, experiment, idx, voxel_size, params, cache):
    pcds = []
    for cam in cams:
        path = data_loader.EXPERIMENTS[experiment][cam]
        pcd = load_pointcloud(path, cam, idx, params, cache)
        pcd = preprocess(pcd, voxel_size)

        pcds.append(pcd)

    return pcds


def pairwise_registration(source, target, max_correspondence_distance_coarse,
                          max_correspondence_distance_fine):
    icp_coarse = o3d.pipelines.registration.registration_icp(
        source, target, max_correspondence_distance_coarse, np.identity(4),
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    icp_fine = o3d.pipelines.registration.registration_icp(
        source, target, max_correspondence_distance_fine,
        icp_coarse.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    
    transformation_icp = icp_fine.transformation
    information_icp = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
        source, target, max_correspondence_distance_fine,
        icp_fine.transformation)
    
    return transformation_icp, information_icp


def full_registration(pcds, max_correspondence_distance_coarse,
                      max_correspondence_distance_fine):
    pose_graph = o3d.pipelines.registration.PoseGraph()
    odometry = np.identity(4)
    pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(
        odometry))
    n_pcds = len(pcds)
    for source_id in range(n_pcds):
        for target_id in range(source_id + 1, n_pcds):
            transformation_icp, information_icp = pairwise_registration(
                pcds[source_id], pcds[target_id],
                max_correspondence_distance_coarse,
                max_correspondence_distance_fine)
            
            if target_id == source_id + 1:  # odometry case
                odometry = np.dot(transformation_icp, odometry)
                pose_graph.nodes.append(
                    o3d.pipelines.registration.PoseGraphNode(
                        np.linalg.inv(odometry)))
                pose_graph.edges.append(
                    o3d.pipelines.registration.PoseGraphEdge(
                        source_id,
                        target_id,
                        transformation_icp,
                        information_icp,
                        uncertain=False))
            else:  # loop closure case
                pose_graph.edges.append(
                    o3d.pipelines.registration.PoseGraphEdge(
                        source_id,
                        target_id,
                        transformation_icp,
                        information_icp,
                        uncertain=True))
    return pose_graph


def register(pcds, voxel_size):
    for pcd in pcds:
        pcd.estimate_normals()

    max_correspondence_distance_coarse = voxel_size * 15
    max_correspondence_distance_fine = voxel_size * 1.5
    with o3d.utility.VerbosityContextManager(
            o3d.utility.VerbosityLevel.Info) as cm:
        pose_graph = full_registration(
            pcds,
            max_correspondence_distance_coarse,
            max_correspondence_distance_fine)

    option = o3d.pipelines.registration.GlobalOptimizationOption(
        max_correspondence_distance=max_correspondence_distance_fine,
        edge_prune_threshold=0.25,
        reference_node=0)
    with o3d.utility.VerbosityContextManager(
            o3d.utility.VerbosityLevel.Info) as cm:
        o3d.pipelines.registration.global_optimization(
            pose_graph,
            o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
            o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
            option)
        
    return pose_graph


def finetune_extrinsics(cams, experiment, interval, voxel_size,
                        params, cache):
    file_count = count_depth_images(experiment)

    pose_avg = [None] * 5
    for i in range(0, file_count, interval):
        pcds = get_subject(cams, experiment, i, voxel_size, params, cache)

        pose_graph = register(pcds, voxel_size)

        def key_callback(vis):
            store_extrinsics()
            print('Saved extrinsics...')
            sys.exit()

        def store_extrinsics():
            extrinsics = []
            for pose in pose_graph.nodes:
                extrinsics.append(np.array(pose.pose))
            extrinsics = np.array(extrinsics)

            if not os.path.exists(DIR_OUTPUT):
                os.mkdir(DIR_OUTPUT)

            path = os.path.join(DIR_OUTPUT,
                                f'extinsics_finetuned_{experiment}.pkl')

            with open(path, 'wb') as handle:
                pickle.dump(extrinsics, handle,
                            protocol=pickle.HIGHEST_PROTOCOL)

        vis = o3d.visualization.VisualizerWithKeyCallback()
        vis.create_window(visible=True)    
        vis.register_key_callback(83, key_callback)
        # Call only after creating visualizer window.
        vis.get_render_option().background_color = COLOR_SPACE_GRAY
        vis.get_render_option().show_coordinate_frame = True
        for point_id, pcd_down in enumerate(pcds):
            pcd_down_copy = copy.deepcopy(pcd_down)
            pcd_down_copy.transform(pose_graph.nodes[point_id].pose)
            vis.add_geometry(pcd_down_copy)
        vis.run()

    return pose_avg


# TODO: Move to config file
VOXEL_SIZE = .005
EXPERIMENT = 'a2'
INTERVAL = 25

# TODO: Move cameras to dataloader
cam24 = 'azure_kinect2_4_calib_snap'
cam15 = 'azure_kinect1_5_calib_snap'
cam14 = 'azure_kinect1_4_calib_snap'
cam34 = 'azure_kinect3_4_calib_snap'
cam35 = 'azure_kinect3_5_calib_snap'
if __name__ == "__main__":
    cache = diskcache.Cache('../calibration/cache')

    cams = [
        cam24,
        cam15,
        # cam14,
        cam34,
        cam35
    ]
    
    for file in sorted(os.listdir(DIR_INPUT), reverse=False):
        DIR_OUTPUT = './extrinsics_finetuned'
        experiment = file.split('.')[0].split('_')[1]
        file_path = os.path.join(DIR_INPUT, file)
        DIR_OUTPUT = './extrinsics_finetuned'
        print(f"Visualizing {file_path}")
    
        with open(file_path, 'rb') as handle:
            output = pickle.load(handle)

        poses = output['points_3d'].reshape(-1, 52, 3)
        params = output['params']

        finetune_extrinsics(
            cams, EXPERIMENT, INTERVAL, VOXEL_SIZE, params, cache)
