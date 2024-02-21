# TODO: Needs serious refactor!

import sys
sys.path.append('../')

import os
import copy
import diskcache
import numpy as np
import open3d as o3d

from tqdm import tqdm
from utils import data_loader
from sklearn.cluster import KMeans


COLOR_SPACE_GRAY = [0.203921569, 0.239215686, 0.274509804]


def load_pointcloud(path, cam, idx, cache):
    path_depth = os.path.join(path, 'depth/depth{:05d}.png'.format(idx))
    depth = o3d.io.read_image(path_depth)

    intrinsics, extrinsics = get_parameters(cam, cache)

    pc = o3d.geometry.PointCloud.create_from_depth_image(depth, intrinsics, extrinsics)

    return pc


def count_depth_images(experiment):
    path_first_cam = list(data_loader.EXPERIMENTS[experiment].values())[0]
    path = os.path.join(path_first_cam, 'depth')
    file_count = len(os.listdir(path))

    return file_count


def get_mtx(cam, cache):
    if cam == cam24:
        mtx = cache['extrinsics'][cam24 + 'infrared']['mtx_l']
    elif cam == cam15:
        mtx = cache['extrinsics'][cam24 + 'infrared']['mtx_r']
    elif cam == cam14:
        mtx = cache['extrinsics'][cam15 + 'infrared']['mtx_r']
    elif cam == cam34:
        mtx = cache['extrinsics'][cam14 + 'infrared']['mtx_r']
    elif cam == cam35:
        mtx = cache['extrinsics'][cam34 + 'infrared']['mtx_r']

    return mtx


def get_rotation_transition(cache):
    R = cache['extrinsics'][cam24 + 'infrared']['rotation']
    T = cache['extrinsics'][cam24 + 'infrared']['transition']
    R2 = cache['extrinsics'][cam15 + 'infrared']['rotation']
    T2 = cache['extrinsics'][cam15 + 'infrared']['transition']
    R3 = cache['extrinsics'][cam14 + 'infrared']['rotation']
    T3 = cache['extrinsics'][cam14 + 'infrared']['transition']
    R4 = cache['extrinsics'][cam34 + 'infrared']['rotation']
    T4 = cache['extrinsics'][cam34 + 'infrared']['transition']

    return R, T, R2, T2, R3, T3, R4, T4


def get_extrinsics(cam, R, T, R2, T2, R3, T3, R4, T4):
    extrinsics = np.identity(4)
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
    mtx = get_mtx(cam, cache)
    
    R, T, R2, T2, R3, T3, R4, T4 = get_rotation_transition(cache)

    intrinsics = o3d.camera.PinholeCameraIntrinsic(
        data_loader.IMAGE_INFRARED_WIDTH,
        data_loader.IMAGE_INFRARED_HEIGHT,
        mtx[0, 0], mtx[1, 1], mtx[0, 2], mtx[1, 2])
    
    extrinsics = get_extrinsics(cam, R, T, R2, T2, R3, T3, R4, T4)

    return intrinsics, extrinsics


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
    vis.get_render_option().background_color = [0.203921569, 0.239215686, 0.274509804]
    vis.get_render_option().show_coordinate_frame = True
    vis.add_geometry(pointcloud)
    vis.run()


def get_subject(experiment, subject, idx, voxel_size, cache):
    start_pts = np.array([[-0.24240173, -0.1135224, 1.48424285],
                          [0.76507662, -0.2935405, 2.00135939]])

    # Cam24
    path = data_loader.EXPERIMENTS[experiment][cam24]
    pc24 = load_pointcloud(path, cam24, idx, cache)
    pc24 = preprocess(pc24, voxel_size)
    pc_np = np.asarray(pc24.points)
    kmeans = KMeans(n_clusters=2, random_state=47, init=start_pts, n_init=1).fit(pc_np)
    # TODO: Explain what (subject + 1 % 2) is
    pc24.points = o3d.utility.Vector3dVector(pc_np[kmeans.labels_ == (subject + 1) % 2])
    
    # Cam15
    path = data_loader.EXPERIMENTS[experiment][cam15]
    pc15 = load_pointcloud(path, cam15, idx, cache)
    pc15 = preprocess(pc15, voxel_size)
    pc_np = np.asarray(pc15.points)
    kmeans = KMeans(n_clusters=2, random_state=47, init=start_pts, n_init=1).fit(pc_np)
    pc15.points = o3d.utility.Vector3dVector(pc_np[kmeans.labels_ == (subject + 1) % 2])

    # Cam14
    path = data_loader.EXPERIMENTS[experiment][cam14]
    pc14 = load_pointcloud(path, cam14, idx, cache)
    pc14 = preprocess(pc14, voxel_size)

    # Cam34
    path = data_loader.EXPERIMENTS[experiment][cam34]
    pc34 = load_pointcloud(path, cam34, idx, cache)
    pc34 = preprocess(pc34, voxel_size)
    pc_np = np.asarray(pc34.points)
    kmeans = KMeans(n_clusters=2, random_state=47, init=start_pts, n_init=1).fit(pc_np)
    pc34.points = o3d.utility.Vector3dVector(pc_np[kmeans.labels_ == (subject + 1) % 2])

    # Cam35
    path = data_loader.EXPERIMENTS[experiment][cam35]
    pc35 = load_pointcloud(path, cam35, idx, cache)
    pc35 = preprocess(pc35, voxel_size)
    pc_np = np.asarray(pc35.points)
    kmeans = KMeans(n_clusters=2, random_state=47, init=start_pts, n_init=1).fit(pc_np)
    pc35.points = o3d.utility.Vector3dVector(pc_np[kmeans.labels_ == (subject + 1) % 2])

    return {
        cam24: pc24,
        cam15: pc15,
        cam14: pc14,
        cam34: pc34,
        cam35: pc35,
    }


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
    pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(odometry))
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
                    o3d.pipelines.registration.PoseGraphEdge(source_id,
                                                             target_id,
                                                             transformation_icp,
                                                             information_icp,
                                                             uncertain=False))
            else:  # loop closure case
                pose_graph.edges.append(
                    o3d.pipelines.registration.PoseGraphEdge(source_id,
                                                             target_id,
                                                             transformation_icp,
                                                             information_icp,
                                                             uncertain=True))
    return pose_graph


def finetune_extrinsics(cams, experiment, subject, interval, voxel_size, cache):
    file_count = count_depth_images(experiment)

    pose_avg = [None] * 5
    for i in tqdm(range(0, file_count, interval)):
        pcs_subject = get_subject('a1', subject, i, voxel_size, cache)

        for cam in cams:
            pcs_subject[cam].estimate_normals()

        pcds_down = [pcs_subject[cam] for cam in cams]

        max_correspondence_distance_coarse = voxel_size * 15
        max_correspondence_distance_fine = voxel_size * 1.5
        with o3d.utility.VerbosityContextManager(
                o3d.utility.VerbosityLevel.Info) as cm:
            pose_graph = full_registration(
                pcds_down,
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

        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=True)
        # Call only after creating visualizer window.
        vis.get_render_option().background_color = COLOR_SPACE_GRAY
        vis.get_render_option().show_coordinate_frame = True
        pose = []
        for point_id, pcd_down in enumerate(pcds_down):
            pcd_down_copy = copy.deepcopy(pcd_down)
            pcd_down_copy.transform(pose_graph.nodes[point_id].pose)
            pose.append(np.array(pose_graph.nodes[point_id].pose))
            vis.add_geometry(pcd_down_copy)

        print(pose, '\n')

        vis.run()

    return pose_avg


# TODO: Move to config file
VOXEL_SIZE = .005
EXPERIMENT = 'a1'
INTERVAL = 25
SUBJECT = 1

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

    finetune_extrinsics(cams, EXPERIMENT, SUBJECT, INTERVAL, VOXEL_SIZE, cache)
