import sys
sys.path.append('../')

import os
import cv2
import time
import pickle
import diskcache
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans

from utils import data_loader
from calibration import rgb_depth_map


MESH = False
PARAM_CALIB_SIZE = 16


def load_pointcloud(path, cam, idx, cache):
    path_depth = os.path.join(path, 'depth/depth{:05d}.png'.format(idx))
    path_color = os.path.join(path, 'color/color{:05d}.jpg'.format(idx))

    color = cv2.imread(path_color)
    color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
    color = data_loader.downsample_keep_aspect_ratio(
        color,
        (
            data_loader.IMAGE_INFRARED_WIDTH,
            data_loader.IMAGE_INFRARED_HEIGHT
        )
    )
    color = rgb_depth_map.align_image_rgb(color, cam, cache)
    color = o3d.geometry.Image((color).astype(np.uint8))

    depth = o3d.io.read_image(path_depth)

    intrinsics, extrinsics = get_parameters(cam)

    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(color, depth, convert_rgb_to_intensity=False)
    pc = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsics, extrinsics)

    # pc = o3d.geometry.PointCloud.create_from_depth_image(depth, intrinsics, extrinsics)

    return pc


# TODO: Shorten
def get_parameters(cam):
    file_path = '../pose_estimation/keypoints_3d_ba/keypoints3d_a1_ba.pkl'
    with open(file_path, 'rb') as handle:
        output = pickle.load(handle)

    params = output['params']

    if cam == cam24:
        base_index = 0 * PARAM_CALIB_SIZE
    elif cam == cam15:
        base_index = 1 * PARAM_CALIB_SIZE
    elif cam == cam14:
        raise Exception('Invalid camera.')
    elif cam == cam34:
        base_index = 2 * PARAM_CALIB_SIZE
    elif cam == cam35:
        base_index = 3 * PARAM_CALIB_SIZE
    
    mtx = np.zeros((3, 3), dtype=float)
    mtx[0, 0] = params[base_index + 12]
    mtx[1, 1] = params[base_index + 13]
    mtx[0, 2] = params[base_index + 14]
    mtx[1, 2] = params[base_index + 15]

    intrinsics = o3d.camera.PinholeCameraIntrinsic(
        data_loader.IMAGE_INFRARED_WIDTH,
        data_loader.IMAGE_INFRARED_HEIGHT,
        mtx[0, 0], mtx[1, 1], mtx[0, 2], mtx[1, 2])
    
    rotation = params[base_index:base_index + 9].reshape(3, 3)
    translation = params[base_index + 9:base_index + 12] / 1000
    extrinsics = np.identity(4, dtype=float)
    extrinsics[:3, :3] = rotation
    extrinsics[:3, 3] = translation.reshape(3)

    return intrinsics, extrinsics


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


def get_subject(experiment, subject, idx, cache):
    start_pts = np.array([[0, 0, 0], [1, -1, 3]])
    
    # Cam24
    path = data_loader.EXPERIMENTS[experiment][cam24]
    pc24 = load_pointcloud(path, cam24, idx, cache)
    pc24 = preprocess(pc24)
    pc24_np = np.asarray(pc24.points)
    pc24c_np = np.asarray(pc24.colors)
    kmeans = KMeans(n_clusters=2, random_state=47, init=start_pts, n_init=1).fit(pc24_np)
    # TODO: Explain what (subject + 1 % 2) is
    pc24.points = o3d.utility.Vector3dVector(pc24_np[kmeans.labels_ == (subject + 1) % 2])
    pc24.colors = o3d.utility.Vector3dVector(pc24c_np[kmeans.labels_ == (subject + 1) % 2])
    
    # Cam15
    path = data_loader.EXPERIMENTS[experiment][cam15]
    pc15 = load_pointcloud(path, cam15, idx, cache)
    pc15 = preprocess(pc15)
    pc15_np = np.asarray(pc15.points)
    pc15c_np = np.asarray(pc15.colors)
    kmeans = KMeans(n_clusters=2, random_state=47, init=start_pts, n_init=1).fit(pc15_np)
    pc15.points = o3d.utility.Vector3dVector(pc15_np[kmeans.labels_ == (subject + 1) % 2])
    pc15.colors = o3d.utility.Vector3dVector(pc15c_np[kmeans.labels_ == (subject + 1) % 2])

    # # Cam14
    # path = data_loader.EXPERIMENTS[experiment][cam14]
    # pc14 = load_pointcloud(path, cam14, idx, cache)
    # pc14 = preprocess(pc14)

    # Cam34
    path = data_loader.EXPERIMENTS[experiment][cam34]
    pc34 = load_pointcloud(path, cam34, idx, cache)
    pc34 = preprocess(pc34)
    pc34_np = np.asarray(pc34.points)
    pc34c_np = np.asarray(pc34.colors)
    kmeans = KMeans(n_clusters=2, random_state=47, init=start_pts, n_init=1).fit(pc34_np)
    pc34.points = o3d.utility.Vector3dVector(pc34_np[kmeans.labels_ == (subject + 1) % 2])
    pc34.colors = o3d.utility.Vector3dVector(pc34c_np[kmeans.labels_ == (subject + 1) % 2])

    # Cam35
    path = data_loader.EXPERIMENTS[experiment][cam35]
    pc35 = load_pointcloud(path, cam35, idx, cache)
    pc35 = preprocess(pc35)
    pc35_np = np.asarray(pc35.points)
    pc35c_np = np.asarray(pc35.colors)
    kmeans = KMeans(n_clusters=2, random_state=47, init=start_pts, n_init=1).fit(pc35_np)
    pc35.points = o3d.utility.Vector3dVector(pc35_np[kmeans.labels_ == (subject + 1) % 2])
    pc35.colors = o3d.utility.Vector3dVector(pc35c_np[kmeans.labels_ == (subject + 1) % 2])

    return {
        cam24: pc24,
        cam15: pc15,
        # cam14: pc14,
        cam34: pc34,
        cam35: pc35,
    }


EXPERIMENT = 'a1'
SUBJECT = 0
COLOR_SPACE_GRAY = [0.203921569, 0.239215686, 0.274509804]


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

    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.get_render_option().background_color = COLOR_SPACE_GRAY
    vis.get_render_option().show_coordinate_frame = True

    geometry = o3d.geometry.PointCloud()
    geometry_mesh = o3d.geometry.TriangleMesh()

    for i in range(1000):
        subject = get_subject(EXPERIMENT, SUBJECT, i, cache)

        pcds_down = [subject[cam] for cam in cams]

        # for point_id in range(len(pcds_down)):
        #     pcds_down[point_id].transform(A1_S0[point_id])

        pc_combines = pcds_down[0]
        for idx in range(1, len(pcds_down)):
            pc_combines += pcds_down[idx]

        geometry.points = pc_combines.points
        # Add full color points clouds
        geometry.colors = pc_combines.colors
        if i == 0:
            vis.add_geometry(geometry)
        else:
            vis.update_geometry(geometry)

        if MESH:
            alpha = .02
            voxel_down_pcd = pc_combines.voxel_down_sample(voxel_size=0.02)
            mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(voxel_down_pcd, alpha)
            mesh = mesh.filter_smooth_simple(number_of_iterations=1)
            mesh.compute_vertex_normals()
            mesh.compute_triangle_normals()

            geometry_mesh.vertices = mesh.vertices
            geometry_mesh.triangles = mesh.triangles
            geometry_mesh.vertex_normals = mesh.vertex_normals
            geometry_mesh.triangle_normals = mesh.triangle_normals
            if i == 0:
                vis.add_geometry(geometry_mesh)
                vis.get_render_option().mesh_show_back_face = True
            else:
                vis.update_geometry(geometry_mesh)
            
        vis.poll_events()
        vis.update_renderer()

        print(f"Update {i}: {time.time()}")
        # time.sleep(.05)
