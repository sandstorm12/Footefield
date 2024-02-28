import sys
sys.path.append('../')

import os
import time
import copy
import diskcache
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans

from utils import data_loader


MESH = False


def load_pointcloud(path, cam, idx, cache):
    path_depth = os.path.join(path, 'depth/depth{:05d}.png'.format(idx))
    depth = o3d.io.read_image(path_depth)

    intrinsics, extrinsics = get_parameters(cam, cache)

    pc = o3d.geometry.PointCloud.create_from_depth_image(depth, intrinsics, extrinsics)

    return pc


# TODO: Shorten
def get_parameters(cam, cache):
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
    
    R = cache['extrinsics'][cam24 + 'infrared']['rotation']
    T = cache['extrinsics'][cam24 + 'infrared']['transition']
    R2 = cache['extrinsics'][cam15 + 'infrared']['rotation']
    T2 = cache['extrinsics'][cam15 + 'infrared']['transition']
    R3 = cache['extrinsics'][cam14 + 'infrared']['rotation']
    T3 = cache['extrinsics'][cam14 + 'infrared']['transition']
    R4 = cache['extrinsics'][cam34 + 'infrared']['rotation']
    T4 = cache['extrinsics'][cam34 + 'infrared']['transition']

    intrinsics = o3d.camera.PinholeCameraIntrinsic(
        data_loader.IMAGE_INFRARED_WIDTH,
        data_loader.IMAGE_INFRARED_HEIGHT,
        mtx[0, 0], mtx[1, 1], mtx[0, 2], mtx[1, 2])
    
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


def show(pointcloud):
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=True)
    vis.get_render_option().background_color = [.5, .5, .5]
    vis.add_geometry(pointcloud)
    vis.run()


def get_subject(experiment, subject, idx, cache):
    start_pts = np.array([[0, 0, 0], [1, -1, 3]])
    
    # Cam24
    path = data_loader.EXPERIMENTS[experiment][cam24]
    pc24 = load_pointcloud(path, cam24, idx, cache)
    pc24 = preprocess(pc24)
    pc_np = np.asarray(pc24.points)
    kmeans = KMeans(n_clusters=2, random_state=47, init=start_pts, n_init=1).fit(pc_np)
    # TODO: Explain what (subject + 1 % 2) is
    pc24.points = o3d.utility.Vector3dVector(pc_np[kmeans.labels_ == (subject + 1) % 2])
    
    # Cam15
    path = data_loader.EXPERIMENTS[experiment][cam15]
    pc15 = load_pointcloud(path, cam15, idx, cache)
    pc15 = preprocess(pc15)
    pc_np = np.asarray(pc15.points)
    kmeans = KMeans(n_clusters=2, random_state=47, init=start_pts, n_init=1).fit(pc_np)
    pc15.points = o3d.utility.Vector3dVector(pc_np[kmeans.labels_ == (subject + 1) % 2])

    # Cam14
    path = data_loader.EXPERIMENTS[experiment][cam14]
    pc14 = load_pointcloud(path, cam14, idx, cache)
    pc14 = preprocess(pc14)

    # Cam34
    path = data_loader.EXPERIMENTS[experiment][cam34]
    pc34 = load_pointcloud(path, cam34, idx, cache)
    pc34 = preprocess(pc34)
    pc_np = np.asarray(pc34.points)
    kmeans = KMeans(n_clusters=2, random_state=47, init=start_pts, n_init=1).fit(pc_np)
    pc34.points = o3d.utility.Vector3dVector(pc_np[kmeans.labels_ == (subject + 1) % 2])

    # Cam35
    path = data_loader.EXPERIMENTS[experiment][cam35]
    pc35 = load_pointcloud(path, cam35, idx, cache)
    pc35 = preprocess(pc35)
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


def rotation_matrix_from_euler_angles(roll, pitch, yaw):
    """
        Creates a 3x3 rotation matrix from euler angles (roll, pitch, yaw).

        Args:
        roll (float): Roll angle in radians.
        pitch (float): Pitch angle in radians.
        yaw (float): Yaw angle in radians.

        Returns:
        rotation_matrix (np.ndarray): 3x3 rotation matrix.
    """

    rotation_matrix = np.identity(3)

    # Roll
    c_roll = np.cos(roll)
    s_roll = np.sin(roll)

    rotation_matrix = np.dot(rotation_matrix, np.array(
        [[1, 0, 0],
        [0, c_roll, -s_roll],
        [0, s_roll, c_roll]]))

    # Pitch
    c_pitch = np.cos(pitch)
    s_pitch = np.sin(pitch)

    rotation_matrix = np.dot(rotation_matrix, np.array(
        [[c_pitch, 0, s_pitch],
        [0, 1, 0],
        [-s_pitch, 0, c_pitch]]))

    # Yaw
    c_yaw = np.cos(yaw)
    s_yaw = np.sin(yaw)

    rotation_matrix = np.dot(rotation_matrix, np.array(
        [[c_yaw, s_yaw, 0],
        [-s_yaw, c_yaw, 0],
        [0, 0, 1]]))

    return rotation_matrix


# TODO: I hate this part. Refactor please!
A1_S0 = [
    np.array(
        [
            [1.00000000e00, -1.03459615e-21, 0.00000000e00, 0.00000000e00],
            [-1.03459615e-21, 1.00000000e00, 8.47032947e-22, 0.00000000e00],
            [-1.35525272e-20, -8.47032947e-22, 1.00000000e00, 0.00000000e00],
            [0.00000000e00, 0.00000000e00, 0.00000000e00, 1.00000000e00],
        ]
    ),
    np.array(
        [
            [9.99916624e-01, 1.28957383e-02, 6.66547295e-04, -4.70369841e-02],
            [-1.29022617e-02, 9.99856895e-01, 1.09417583e-02, -2.78448444e-03],
            [-5.25349858e-04, -1.09494459e-02, 9.99939915e-01, 7.94679445e-02],
            [0.00000000e00, 0.00000000e00, 0.00000000e00, 1.00000000e00],
        ]
    ),
    np.array(
        [
            [0.98649811, -0.03075712, 0.16085858, -0.32690532],
            [0.03765095, 0.99849067, -0.0399848, 0.00724568],
            [-0.15938598, 0.0455014, 0.98616719, 0.32749762],
            [0.0, 0.0, 0.0, 1.0],
        ]
    ),
    np.array(
        [
            [0.97858678, -0.06511944, 0.19526231, -0.41950537],
            [0.07997343, 0.99439407, -0.06917138, 0.01705819],
            [-0.18966328, 0.08330599, 0.97830872, 0.42940934],
            [0.0, 0.0, 0.0, 1.0],
        ]
    ),
    np.array(
        [
            [0.96704779, -0.13024607, 0.21875681, -0.36226442],
            [0.15761733, 0.9810527, -0.11266044, -0.02076312],
            [-0.19993838, 0.14342789, 0.96925388, 0.48615835],
            [0.0, 0.0, 0.0, 1.0],
        ]
    ),
]

A1_S1 = [
    np.array(
        [
            [1.00000000e00, 5.11151574e-23, 0.00000000e00, 0.00000000e00],
            [-7.95946466e-22, 1.00000000e00, -2.64697796e-23, 2.11758237e-22],
            [3.38813179e-21, -2.64697796e-23, 1.00000000e00, 0.00000000e00],
            [0.00000000e00, 0.00000000e00, 0.00000000e00, 1.00000000e00],
        ]
    ),
    np.array(
        [
            [0.96234239, -0.07933012, 0.2600074, -0.49241727],
            [0.0796393, 0.99677978, 0.00936275, 0.01397312],
            [-0.25991287, 0.01169664, 0.96556123, 0.04237567],
            [0.0, 0.0, 0.0, 1.0],
        ]
    ),
    np.array(
        [
            [0.94058297, -0.10176303, 0.32395673, -0.54400412],
            [0.11534682, 0.99306, -0.02295511, -0.07126208],
            [-0.31937249, 0.05895857, 0.94579337, 0.30544206],
            [0.0, 0.0, 0.0, 1.0],
        ]
    ),
    np.array(
        [
            [0.91785645, -0.05935663, 0.39244913, -0.56627694],
            [0.08673478, 0.99485313, -0.05238628, 0.08786158],
            [-0.38731978, 0.08212208, 0.91828065, 0.30544401],
            [0.0, 0.0, 0.0, 1.0],
        ]
    ),
]


EXPERIMENT = 'a1'
SUBJECT = 1
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

        for point_id in range(len(pcds_down)):
            pcds_down[point_id].transform(A1_S1[point_id])

        pc_combines = pcds_down[0]
        for idx in range(1, len(pcds_down)):
            pc_combines += pcds_down[idx]

        geometry.points = pc_combines.points        
        if i == 0:
            vis.add_geometry(geometry)
        else:
            vis.update_geometry(geometry)

        o3d.io.write_point_cloud(f"./pointclouds/{EXPERIMENT}_{SUBJECT}_{i}.pcd", geometry)

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
