import sys
sys.path.append('../')

import os
import cv2
import time
import diskcache
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans

from utils import data_loader
from calibration import rgb_depth_map


MESH = False


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

    intrinsics, extrinsics = get_parameters(cam, cache)

    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(color, depth, convert_rgb_to_intensity=False)
    pc = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsics, extrinsics)

    # pc = o3d.geometry.PointCloud.create_from_depth_image(depth, intrinsics, extrinsics)

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

    # Cam14
    path = data_loader.EXPERIMENTS[experiment][cam14]
    pc14 = load_pointcloud(path, cam14, idx, cache)
    pc14 = preprocess(pc14)

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


# TODO: I hate this part. Refactor please! Move to a file.
A1_S0 = [
    np.array(
        [
            [1.00000000e00, -6.60128903e-24, 0.00000000e00, 0.00000000e00],
            [-6.61744490e-24, 1.00000000e00, 4.23516474e-22, 4.23516474e-22],
            [3.38813179e-21, 2.11758237e-22, 1.00000000e00, 0.00000000e00],
            [0.00000000e00, 0.00000000e00, 0.00000000e00, 1.00000000e00],
        ]
    ),
    np.array(
        [
            [0.99952994, -0.01112429, 0.02856824, -0.10080341],
            [0.01060807, 0.99977885, 0.01815839, -0.02969474],
            [-0.02876392, -0.0178468, 0.9994269, 0.11301244],
            [0.0, 0.0, 0.0, 1.0],
        ]
    ),
    np.array(
        [
            [0.98304617, -0.03331575, 0.18030609, -0.36168713],
            [0.04165163, 0.99822173, -0.04264401, 0.02754495],
            [-0.17856475, 0.04943107, 0.98268571, 0.3484725],
            [0.0, 0.0, 0.0, 1.0],
        ]
    ),
    np.array(
        [
            [0.98697537, -0.03752174, 0.15643442, -0.32977598],
            [0.04923893, 0.99620933, -0.07171119, 0.07267861],
            [-0.15315071, 0.07847984, 0.98508161, 0.40158207],
            [0.0, 0.0, 0.0, 1.0],
        ]
    ),
    np.array(
        [
            [0.99230664, -0.11847563, 0.03593131, 0.05164638],
            [0.11909673, 0.99275914, -0.01566076, -0.16954756],
            [-0.03381572, 0.01981958, 0.99923155, 0.2724472],
            [0.0, 0.0, 0.0, 1.0],
        ]
    ),
]


A1_S1 = [
    np.array(
        [
            [1.00000000e00, 9.58269353e-22, 0.00000000e00, -6.77626358e-21],
            [1.10667719e-22, 1.00000000e00, 0.00000000e00, -2.11758237e-22],
            [3.38813179e-21, 0.00000000e00, 1.00000000e00, 0.00000000e00],
            [0.00000000e00, 0.00000000e00, 0.00000000e00, 1.00000000e00],
        ]
    ),
    np.array(
        [
            [0.99175747, -0.06157104, 0.11236601, -0.25912419],
            [0.05717899, 0.9974842, 0.04190276, -0.03833833],
            [-0.11466331, -0.0351324, 0.99278298, 0.0226211],
            [0.0, 0.0, 0.0, 1.0],
        ]
    ),
    np.array(
        [
            [0.98776962, -0.04775228, 0.14842806, -0.30381025],
            [0.04739478, 0.99885853, 0.00594664, -0.07769243],
            [-0.1485426, 0.0011608, 0.98890533, 0.23097955],
            [0.0, 0.0, 0.0, 1.0],
        ]
    ),
    np.array(
        [
            [0.97873009, -0.08813203, 0.18525702, -0.26444784],
            [0.08553369, 0.99609264, 0.02198713, -0.05627334],
            [-0.18647092, -0.00567375, 0.9824441, 0.14828827],
            [0.0, 0.0, 0.0, 1.0],
        ]
    ),
]



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
        cam14,
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
            pcds_down[point_id].transform(A1_S0[point_id])

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
