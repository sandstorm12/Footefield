import sys
sys.path.append('../')

import os
import copy
import diskcache
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans

from utils import data_loader


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

    pointcloud = pointcloud.voxel_down_sample(voxel_size=0.02)

    return pointcloud


def show(pointcloud):
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=True)
    vis.get_render_option().background_color = [.9, .9, .9]
    vis.add_geometry(pointcloud)
    vis.run()


def get_subject(experiment, idx, cache):
    # Cam24
    path = data_loader.EXPERIMENTS[experiment][cam24]
    pc24 = load_pointcloud(path, cam24, 0, cache)
    pc24 = preprocess(pc24)
    pc_np = np.asarray(pc24.points)
    kmeans = KMeans(n_clusters=2, random_state=47, n_init="auto").fit(pc_np)
    # TODO: Explain what (idx + 1 % 2) is
    pc24.points = o3d.utility.Vector3dVector(pc_np[kmeans.labels_ == (idx + 1) % 2])
    
    # Cam15
    path = data_loader.EXPERIMENTS[experiment][cam15]
    pc15 = load_pointcloud(path, cam15, 0, cache)
    pc15 = preprocess(pc15)
    pc_np = np.asarray(pc15.points)
    kmeans = KMeans(n_clusters=2, random_state=47, n_init="auto").fit(pc_np)
    pc15.points = o3d.utility.Vector3dVector(pc_np[kmeans.labels_ == (idx) % 2])

    # Cam14
    path = data_loader.EXPERIMENTS[experiment][cam14]
    pc14 = load_pointcloud(path, cam14, 0, cache)
    pc14 = preprocess(pc14)

    # Cam34
    path = data_loader.EXPERIMENTS[experiment][cam34]
    pc34 = load_pointcloud(path, cam34, 0, cache)
    pc34 = preprocess(pc34)
    pc_np = np.asarray(pc34.points)
    kmeans = KMeans(n_clusters=2, random_state=47, n_init="auto").fit(pc_np)
    pc34.points = o3d.utility.Vector3dVector(pc_np[kmeans.labels_ == (idx + 1) % 2])

    # Cam35
    path = data_loader.EXPERIMENTS[experiment][cam35]
    pc35 = load_pointcloud(path, cam35, 0, cache)
    pc35 = preprocess(pc35)
    pc_np = np.asarray(pc35.points)
    kmeans = KMeans(n_clusters=2, random_state=47, n_init="auto").fit(pc_np)
    pc35.points = o3d.utility.Vector3dVector(pc_np[kmeans.labels_ == (idx + 1) % 2])

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


def key_callback_w(vis):
    transition[1] += .01
    pc_source.translate(np.array([0, .01, 0]))
    vis.update_geometry(pc_source)

    print(transition, angle)


def key_callback_s(vis):
    transition[1] += -.01
    pc_source.translate(np.array([0, -.01, 0]))
    vis.update_geometry(pc_source)

    print(transition, angle)


def key_callback_a(vis):
    transition[0] += .01
    pc_source.translate(np.array([.01, 0, 0]))
    vis.update_geometry(pc_source)

    print(transition, angle)


def key_callback_d(vis):
    transition[0] += -.01
    pc_source.translate(np.array([-.01, 0, 0]))
    vis.update_geometry(pc_source)

    print(transition, angle)


def key_callback_e(vis):
    transition[2] += .01
    pc_source.translate(np.array([0, 0, 0.01]))
    vis.update_geometry(pc_source)

    print(transition, angle)


def key_callback_r(vis):
    transition[2] += -.01
    pc_source.translate(np.array([0, 0, -0.01]))
    vis.update_geometry(pc_source)

    print(transition, angle)


def key_callback_t(vis):
    angle[1] += .1
    rm = rotation_matrix_from_euler_angles(0, np.deg2rad(1), 0)
    pc_source.rotate(rm)
    vis.update_geometry(pc_source)

    print(transition, angle)


def key_callback_g(vis):
    angle[1] += -.1
    rm = rotation_matrix_from_euler_angles(0, np.deg2rad(-1), 0)
    pc_source.rotate(rm)
    vis.update_geometry(pc_source)

    print(transition, angle)


def key_callback_f(vis):
    angle[0] += .1
    rm = rotation_matrix_from_euler_angles(np.deg2rad(1), 0, 0)
    pc_source.rotate(rm)
    vis.update_geometry(pc_source)

    print(transition, angle)


def key_callback_h(vis):
    angle[0] += -.1
    rm = rotation_matrix_from_euler_angles(np.deg2rad(-1), 0, 0)
    pc_source.rotate(rm)
    vis.update_geometry(pc_source)

    print(transition, angle)


def key_callback_y(vis):
    angle[2] += .1
    rm = rotation_matrix_from_euler_angles(0, 0, np.deg2rad(1))
    pc_source.rotate(rm)
    vis.update_geometry(pc_source)

    print(transition, angle)


def key_callback_u(vis):
    angle[2] += -.1
    rm = rotation_matrix_from_euler_angles(0, 0, np.deg2rad(-1))
    pc_source.rotate(rm)
    vis.update_geometry(pc_source)

    print(transition, angle)


cam24 = 'azure_kinect2_4_calib_snap'
cam15 = 'azure_kinect1_5_calib_snap'
cam14 = 'azure_kinect1_4_calib_snap'
cam34 = 'azure_kinect3_4_calib_snap'
cam35 = 'azure_kinect3_5_calib_snap'
if __name__ == "__main__":
    cache = diskcache.Cache('../calibration/cache')

    subject_1 = get_subject('a1', 0, cache)

    # pc_s1 = subject_1[cam24] + \
    #     subject_1[cam15] + \
    #     subject_1[cam34] + \
    #     subject_1[cam35]
    # show(pc_s1)

    pc_target = subject_1[cam34]
    pc_source = subject_1[cam35]

    transition = [0, 0, 0]
    angle = [0, 0, 0]
    rm = rotation_matrix_from_euler_angles(np.deg2rad(angle[0]),
                                        np.deg2rad(angle[1]),
                                        np.deg2rad(angle[2]))
    pc_source = pc_source.rotate(rm)
    pc_source = pc_source.translate(transition)

    pc_target.paint_uniform_color([0, 0, 1])
    pc_source.paint_uniform_color([0, 1, 0])

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window()

    vis.add_geometry(pc_target)
    vis.add_geometry(pc_source)

    opt = vis.get_render_option()
    opt.show_coordinate_frame = True
    opt.background_color = [.5, .5, .5]

    vis.register_key_callback(87, key_callback_w)
    vis.register_key_callback(83, key_callback_s)
    vis.register_key_callback(65, key_callback_a)
    vis.register_key_callback(68, key_callback_d)
    vis.register_key_callback(69, key_callback_e)
    vis.register_key_callback(82, key_callback_r)

    vis.register_key_callback(84, key_callback_t)
    vis.register_key_callback(71, key_callback_g)
    vis.register_key_callback(70, key_callback_f)
    vis.register_key_callback(72, key_callback_h)
    vis.register_key_callback(89, key_callback_y)
    vis.register_key_callback(85, key_callback_u)

    vis.run()
    vis.destroy_window()
