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

    pointcloud = pointcloud.voxel_down_sample(voxel_size=0.04)

    return pointcloud


def show(pointcloud):
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=True)
    vis.get_render_option().background_color = [.9, .9, .9]
    vis.add_geometry(pointcloud)
    vis.run()


def get_subject(experiment, idx, cache):
    start_pts = np.array([[0, 0, 0], [2e3, 2e3, 2e3]])

    # Cam24
    path = data_loader.EXPERIMENTS[experiment][cam24]
    pc24 = load_pointcloud(path, cam24, 0, cache)
    pc24 = preprocess(pc24)
    pc_np = np.asarray(pc24.points)
    kmeans = KMeans(n_clusters=2, random_state=47, init=start_pts, n_init=1).fit(pc_np)
    # TODO: Explain what (idx + 1 % 2) is
    pc24.points = o3d.utility.Vector3dVector(pc_np[kmeans.labels_ == (idx + 1) % 2])
    
    # Cam15
    path = data_loader.EXPERIMENTS[experiment][cam15]
    pc15 = load_pointcloud(path, cam15, 0, cache)
    pc15 = preprocess(pc15)
    pc_np = np.asarray(pc15.points)
    kmeans = KMeans(n_clusters=2, random_state=47, init=start_pts, n_init=1).fit(pc_np)
    pc15.points = o3d.utility.Vector3dVector(pc_np[kmeans.labels_ == (idx + 1) % 2])

    # Cam14
    path = data_loader.EXPERIMENTS[experiment][cam14]
    pc14 = load_pointcloud(path, cam14, 0, cache)
    pc14 = preprocess(pc14)

    # Cam34
    path = data_loader.EXPERIMENTS[experiment][cam34]
    pc34 = load_pointcloud(path, cam34, 0, cache)
    pc34 = preprocess(pc34)
    pc_np = np.asarray(pc34.points)
    kmeans = KMeans(n_clusters=2, random_state=47, init=start_pts, n_init=1).fit(pc_np)
    pc34.points = o3d.utility.Vector3dVector(pc_np[kmeans.labels_ == (idx + 1) % 2])

    # Cam35
    path = data_loader.EXPERIMENTS[experiment][cam35]
    pc35 = load_pointcloud(path, cam35, 0, cache)
    pc35 = preprocess(pc35)
    pc_np = np.asarray(pc35.points)
    kmeans = KMeans(n_clusters=2, random_state=47, init=start_pts, n_init=1).fit(pc_np)
    pc35.points = o3d.utility.Vector3dVector(pc_np[kmeans.labels_ == (idx + 1) % 2])

    return {
        cam24: pc24,
        cam15: pc15,
        cam14: pc14,
        cam34: pc34,
        cam35: pc35,
    }


def register(pointcloud, target):
    pointcloud.estimate_normals()
    target.estimate_normals()

    threshold = .06
    trans_init = np.identity(4)

    print("Apply point-to-plane ICP")
    reg_p2l = o3d.pipelines.registration.registration_icp(
        pointcloud, target, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(
            relative_fitness=0.000001, relative_rmse=0.000001, max_iteration=1000))
    
    print("Fitness:", reg_p2l.fitness)

    pointcloud = copy.deepcopy(pointcloud)
    pointcloud.transform(reg_p2l.transformation)

    return pointcloud


cam24 = 'azure_kinect2_4_calib_snap'
cam15 = 'azure_kinect1_5_calib_snap'
cam14 = 'azure_kinect1_4_calib_snap'
cam34 = 'azure_kinect3_4_calib_snap'
cam35 = 'azure_kinect3_5_calib_snap'
if __name__ == "__main__":
    cache = diskcache.Cache('../calibration/cache')

    subject_0 = get_subject('a1', 0, cache)

    # pc_s0 = subject_0[cam24] + \
    #     subject_0[cam15] + \
    #     subject_0[cam14] + \
    #     subject_0[cam34] + \
    #     subject_0[cam35]
    # show(pc_s0)


    # # Single
    pc15_trans = register(subject_0[cam15], subject_0[cam24]) # Good
    # subject_0[cam24].paint_uniform_color([0, 0, 1])
    # subject_0[cam15].paint_uniform_color([0, 1, 0])
    # pc15_trans.paint_uniform_color([0, 1, 0])
    # show(subject_0[cam15] + subject_0[cam24])
    # show(pc15_trans + subject_0[cam24])

    pc_15_24 = pc15_trans + subject_0[cam24]

    # Single
    pc15_trans = register(subject_0[cam15], subject_0[cam14]) # Good
    # subject_0[cam14].paint_uniform_color([0, 0, 1])
    # subject_0[cam15].paint_uniform_color([0, 1, 0])
    # pc15_trans.paint_uniform_color([0, 1, 0])
    # show(subject_0[cam15] + subject_0[cam14])
    # show(pc15_trans + subject_0[cam14])

    pc_15_14 = pc15_trans + subject_0[cam14]

    # Total
    pct_trans = register(pc_15_24, pc_15_14) # Good
    pc_15_14.paint_uniform_color([0, 0, 1])
    pc_15_24.paint_uniform_color([0, 1, 0])
    pct_trans.paint_uniform_color([0, 1, 0])
    show(pc_15_24 + pc_15_14)
    show(pct_trans + pc_15_14)
    
    # pc34_trans = register(subject_0[cam34], subject_0[cam14]) # Good
    # subject_0[cam14].paint_uniform_color([0, 0, 1])
    # subject_0[cam34].paint_uniform_color([0, 1, 0])
    # pc34_trans.paint_uniform_color([0, 1, 0])
    # show(subject_0[cam34] + subject_0[cam14])
    # show(pc34_trans + subject_0[cam14])

    # pc34_trans = register(subject_0[cam34], subject_0[cam35]) # Not Good
    # subject_0[cam35].paint_uniform_color([0, 0, 1])
    # subject_0[cam34].paint_uniform_color([0, 1, 0])
    # pc34_trans.paint_uniform_color([0, 1, 0])
    # show(subject_0[cam34] + subject_0[cam35])
    # show(pc34_trans + subject_0[cam35])

    # pc24_trans = register(subject_0[cam24], subject_0[cam35]) # Not Good
    # subject_0[cam35].paint_uniform_color([0, 0, 1])
    # subject_0[cam24].paint_uniform_color([0, 1, 0])
    # pc24_trans.paint_uniform_color([0, 1, 0])
    # show(subject_0[cam24] + subject_0[cam35])
    # show(pc24_trans + subject_0[cam35])
