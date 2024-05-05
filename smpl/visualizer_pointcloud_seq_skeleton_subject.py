import os
import sys
sys.path.append('../')

import cv2
import time
import glob
import pickle
import diskcache
import numpy as np
import open3d as o3d

from utils import data_loader
from sklearn.cluster import KMeans


VIS_MESH = True

STORE_DIR = '../pose_estimation/keypoints_3d_ba'
PARAM_CALIB_SIZE = 16
DIR_PARMAS_GLOBAL = "./extrinsics_global"
DIR_STORE = '/home/hamid/Documents/phd/footefield/Pose_to_SMPL/fit/output/HALPE/'
DIR_PARAMS = '../pose_estimation/keypoints_3d_pose2smpl/'

HALPE_LINES = np.array(
    [(0, 1), (0, 2), (1, 3), (2, 4), (5, 18), (6, 18), (5, 7),
     (7, 9), (6, 8), (8, 10), (17, 18), (18, 19), (19, 11),
     (19, 12), (11, 13), (12, 14), (13, 15), (14, 16), (20, 24),
     (21, 25), (23, 25), (22, 24), (15, 24), (16, 25)])


def get_cam(cam_name):
    return f'azure_kinect{cam_name}_calib_snap'


# TODO: Refactor
def get_depth_image(cam_name, experiment, idx):
    cam_num = cam_name[12:15]
    img_depth = '/home/hamid/Documents/phd/footefield/data/AzureKinectRecord_0729/{}/azure_kinect{}/depth/depth{:05d}.png'.format(experiment, cam_num, idx)

    return img_depth


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


def get_extrinsics(cam, cache):
    R = cache['extrinsics'][cam24 + 'infrared']['rotation']
    T = cache['extrinsics'][cam24 + 'infrared']['transition']
    R2 = cache['extrinsics'][cam15 + 'infrared']['rotation']
    T2 = cache['extrinsics'][cam15 + 'infrared']['transition']
    R3 = cache['extrinsics'][cam14 + 'infrared']['rotation']
    T3 = cache['extrinsics'][cam14 + 'infrared']['transition']
    R4 = cache['extrinsics'][cam34 + 'infrared']['rotation']
    T4 = cache['extrinsics'][cam34 + 'infrared']['transition']
    
    extrinsics = np.identity(4, dtype=float)
    if cam == cam24:
        r = np.array([[1, 0, 0],
                     [0, 1, 0],
                     [0, 0, 1]])
        t = np.array([0, 0, 0])
        extrinsics[:3, :3] = r
        extrinsics[:3, 3] = t.reshape(3)
    elif cam == cam15:
        extrinsics[:3, :3] = R
        extrinsics[:3, 3] = T.reshape(3)
    elif cam == cam14:
        R2_com = np.dot(R2, R)
        T2_com = (np.dot(R2, T).reshape(3, 1) + T2).reshape(3,)
        extrinsics[:3, :3] = R2_com
        extrinsics[:3, 3] = T2_com
    elif cam == cam34:
        R2_com = np.dot(R2, R)
        T2_com = (np.dot(R2, T).reshape(3, 1) + T2).reshape(3,)
        R3_com = np.dot(R3, R2_com)
        T3_com = (np.dot(R3, T2_com).reshape(3, 1) + T3).reshape(3,)
        extrinsics[:3, :3] = R3_com
        extrinsics[:3, 3] = T3_com
    elif cam == cam35:
        R2_com = np.dot(R2, R)
        T2_com = (np.dot(R2, T).reshape(3, 1) + T2).reshape(3,)
        R3_com = np.dot(R3, R2_com)
        T3_com = (np.dot(R3, T2_com).reshape(3, 1) + T3).reshape(3,)
        R4_com = np.dot(R4, R3_com)
        T4_com = (np.dot(R4, T3_com).reshape(3, 1) + T4).reshape(3,)
        extrinsics[:3, :3] = R4_com
        extrinsics[:3, 3] = T4_com

    return extrinsics


def get_params_depth(cam, cache):
    mtx = cache['depth_matching'][cam]['mtx_r']
    dist = cache['depth_matching'][cam]['dist_r']
    R = cache['depth_matching'][cam]['rotation']
    T = cache['depth_matching'][cam]['transition']

    extrinsics = np.identity(4, dtype=float)
    extrinsics[:3, :3] = R
    extrinsics[:3, 3] = T.ravel() / 1000

    return mtx, dist, extrinsics


def get_pcd(subject, cam, experiment, idx, extrinsics, cache):
    img_depth = get_depth_image(cam, experiment, idx)
    mtx, dist, _ = get_params_depth(cam, cache)

    img_depth = cv2.imread(img_depth, -1)
    mapx, mapy = cv2.initUndistortRectifyMap(
        mtx, dist, None, mtx, (640, 576), cv2.CV_32FC2)
    img_depth = cv2.remap(img_depth, mapx, mapy, cv2.INTER_NEAREST)
    depth = o3d.geometry.Image(img_depth)

    intrinsics = o3d.camera.PinholeCameraIntrinsic(
        640, 576, mtx[0, 0], mtx[1, 1], mtx[0, 2], mtx[1, 2])
    pcd = o3d.geometry.PointCloud.create_from_depth_image(
        depth, intrinsics, extrinsics['extrinsics'][cam]['base'])
    pcd = pcd.transform(extrinsics['extrinsics'][cam]['offset'])

    start_pts = np.array([[0, 0, 0], [1, -1, 3]])
    pcd_np = np.asarray(pcd.points)
    kmeans = KMeans(n_clusters=2, random_state=47,
                    init=start_pts, n_init=1).fit(pcd_np)
    # TODO: Explain what (subject + 1 % 2) is
    pcd.points = o3d.utility.Vector3dVector(
        pcd_np[kmeans.labels_ == (subject + 1) % 2])

    return pcd


def remove_outliers(pointcloud):
    _, ind = pointcloud.remove_statistical_outlier(
        nb_neighbors=16,
        std_ratio=.05)
    pointcloud = pointcloud.select_by_index(ind)
    
    return pointcloud


def preprocess(pointcloud):
    pointcloud = remove_outliers(pointcloud)

    # pointcloud = pointcloud.voxel_down_sample(voxel_size=0.005)

    return pointcloud

# TODO: Shorten
def visualize_poses(poses, verts, faces, subject, experiment,
                    extrinsics, cache):
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window()
    vis.get_render_option().background_color = data_loader.COLOR_SPACE_GRAY
    vis.get_render_option().show_coordinate_frame = True
    
    geometry_combined = o3d.geometry.PointCloud()
    lines = o3d.geometry.LineSet()
    mesh = [o3d.geometry.TriangleMesh() for i in range(len(verts))]
    for i in range(len(verts)):
        mesh[i].triangles = o3d.utility.Vector3iVector(faces[i])
    mesh_line = [o3d.geometry.LineSet() for i in range(len(verts))]
    for idx in range(len(poses)):
        pcd = get_pcd(subject, cam24, experiment, idx, extrinsics, cache)
        pcd += get_pcd(subject, cam15, experiment, idx, extrinsics, cache)
        # pcd14 = get_pcd(subject, cam14, experiment, idx, extrinsics, cache) 
        pcd += get_pcd(subject, cam34, experiment, idx, extrinsics, cache)
        pcd += get_pcd(subject, cam35, experiment, idx, extrinsics, cache)

        pcd_combined = preprocess(pcd)
        pcd = pcd_combined.transform(extrinsics['global'])

        geometry_combined.points = pcd_combined.points
        if idx == 0:
            vis.add_geometry(geometry_combined)
        else:
            vis.update_geometry(geometry_combined)

        if VIS_MESH:
            idx_mesh = (subject + 1) % 2
            print(verts[idx_mesh][idx].shape, verts[idx_mesh][idx].dtype)
            mesh[idx_mesh].vertices = o3d.utility.Vector3dVector(
                verts[idx_mesh][idx])
            print(verts[idx_mesh][idx])
            mesh_line_temp = o3d.geometry.LineSet.create_from_triangle_mesh(
                mesh[idx_mesh])
            mesh_line[idx_mesh].points = mesh_line_temp.points
            mesh_line[idx_mesh].lines = mesh_line_temp.lines
            if idx == 0:
                vis.add_geometry(mesh_line[idx_mesh])
            else:
                vis.update_geometry(mesh_line[idx_mesh])
            
        delay_ms = 100
        for _ in range(delay_ms // 10):
            vis.poll_events()
            vis.update_renderer()
            time.sleep(.01)

        print(f"Update {idx}: {time.time()}")


def load_global_extrinsics():
    extrinsics_global = {}
    for path in glob.glob(os.path.join(DIR_PARMAS_GLOBAL, '*')):
        experiment = path.split('.')[-2].split('_')[-2]
        subject = path.split('.')[-2].split('_')[-1]
        with open(path, 'rb') as handle:
            params = pickle.load(handle)

        extrinsics_global[experiment + '_' + subject] = params

    return extrinsics_global


def get_corresponding_files(path):
    file_name = path.split('/')[-1].split('.')[0]

    files = [
        (file_name + '_0_normalized_params.pkl', file_name + '_0_params.pkl'),
        (file_name + '_1_normalized_params.pkl', file_name + '_1_params.pkl'),
    ]

    return files


def load_smpl(file_org):
    files_smpl = get_corresponding_files(file_org)

    verts_all = []
    faces_all = []
    for file_smpl in files_smpl:
        # Load SMPL data
        path_smpl = os.path.join(DIR_STORE, file_smpl[0])
        with open(path_smpl, 'rb') as handle:
            smpl = pickle.load(handle)
        verts = np.array(smpl['verts'])
        faces = np.array(smpl['th_faces'])
        scale_smpl = smpl['scale']
        translation_smpl = smpl['translation']

        # Load alignment params
        path_params = os.path.join(DIR_PARAMS, file_smpl[1])
        with open(path_params, 'rb') as handle:
            params = pickle.load(handle)
        rotation = params['rotation']
        scale = params['scale'] * scale_smpl
        translation = params['translation']

        rotation_inverted = np.linalg.inv(rotation)
        verts = verts + translation_smpl
        verts = verts.dot(rotation_inverted.T)
        verts = verts * scale
        verts = verts + translation
        verts = verts / 1000

        verts_all.append(verts)
        faces_all.append(faces)

    return verts_all, faces_all


SUBJECT = 0


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

    extrinsics_global = load_global_extrinsics()

    for file in sorted(os.listdir(STORE_DIR), reverse=False):
        experiment = file.split('.')[0].split('_')[1]
        file_path = os.path.join(STORE_DIR, file)
        print(f"Visualizing {file_path}")
    
        with open(file_path, 'rb') as handle:
            output = pickle.load(handle)

        poses = output['points_3d'].reshape(-1, 52, 3)
        params = output['params']

        verts_all, faces_all = load_smpl(file_path)

        visualize_poses(
            poses, verts_all, faces_all, SUBJECT, 
            experiment, extrinsics_global[experiment + '_' + str(SUBJECT)],
            cache)
