import os
import sys
sys.path.append('../')

import cv2
import time
import copy
import glob
import pickle
import diskcache
import numpy as np
import open3d as o3d

from utils import data_loader


VIS_MESH = True

STORE_DIR = '../pose_estimation/keypoints_3d_ba'
PARAM_CALIB_SIZE = 16
DIR_PARMAS_FINETUNED = "./extrinsics_finetuned"
DIR_STORE = '/home/hamid/Documents/phd/footefield/Pose_to_SMPL/fit/output/HALPE/'
DIR_PARAMS = '../pose_estimation/keypoints_3d_pose2smpl/'
DIR_OUTPUT = './extrinsics_global'

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


def get_pcd(cam, experiment, idx, extrinsics, cache):
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
        depth, intrinsics, extrinsics['base'])
    pcd = pcd.transform(extrinsics['offset'])

    return pcd


def remove_outliers(pointcloud):
    _, ind = pointcloud.remove_statistical_outlier(
        nb_neighbors=5,
        std_ratio=.02)
    pointcloud = pointcloud.select_by_index(ind)
    
    return pointcloud


def preprocess(pointcloud):
    pointcloud = remove_outliers(pointcloud)

    pointcloud = pointcloud.voxel_down_sample(voxel_size=0.005)

    return pointcloud


def pairwise_registration(source, target, voxel_size):
    max_correspondence_distance_coarse = voxel_size * 15
    max_correspondence_distance_fine = voxel_size * 1.5

    source.estimate_normals()
    target.estimate_normals()

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


def visualize_poses(poses, verts, faces, experiment, extrinsics, cache):
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window()
    vis.get_render_option().background_color = data_loader.COLOR_SPACE_GRAY
    vis.get_render_option().show_coordinate_frame = True

    geometry_combined = o3d.geometry.PointCloud()
    mesh = [o3d.geometry.TriangleMesh() for i in range(len(verts))]
    pcd_mesh_0 = o3d.geometry.PointCloud()
    pcd_mesh_1 = o3d.geometry.PointCloud()
    pcd_mesh = o3d.geometry.PointCloud()
    for i in range(len(verts)):
        mesh[i].triangles = o3d.utility.Vector3iVector(faces[i])
    mesh_line = [o3d.geometry.LineSet() for i in range(len(verts))]
    for idx in range(0, len(poses), 25):
        pcd = get_pcd(cam24, experiment, idx, extrinsics[cam24], cache)
        pcd += get_pcd(cam15, experiment, idx, extrinsics[cam15], cache)
        # # pcd14 = get_pcd(cam14, experiment, idx, extrinsics[?], params)
        pcd += get_pcd(cam34, experiment, idx, extrinsics[cam34], cache)
        pcd += get_pcd(cam35, experiment, idx, extrinsics[cam35], cache)

        pcd_combined = preprocess(pcd)

        geometry_combined.points = pcd_combined.points
        geometry_combined.paint_uniform_color([1, 0, 0])
        pcd_mesh_0.points = o3d.utility.Vector3dVector(verts[0][idx])
        pcd_mesh_1.points = o3d.utility.Vector3dVector(verts[1][idx])
        pcd_mesh.points = (pcd_mesh_0 + pcd_mesh_1).points
        pcd_mesh.paint_uniform_color([0, 1, 0])

        transformation_icp, _ = pairwise_registration(pcd_mesh, geometry_combined, voxel_size=0.005)
        
        def key_callback(vis):
            store_extrinsics()
            print('Saved extrinsics...')
            sys.exit()

        def store_extrinsics():
            if not os.path.exists(DIR_OUTPUT):
                os.mkdir(DIR_OUTPUT)

            path = os.path.join(DIR_OUTPUT,
                                f'extinsics_global_{experiment}.pkl')
            
            params = {
                'extrinsics': extrinsics,
                'global': transformation_icp,
            }

            with open(path, 'wb') as handle:
                pickle.dump(params, handle,
                            protocol=pickle.HIGHEST_PROTOCOL)


        vis = o3d.visualization.VisualizerWithKeyCallback()
        vis.create_window(visible=True)    
        # Call only after creating visualizer window.
        vis.register_key_callback(83, key_callback)
        vis.get_render_option().show_coordinate_frame = True
        vis.get_render_option().background_color = data_loader.COLOR_SPACE_GRAY
        pcd_down_copy = copy.deepcopy(pcd_mesh)
        pcd_down_copy.transform(transformation_icp)
        vis.add_geometry(pcd_down_copy)
        vis.add_geometry(geometry_combined)
        vis.run()


def load_finetuned_extrinsics():
    extrinsics_finetuned = {}
    for path in glob.glob(os.path.join(DIR_PARMAS_FINETUNED, '*')):
        experiment = path.split('.')[-2].split('_')[-1]
        with open(path, 'rb') as handle:
            params = pickle.load(handle)

        extrinsics_finetuned[experiment] = params

    return extrinsics_finetuned


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

    finetuned_extrinsics = load_finetuned_extrinsics()

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
            poses, verts_all, faces_all,
            experiment, finetuned_extrinsics[experiment], cache)
