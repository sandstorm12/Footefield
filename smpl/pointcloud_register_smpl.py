import os
import sys
sys.path.append('../')

import cv2
import glob
import pickle
import diskcache
import numpy as np
import open3d as o3d

from utils import data_loader


DIR_OUTPUT = './extrinsics_global'
DIR_ORG = '../pose_estimation/keypoints_3d_ba'
DIR_PARAMS_FINETUNED = "./extrinsics_finetuned"
DIR_PARAMS_TRANSFORM = '../pose_estimation/keypoints_3d_pose2smpl/'
DIR_SMPL = '/home/hamid/Documents/phd/footefield/Pose_to_SMPL/fit/output/HALPE/'


# TODO: Refactor
def get_depth_image(cam_name, experiment, idx):
    cam_num = cam_name[12:15]
    img_depth = '/home/hamid/Documents/phd/footefield/data/AzureKinectRecord_0729/{}/azure_kinect{}/depth/depth{:05d}.png'.format(experiment, cam_num, idx)

    return img_depth


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
        nb_neighbors=16,
        std_ratio=.05)
    pointcloud = pointcloud.select_by_index(ind)
    
    return pointcloud


def preprocess(pointcloud, voxel_size):
    pointcloud = pointcloud.voxel_down_sample(voxel_size=voxel_size)
    
    pointcloud = remove_outliers(pointcloud)

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
    
    return transformation_icp


def visualize_poses(verts, experiment, extrinsics, voxel_size, cache):
    pcd_mesh = o3d.geometry.PointCloud()
    for idx in range(0, len(verts[0]), 25):
        pcd = get_pcd(cam24, experiment, idx, extrinsics[cam24], cache)
        pcd += get_pcd(cam15, experiment, idx, extrinsics[cam15], cache)
        # pcd14 = get_pcd(cam14, experiment, idx, extrinsics[?], params)
        pcd += get_pcd(cam34, experiment, idx, extrinsics[cam34], cache)
        pcd += get_pcd(cam35, experiment, idx, extrinsics[cam35], cache)
        pcd_combined = preprocess(pcd, voxel_size=voxel_size)
        pcd_combined.paint_uniform_color([1, 1, 1])
        
        pcd_mesh.points = o3d.utility.Vector3dVector(
            np.concatenate((verts[0][idx], verts[1][idx])))
        pcd_mesh.paint_uniform_color([0, 0, 1])

        transformation_icp = pairwise_registration(
            pcd_mesh, pcd_combined, voxel_size=voxel_size)
        
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
        vis.register_key_callback(83, key_callback)
        vis.get_render_option().show_coordinate_frame = True
        vis.get_render_option().background_color = data_loader.COLOR_SPACE_GRAY
        pcd_mesh.transform(transformation_icp)
        vis.add_geometry(pcd_mesh)
        vis.add_geometry(pcd_combined)
        vis.run()


def load_finetuned_extrinsics():
    extrinsics_finetuned = {}
    for path in glob.glob(os.path.join(DIR_PARAMS_FINETUNED, '*')):
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
    for file_smpl in files_smpl:
        # Load SMPL data
        path_smpl = os.path.join(DIR_SMPL, file_smpl[0])
        with open(path_smpl, 'rb') as handle:
            smpl = pickle.load(handle)
        verts = np.array(smpl['verts'])
        scale_smpl = smpl['scale']
        translation_smpl = smpl['translation']

        # Load alignment params
        path_params = os.path.join(DIR_PARAMS_TRANSFORM, file_smpl[1])
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

    return verts_all


EXPERIMENT = 'a1'
VOXEL_SIZE = .025


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

    for file in sorted(os.listdir(DIR_ORG), reverse=False):
        experiment = file.split('.')[0].split('_')[1]
        if experiment != EXPERIMENT:
            continue

        file_path = os.path.join(DIR_ORG, file)
        print(f"Visualizing {file_path}")

        verts_all = load_smpl(file_path)

        visualize_poses(
            verts_all,
            experiment,
            finetuned_extrinsics[experiment],
            VOXEL_SIZE,
            cache)
