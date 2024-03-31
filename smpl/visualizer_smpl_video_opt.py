import sys
sys.path.append('../')

import os
import cv2
import torch
import pickle
import diskcache
import numpy as np

from tqdm import tqdm
from utils import data_loader
from smplpytorch.pytorch.smpl_layer import SMPL_Layer


DIR_STORE = '/home/hamid/Documents/phd/footefield/Pose_to_SMPL/fit/output/HALPE'
DIR_STORE_OPT = './params_smpl'
DIR_PARAMS = '../pose_estimation/keypoints_3d_pose2smpl/'
DIR_STORE_ORG = '../pose_estimation/keypoints_3d_ba'
DIR_OUTPUT = "./output_videos_opt"

PARAM_OUTPUT_SIZE = (1920, 1080)
PARAM_OUTPUT_FPS = 5.0

TYPE_ORG = "org"
TYPE_JTR = "jtr"
TYPE_MESH = "mesh"


def get_parameters(params):
    mtx = params['mtx']
    dist = params['dist']
    rotation = params['rotation']
    translation = params['translation']

    extrinsics = np.zeros((3, 4), dtype=float)
    extrinsics[:3, :3] = rotation
    extrinsics[:3, 3] = translation

    return mtx, dist, extrinsics


# Implemented by Gemini
def project_3d_to_2d(camera_matrix, dist_coeffs, rvec, tvec, object_points):
    image_points, _ = cv2.projectPoints(object_points, rvec, tvec,
                                        camera_matrix, dist_coeffs)

    image_points = image_points.squeeze()

    return image_points


def get_video_writer(experiment, camera, type):
    if not os.path.exists(DIR_OUTPUT):
        os.mkdir(DIR_OUTPUT)

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    writer = cv2.VideoWriter(
        os.path.join(
            DIR_OUTPUT,
            f'visualizer_skeleton_video_{experiment}_{camera}_{type}.avi'
        ),
        fourcc,
        PARAM_OUTPUT_FPS,
        PARAM_OUTPUT_SIZE
    )
    
    return writer


# TODO: Make types constant
def get_connections_by_type(type):
    if type == TYPE_ORG:
        connections = np.concatenate(
            (np.array(data_loader.HALPE_EDGES),
             np.array(data_loader.HALPE_EDGES) + 26))
    elif type == TYPE_JTR:
        connections = np.concatenate(
            (np.array(data_loader.SMPL_EDGES),
             np.array(data_loader.SMPL_EDGES) + 24))
    elif type == TYPE_MESH:
        connections = None
    else:
        raise Exception("Unknown type.")
        
    return connections


def get_point_size_by_type(type):
    if type == TYPE_ORG:
        point_size = 3
    elif type == TYPE_JTR:
        point_size = 3
    elif type == TYPE_MESH:
        point_size = 1
    else:
        raise Exception("Unknown type.")
        
    return point_size
    

def write_video(poses_2d, experiment, camera, type, params, cache):
    img_rgb_paths = data_loader.list_rgb_images(os.path.join(dir, "color"))

    mtx, dist, _ = get_parameters(params)

    writer = get_video_writer(experiment, camera, type)
    for idx, t in enumerate(poses_2d.reshape(poses_2d.shape[0], -1, 2)):
        img_rgb = cv2.imread(img_rgb_paths[idx])

        img_rgb = cv2.undistort(img_rgb, mtx, dist, None, None)

        point_size = get_point_size_by_type(type)
        for point in t:
            cv2.circle(img_rgb, (int(point[0]), int(point[1])),
                       point_size, (0, 255, 0), -1)

        connections = get_connections_by_type(type)
        if connections is not None:
            for connection in connections:
                cv2.line(img_rgb,
                        (int(t[connection[0]][0]), int(t[connection[0]][1])),
                        (int(t[connection[1]][0]), int(t[connection[1]][1])),
                        (255, 255, 255), 1)

        writer.write(img_rgb)


def poses_3d_2_2d(poses_3d, params):
    poses_shape = list(poses_3d.shape)
    poses_shape[-1] = 2
    
    mtx = params['mtx']
    dist = params['dist']
    rotation = params['rotation']
    translation = params['translation']
    poses_2d = project_3d_to_2d(
        mtx, None,
        rotation,
        translation,
        poses_3d.reshape(-1, 3))
    poses_2d[:, 1] = poses_2d[:, 1]
    poses_2d = poses_2d.reshape(poses_shape)

    return poses_2d


def get_corresponding_files(path, experiment):
    file_name = path.split('/')[-1].split('.')[0]

    files = [
        (
            file_name + '_0_normalized_params.pkl',
            file_name + '_0_params.pkl',
            f'params_smpl_{experiment}_1.pkl',
        ),(
            file_name + '_1_normalized_params.pkl',
            file_name + '_1_params.pkl',
            f'params_smpl_{experiment}_0.pkl'
        ),
    ]

    return files


# TODO: Refactor
def get_smpl_parameters(smpl_layer, file_org):
    experiment = file.split('.')[-2].split('_')[-2]
    files_smpl = get_corresponding_files(file_org, experiment)
        
    verts_all = []
    faces_all = []
    for file_smpl in files_smpl:
        # Load SMPL data
        path_smpl = os.path.join(DIR_STORE, file_smpl[0])
        with open(path_smpl, 'rb') as handle:
            smpl = pickle.load(handle)
        scale_smpl = smpl['scale']
        transformation = smpl['transformation']

        # Load SMPL params and get verts
        path_smpl_opt = os.path.join(DIR_STORE_OPT, file_smpl[2])
        with open(path_smpl_opt, 'rb') as handle:
            smpl_params = pickle.load(handle)
        pose_params = smpl_params['alpha']
        shape_params = np.tile(smpl_params['beta'], (pose_params.shape[0], 1))
        faces = smpl_params['faces']
        verts = []
        for idx in tqdm(range(pose_params.shape[0])):
            pose_torch = torch.from_numpy(
                pose_params[idx]).unsqueeze(0).float()
            shape_torch = torch.from_numpy(
                shape_params[idx]).unsqueeze(0).float()

            verts_single, _ = smpl_layer(pose_torch, th_betas=shape_torch)

            verts.append(verts_single.detach().cpu().numpy().astype(float))
        verts = np.array(verts).squeeze()

        # Load alignment params
        path_params = os.path.join(DIR_PARAMS, file_smpl[1])
        with open(path_params, 'rb') as handle:
            params = pickle.load(handle)
        rotation = params['rotation']
        scale = params['scale'] * scale_smpl
        translation = params['translation']

        rotation_inverted = np.linalg.inv(rotation)
        verts = np.concatenate(
            (verts,
                np.ones((verts.shape[0], verts.shape[1], 1))
            ), axis=2)
        verts = np.matmul(verts, transformation)
        verts = verts[:, :, :3] / verts[:, :, -1:]
        verts = verts.dot(rotation_inverted.T)
        verts = verts * scale
        verts = verts + translation

        verts_all.append(verts)
        faces_all.append(faces)

    verts_all = np.array(verts_all).squeeze()
    faces_all = np.array(faces_all)

    verts_all = np.transpose(verts_all, (1, 0, 2, 3))

    return verts_all, faces_all


# TODO: Move the cameras somewhere else
cam24 = 'azure_kinect2_4_calib_snap'
cam15 = 'azure_kinect1_5_calib_snap'
cam14 = 'azure_kinect1_4_calib_snap'
cam34 = 'azure_kinect3_4_calib_snap'
cam35 = 'azure_kinect3_5_calib_snap'

# TODO: Too long
if __name__ == "__main__":
    cache = diskcache.Cache('../calibration/cache')

    smpl_layer = SMPL_Layer(
        center_idx=0,
        gender="neutral",
        model_root='smplpytorch/native/models')

    cameras = [
        cam24,
        cam15,
        # cam14,
        cam34,
        cam35
    ]

    for file in os.listdir(DIR_STORE_ORG):
        experiment = file.split('.')[-2].split('_')[-2]

        file_path = os.path.join(DIR_STORE_ORG, file)
        print(f"Visualizing {file_path}")
        
        with open(file_path, 'rb') as handle:
            output = pickle.load(handle)

        params = output['params']

        verts_all, faces_all = get_smpl_parameters(smpl_layer, file_path)

        for idx_cam, camera in enumerate(tqdm(cameras)):
            dir = data_loader.EXPERIMENTS[experiment][camera]

            poses_2d_verts = poses_3d_2_2d(
                verts_all,
                params[idx_cam])
            
            write_video(poses_2d_verts, experiment, camera,
                        TYPE_MESH, params[idx_cam], cache)
