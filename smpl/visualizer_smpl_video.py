from shlex import join
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


VIS_MESH = True
VIS_ORG = False
VIS_JTR = True

DIR_STORE = '/home/hamid/Documents/phd/footefield/Pose_to_SMPL/fit/output/HALPE/'
DIR_PARAMS = '../pose_estimation/keypoints_3d_pose2smpl/'
DIR_STORE_ORG = '../pose_estimation/keypoints_3d_ba'
DIR_OUTPUT = "./videos_smpl"
DIR_SMPL = 'params_smpl'

PARAM_OUTPUT_SIZE = (1920, 1080)
PARAM_OUTPUT_FPS = 5.0
PARAM_CALIB_SIZE = 16

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


def get_video_writer(experiment, camera):
    if not os.path.exists(DIR_OUTPUT):
        os.mkdir(DIR_OUTPUT)

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    writer = cv2.VideoWriter(
        os.path.join(
            DIR_OUTPUT,
            f'visualizer_skeleton_video_{experiment}_{camera}.avi'
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
    

def write_frame(img_rgb, poses_2d, type):
    point_size = get_point_size_by_type(type)
    for point in poses_2d:
        cv2.circle(img_rgb, (int(point[0]), int(point[1])),
                    point_size, (0, 255, 0), -1)

    connections = get_connections_by_type(type)
    if connections is not None:
        for connection in connections:
            cv2.line(img_rgb,
                    (int(poses_2d[connection[0]][0]),
                     int(poses_2d[connection[0]][1])),
                    (int(poses_2d[connection[1]][0]),
                     int(poses_2d[connection[1]][1])),
                    (255, 255, 255), 1)


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
    # poses_2d[:, 1] = poses_2d[:, 1]
    poses_2d = poses_2d.reshape(poses_shape)

    return poses_2d


def get_corresponding_files(path):
    file_name = path.split('/')[-1].split('.')[0]

    files = [
        (file_name + '_0_normalized_params.pkl', file_name + '_0_params.pkl'),
        (file_name + '_1_normalized_params.pkl', file_name + '_1_params.pkl'),
    ]

    return files


def load_smpl(experiment, subject):
    name = "params_smpl_{}_{}.pkl".format(
        experiment, subject
    )
    path_smpl = os.path.join(DIR_SMPL, name)
    with open(path_smpl, 'rb') as handle:
        smpl = pickle.load(handle)

    alphas = np.array(smpl['alphas'])
    betas = np.array(smpl['betas'])
    scale = np.array(smpl['scale'])
    translation = np.array(smpl['translation'])

    return alphas, betas, scale, translation


# TODO: Shorten
def get_smpl_parameters(experiment):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    smpl_layer = SMPL_Layer(
        center_idx=0,
        gender="neutral",
        model_root='models').to(device)
        
    joints_all = []
    verts_all = []
    faces_all = []
    for subject in [0, 1]:
        # Load SMPL data
        alphas, betas, scale_smpl, translation_smpl = \
            load_smpl(experiment, subject)
    
        alphas = torch.from_numpy(alphas).to(device)
        betas = torch.from_numpy(betas).to(device)
        batch_tensor = torch.ones((alphas.shape[0], 1)).to(device)

        verts, joints = smpl_layer(
            alphas, th_betas=betas * batch_tensor)
        
        verts = verts.detach().cpu().numpy()
        joints = joints.detach().cpu().numpy()
        faces = smpl_layer.th_faces.detach().cpu().numpy()

        # Load alignment params
        path_params = os.path.join(DIR_PARAMS, f"keypoints3d_{experiment}_ba_{subject}_params.pkl")
        with open(path_params, 'rb') as handle:
            params = pickle.load(handle)
        rotation = params['rotation']
        scale = params['scale']
        translation = params['translation']

        rotation_inverted = np.linalg.inv(rotation)

        joints = (joints + translation_smpl) * scale_smpl
        joints = joints.dot(rotation_inverted.T)
        joints = joints * scale
        joints = joints + translation

        verts = (verts + translation_smpl) * scale_smpl
        verts = verts.dot(rotation_inverted.T)
        verts = verts * scale
        verts = verts + translation

        joints_all.append(joints)
        verts_all.append(verts)
        faces_all.append(faces)

    joints_all = np.array(joints_all)
    verts_all = np.array(verts_all)
    faces_all = np.array(faces_all)

    joints_all = np.transpose(joints_all, (1, 0, 2, 3))
    verts_all = np.transpose(verts_all, (1, 0, 2, 3))

    return joints_all, verts_all, faces_all


# TODO: Move the cameras somewhere else
cam24 = 'azure_kinect2_4_calib_snap'
cam15 = 'azure_kinect1_5_calib_snap'
cam14 = 'azure_kinect1_4_calib_snap'
cam34 = 'azure_kinect3_4_calib_snap'
cam35 = 'azure_kinect3_5_calib_snap'

EXPERIMENTS = ['a1', 'a2']

# TODO: Too long
if __name__ == "__main__":
    cache = diskcache.Cache('../calibration/cache')

    cameras = [
        cam24,
        cam15,
        # cam14,
        cam34,
        cam35
    ]

    for experiment in EXPERIMENTS:
        print(f"Visualizing {experiment}")
        
        skeleton_path = os.path.join(DIR_STORE_ORG, f"keypoints3d_{experiment}_ba.pkl")
        with open(skeleton_path, 'rb') as handle:
            output = pickle.load(handle)

        poses = output['points_3d'].reshape(-1, 2, 26, 3)
        params = output['params']

        joints_all, verts_all, faces_all = get_smpl_parameters(experiment)

        for idx_cam, camera in enumerate(tqdm(cameras)):
            dir = data_loader.EXPERIMENTS[experiment][camera]

            img_rgb_paths = data_loader.list_rgb_images(
                os.path.join(dir, "color"))
            
            poses_2d = poses_3d_2_2d(
                poses,
                params[idx_cam]).reshape(poses.shape[0], -1, 2)
            poses_2d_smpl = poses_3d_2_2d(
                joints_all,
                params[idx_cam]).reshape(poses.shape[0], -1, 2)
            poses_2d_verts = poses_3d_2_2d(
                verts_all,
                params[idx_cam]).reshape(poses.shape[0], -1, 2)

            writer = get_video_writer(experiment, camera)
            for idx, t in enumerate(poses_2d.reshape(poses_2d.shape[0], -1, 2)):
                img_rgb = cv2.imread(img_rgb_paths[idx])
                mtx, dist, _ = get_parameters(params[idx_cam])
                img_rgb = cv2.undistort(img_rgb, mtx, dist, None, None)
                if VIS_ORG:
                    write_frame(img_rgb, poses_2d[idx],
                                TYPE_ORG)

                if VIS_JTR:
                    write_frame(img_rgb, poses_2d_smpl[idx],
                                TYPE_JTR)

                if VIS_MESH:
                    write_frame(img_rgb, poses_2d_verts[idx],
                                TYPE_MESH)
                    
                writer.write(img_rgb)
