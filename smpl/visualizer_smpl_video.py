import sys
sys.path.append('../')

import os
import cv2
import yaml
import torch
import argparse
import numpy as np

from tqdm import tqdm
from utils import data_loader
from smplpytorch.pytorch.smpl_layer import SMPL_Layer


TYPE_ORG = "org"
TYPE_JTR = "jtr"
TYPE_MESH = "mesh"


def _get_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-c', '--config',
        help='Path to the config file',
        type=str,
        required=True,
    )

    args = parser.parse_args()

    return args


def _load_configs(path):
    with open(path, 'r') as yaml_file:
        configs = yaml.safe_load(yaml_file)

    return configs


def get_parameters(params):
    mtx = np.array(params['mtx'], np.float32)
    dist = np.array(params['dist'], np.float32)

    return mtx, dist


# Implemented by Gemini
def project_3d_to_2d(camera_matrix, dist_coeffs, rvec, tvec, object_points):
    image_points, _ = cv2.projectPoints(object_points, rvec, tvec,
                                        camera_matrix, dist_coeffs)

    image_points = image_points.squeeze()

    return image_points


def get_video_writer(camera, configs):
    if not os.path.exists(configs['output']):
        os.makedirs(configs['output'])

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    writer = cv2.VideoWriter(
        os.path.join(
            configs['output'],
            f'visualizer_skeleton_video_{camera}.avi'
        ),
        fourcc,
        configs['fps'],
        configs['size'],
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
    
    mtx = np.array(params['mtx'], np.float64)
    # dist = np.array(params['dist'], np.float64)
    rotation = np.array(params['rotation'], np.float64)
    translation = np.array(params['translation'], np.float64)

    poses_2d = project_3d_to_2d(
        mtx, None,
        rotation,
        translation,
        poses_3d.reshape(-1, 3))
    poses_2d = poses_2d.reshape(poses_shape)

    return poses_2d


def get_corresponding_files(path):
    file_name = path.split('/')[-1].split('.')[0]

    files = [
        (file_name + '_0_normalized_params.pkl', file_name + '_0_params.pkl'),
        (file_name + '_1_normalized_params.pkl', file_name + '_1_params.pkl'),
    ]

    return files


def load_smpl(subject, configs):
    with open(configs['params_smpl'], 'rb') as handle:
        params_smpl = yaml.safe_load(handle)

    alphas = np.array(params_smpl[subject]['alphas'], np.float32)
    betas = np.array(params_smpl[subject]['betas'], np.float32)
    translation = np.array(params_smpl[subject]['translation'], np.float32)    
    scale = params_smpl[subject]['scale']
    
    return alphas, betas, scale, translation


# TODO: Shorten
def get_smpl_parameters(configs):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    smpl_layer = SMPL_Layer(
        center_idx=0,
        gender=configs['gender'],
        model_root=configs['models_root']).to(device)
        
    joints_all = []
    verts_all = []
    faces_all = []
    for subject in [0, 1]:
        alphas, betas, scale_smpl, translation_smpl = \
            load_smpl(subject, configs)
    
        alphas = torch.from_numpy(alphas).to(device)
        betas = torch.from_numpy(betas).to(device)
        batch_tensor = torch.ones((alphas.shape[0], 1)).to(device)

        verts, joints = smpl_layer(
            alphas, th_betas=betas * batch_tensor)
        
        verts = verts.detach().cpu().numpy()
        joints = joints.detach().cpu().numpy()
        faces = smpl_layer.th_faces.detach().cpu().numpy()

        # Load alignment params
        with open(configs['skeletons_norm'], 'rb') as handle:
            params = yaml.safe_load(handle)

        rotation = params[subject]['rotation']
        scale = params[subject]['scale']
        translation = params[subject]['translation']

        rotation_inverted = np.linalg.inv(rotation)

        joints = (joints + np.expand_dims(translation_smpl, axis=1)) * scale_smpl
        joints = joints.dot(rotation_inverted.T)
        joints = joints * scale
        joints = joints + translation

        verts = (verts + np.expand_dims(translation_smpl, axis=1)) * scale_smpl
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


if __name__ == "__main__":
    args = _get_arguments()
    configs = _load_configs(args.config)

    print(f"Config loaded: {configs}")

    joints_all, verts_all, faces_all = get_smpl_parameters(configs)

    with open(configs['skeletons'], 'rb') as handle:
        poses = np.array(yaml.safe_load(handle))

    with open(configs['params'], 'rb') as handle:
        params = yaml.safe_load(handle)

    cameras = configs['images'].keys()
    for idx_cam, camera in enumerate(tqdm(cameras)):
        dir = configs['images'][camera]

        img_rgb_paths = data_loader.list_rgb_images(dir)
        
        poses_2d = poses_3d_2_2d(
            poses,
            params[camera]).reshape(poses.shape[0], -1, 2)
        poses_2d_smpl = poses_3d_2_2d(
            joints_all,
            params[camera]).reshape(joints_all.shape[0], -1, 2)
        poses_2d_verts = poses_3d_2_2d(
            verts_all,
            params[camera]).reshape(verts_all.shape[0], -1, 2)

        writer = get_video_writer(camera, configs)
        for idx in range(joints_all.shape[0]):
            img_rgb = cv2.imread(img_rgb_paths[idx])
            mtx, dist = get_parameters(params[camera])
            img_rgb = cv2.undistort(img_rgb, mtx, dist, None, None)
            if configs['visualize_skeleton']:
                write_frame(img_rgb, poses_2d[idx],
                            TYPE_ORG)

            if configs['visualize_joints']:
                write_frame(img_rgb, poses_2d_smpl[idx],
                            TYPE_JTR)

            if configs['visualize_mesh']:
                write_frame(img_rgb, poses_2d_verts[idx],
                            TYPE_MESH)
                
            writer.write(img_rgb)
