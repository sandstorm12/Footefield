import sys
sys.path.append('../')

import os
import cv2
import time
import yaml
import torch
import pickle
import argparse
import numpy as np
import open3d as o3d
import torch.nn.functional as F

from tqdm import tqdm
from utils import data_loader
from smplpytorch.pytorch.smpl_layer import SMPL_Layer


DIR_SKELETONS = '../pose_estimation/keypoints_3d_pose2smpl'
DIR_ORG = '../pose_estimation/keypoints_3d_ba'
DIR_OUTPUT = './params_smpl'
DIR_SMPL = 'params_smpl'
DIR_PARAMS = '../pose_estimation/keypoints_3d_pose2smpl/'

EXPERIMENTS = ['a1', 'a2']
SUBJECTS = [0, 1]
EPOCHS = 2000
VISUALIZE = True
VISUALIZE_MESH = True
PARAM_COEFF_POSE = 0 #1e-2
PARAM_LR = 2e-3
PATH_MODEL = 'models'

SMPL_SKELETON_MAP = np.array([ # (SMPL, SKELETON)
    [0, 19, 1.],
    [1, 11, 1.],
    [2, 12, 1.],
    [4, 13, 1.],
    [5, 14, 1.],
    [6, 26, .25],
    [7, 15, 1.],
    [8, 16, 1.],
    [10, 27, .25],
    [11, 28, .25],
    [12, 18, 1.],
    [16, 5, 1.],
    [17, 6, 1.],
    [15, 0, .25],
    [18, 7, 1.],
    [19, 8, 1.],
    [20, 9, 1.],
    [21, 10, 1.],
])


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


def calc_distance(joints, skeleton):
    skeleton_selected = skeleton[:, SMPL_SKELETON_MAP[:, 1]]
    output_selected = joints[:, SMPL_SKELETON_MAP[:, 0]]

    loss = F.smooth_l1_loss(output_selected, skeleton_selected, reduction='none')
    
    # Just for test, optimize
    loss = torch.mean(loss, dim=(0, 2))
    loss = torch.mean(loss * torch.from_numpy(SMPL_SKELETON_MAP[:, 2]).float().cuda())

    return loss


def calc_smoothness(joints):
    loss = torch.nn.functional.smooth_l1_loss(
        joints[1:, SMPL_SKELETON_MAP[:, 0]],
        joints[:-1, SMPL_SKELETON_MAP[:, 0]])

    return loss


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


def init_torch_params(skeletons, device):
    alphas = (
        torch.rand(
            skeletons.shape[0],
            72,
            dtype=torch.float32
        ) * .03).to(device)
    betas = (torch.rand(1, 10, dtype=torch.float32) * .03).to(device)
    translation = torch.zeros(3, dtype=torch.float32).to(device)
    scale = torch.ones(1, dtype=torch.float32).to(device)

    batch_tensor = torch.ones((skeletons.shape[0], 1)).to(device)

    alphas.requires_grad = True
    betas.requires_grad = False
    translation.requires_grad = True
    scale.requires_grad = True

    return alphas, betas, translation, scale, batch_tensor


def get_optimizer(alphas, betas, translation, scale):
    optim_params = [
        {'params': alphas, 'lr': PARAM_LR},
        {'params': betas, 'lr': PARAM_LR},
        {'params': scale, 'lr': PARAM_LR},
        {'params': translation, 'lr': PARAM_LR},]
    optimizer = torch.optim.Adam(optim_params)

    return optimizer


def optimize(smpl_layer, skeletons, configs):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    smpl_layer.to(device)

    alphas, betas, translation, scale, batch_tensor = \
        init_torch_params(skeletons, device)
    optimizer = get_optimizer(alphas, betas, translation, scale)
    skeletons_torch = torch.from_numpy(skeletons).float().to(device)

    loss_init = None
    bar = tqdm(range(configs['epochs']))
    for _ in bar:
        _, joints = model(
            alphas,
            th_betas=betas * batch_tensor)
        joints = joints * scale + translation

        loss = calc_distance(joints, skeletons_torch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if loss_init is None:
            loss_init = loss.detach().cpu().item()
        bar.set_description(
            "L: {:.2E} S:{:.2f}".format(
                loss.detach().cpu().item(),
                scale.item(),
            )
        )

    print('Loss went from {:.4f} to {:.4f}'.format(loss_init, loss))

    return alphas.detach().cpu().numpy(), \
        betas.detach().cpu().numpy(), \
        translation.detach().cpu().numpy(), \
        scale.detach().cpu().numpy(), \


# TODO: Shorten
def visualize_poses(alphas, betas,
                    translation, scale,
                    faces, skeletons):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    alphas = torch.from_numpy(alphas).to(device)
    betas = torch.from_numpy(betas).to(device)
    batch_tensor = torch.ones((skeletons.shape[0], 1)).to(device)

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window()
    vis.get_render_option().background_color = data_loader.COLOR_SPACE_GRAY
    vis.get_render_option().show_coordinate_frame = True

    verts, joints = model(
        alphas,
        th_betas=betas * batch_tensor)
    
    verts = verts.detach().cpu().numpy().squeeze()
    joints = joints.detach().cpu().numpy().squeeze()
    faces = faces.detach().cpu().numpy().squeeze()

    joints = ((joints + translation) * scale).squeeze()
    verts = ((verts + translation) * scale).squeeze()

    pcd_skeleton_org = o3d.geometry.PointCloud()
    pcd_joints = o3d.geometry.PointCloud()
    mesh = o3d.geometry.TriangleMesh()
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    mesh_line = o3d.geometry.LineSet()
    for idx in range(len(skeletons)):
        pcd_skeleton_org.points = o3d.utility.Vector3dVector(skeletons[idx])
        pcd_skeleton_org.paint_uniform_color([1, 1, 1])
        if idx == 0:
            vis.add_geometry(pcd_skeleton_org)
        else:
            vis.update_geometry(pcd_skeleton_org)

        pcd_joints.points = o3d.utility.Vector3dVector(joints[idx])
        pcd_joints.paint_uniform_color([1, 0, 1])
        if idx == 0:
            vis.add_geometry(pcd_joints)
        else:
            vis.update_geometry(pcd_joints)

        if VISUALIZE_MESH:
            mesh.vertices = o3d.utility.Vector3dVector(
                verts[idx])
            mesh_line_temp = o3d.geometry.LineSet.create_from_triangle_mesh(
                mesh)
            mesh_line.points = mesh_line_temp.points
            mesh_line.lines = mesh_line_temp.lines
            if idx == 0:
                vis.add_geometry(mesh_line)
            else:
                vis.update_geometry(mesh_line)
            
        delay_ms = 100
        for _ in range(delay_ms // 10):
            vis.poll_events()
            vis.update_renderer()
            time.sleep(.01)


def get_params_color(expriment):
    file = f"keypoints3d_{expriment}_ba.pkl"
    file_path = os.path.join(DIR_ORG, file)
    with open(file_path, 'rb') as handle:
        output = pickle.load(handle)

    params = output['params']

    return params


def load_skeletons(experiment, subject):
    path = os.path.join(DIR_SKELETONS,
                        f'keypoints3d_{experiment}_ba_{subject}_normalized.npy')
    skeletons = np.load(path)

    return skeletons


def store_smpl_parameters(alphas, betas,
        translation, scale, experiment, subject):
    if not os.path.exists(DIR_OUTPUT):
        os.mkdir(DIR_OUTPUT)

    path = os.path.join(
        DIR_OUTPUT,
        f'params_smpl_{experiment}_{subject}.pkl')
    
    params = {
        'alphas': alphas,
        'betas': betas,
        'translation': translation,
        'scale': scale,
    }

    with open(path, 'wb') as handle:
        pickle.dump(params, handle,
                    protocol=pickle.HIGHEST_PROTOCOL)
        
    print(f'Stored results: {path}')


def _store_artifacts(artifact, output):
    with open(output, 'w') as handle:
        yaml.dump(artifact, handle)


if __name__ == '__main__':
    args = _get_arguments()
    configs = _load_configs(args.config)

    print(f"Config loaded: {configs}")

    model = SMPL_Layer(
        center_idx=0,
        gender=configs['gender'],
        model_root=configs['models_root'])
    
    with open(configs['skeletons'], 'rb') as handle:
        bundles = yaml.safe_load(handle)

    params_smpl = []
    for bundle in bundles:
        poses = np.array(bundle['pose_normalized'])

        alphas, betas, translation, scale = optimize(
            model, poses, configs)
        
        # Do we need to add a rotation parameter as well?
        params_smpl_person = {
            'alphas': alphas.tolist(),
            'betas': betas.tolist(),
            'translation': translation.tolist(),
            'scale': scale.item(),
        }
        params_smpl.append(params_smpl_person)

        if configs['visualize']:
            visualize_poses(
                alphas, betas,
                translation, scale, model.th_faces, poses)

    _store_artifacts(params_smpl, configs['output'])
