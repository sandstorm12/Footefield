import sys
sys.path.append('../')

import os
import time
import torch
import pickle
import numpy as np
import open3d as o3d
import torch.nn.functional as F

from tqdm import tqdm
from utils import data_loader
from smplpytorch.pytorch.smpl_layer import SMPL_Layer


DIR_SKELETONS = '../pose_estimation/keypoints_3d_pose2smpl'
DIR_OUTPUT = './params_smpl'

EXPERIMENTS = ['a1', 'a2']
SUBJECTS = [0, 1]
EPOCHS = 1000
VISUALIZE = False
VISUALIZE_MESH = True
PARAM_COEFF_POSE = .01
PATH_MODEL = 'models'

SMPL_SKELETON_MAP = np.array([ # (SMPL, SKELETON)
    [0, 19],
    [1, 11],
    [2, 12],
    [4, 13],
    [5, 14],
    [6, 26],
    [7, 15],
    [8, 16],
    [10, 20],
    [11, 21],
    [12, 18],
    [13, 5],
    [14, 6],
    [15, 0],
    [18, 7],
    [19, 8],
    [20, 9],
    [21, 10],
])


def calc_distance(joints, skeleton):
    skeleton_selected = skeleton[:, SMPL_SKELETON_MAP[:, 1]]
    output_selected = joints[:, SMPL_SKELETON_MAP[:, 0]]

    loss = F.smooth_l1_loss(output_selected, skeleton_selected, reduction='mean')

    return loss


def calc_smoothness(joints):
    loss = torch.nn.functional.mse_loss(
        joints[1:, SMPL_SKELETON_MAP[:, 0]],
        joints[:-1, SMPL_SKELETON_MAP[:, 0]])

    return loss


# TODO: Shorten
def optimize_beta(smpl_layer, skeletons, epochs):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    smpl_layer.to(device)

    alphas = (torch.rand(skeletons.shape[0], 72, dtype=torch.float32) * .03).to(device)
    betas = (torch.rand(1, 10, dtype=torch.float32) * .03).to(device)
    translation = torch.zeros(3, dtype=torch.float32).to(device)
    scale = torch.ones(1, dtype=torch.float32).to(device)

    batch_tensor = torch.ones((skeletons.shape[0], 1)).to(device)

    alphas.requires_grad = True
    betas.requires_grad = False
    translation.requires_grad = True
    scale.requires_grad = True

    lr = 2e-3
    optim_params = [
        {'params': alphas, 'lr': lr},
        {'params': betas, 'lr': lr * 0},
        {'params': scale, 'lr': lr},
        {'params': translation, 'lr': lr},]
    optimizer = torch.optim.Adam(optim_params)

    skeletons_torch = torch.from_numpy(skeletons).float().to(device)

    # TODO: maybe add transfromation term as well
    loss_init = None
    bar = tqdm(range(epochs))
    for _ in bar:
        verts, joints = model(
            alphas,
            th_betas=betas * batch_tensor)

        joints = joints * scale + translation

        loss_distance = calc_distance(joints, skeletons_torch)

        loss_smooth = calc_smoothness(joints)

        loss = loss_distance + loss_smooth * PARAM_COEFF_POSE

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss = loss.detach().cpu().item()
        if loss_init is None:
            loss_init = loss

        bar.set_description(
            "L: {:.4f} D: {:.4f} Si:{:.2f}".format(
                loss,
                loss_distance,
                loss_smooth * PARAM_COEFF_POSE,
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


def load_skeletons(experiment, subject):
    path = os.path.join(DIR_SKELETONS,
                        f'keypoints3d_{experiment}_ba_{subject}_normalized.npy')
    skeletons = np.load(path)

    return skeletons


def store_smplx_parameters(alphas, betas,
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


if __name__ == '__main__':
    model = SMPL_Layer(
        center_idx=0,
        gender='neutral',
        model_root='models')

    for experiment in EXPERIMENTS:
        for subject in SUBJECTS:
            print(f'Optimizing {experiment} {subject}')

            skeletons = load_skeletons(experiment, subject)

            alphas, betas, translation, scale = optimize_beta(
                model, skeletons, EPOCHS)

            if VISUALIZE:
                visualize_poses(
                    alphas, betas,
                    translation, scale, model.th_faces, skeletons)

            store_smplx_parameters(
                alphas, betas,
                translation, scale,
                experiment, subject)
