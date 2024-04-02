import sys
sys.path.append('../')

import os
import time
import torch
import pickle
import numpy as np
import open3d as o3d

from tqdm import tqdm
from utils import data_loader
from pytorch3d.loss import chamfer_distance
from smplpytorch.pytorch.smpl_layer import SMPL_Layer


DIR_SMPL = '/home/hamid/Documents/phd/footefield/Pose_to_SMPL/fit/output/HALPE/'
DIR_POINTCLOUD = './pointcloud_normalized'
DIR_OUTPUT = './params_smpl'

EXPERIMENTS = ['a1', 'a2']
SUBJECTS = [0, 1]
EPOCHS = 50
VISUALIZE = True
PARAM_COEFF_POSE = .0
PARAM_COEFF_DET = .001


# TODO: Shorten
def optimize_beta(smpl_layer, pose_params, shape_params, pcds, epochs):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    smpl_layer.to(device)

    pose_torch = torch.from_numpy(
        pose_params).float().to(device)
    shape_torch = (torch.rand(10) * 0.03).to(device)
    transformation = torch.eye(4).to(device)
    
    pose_torch.requires_grad = True
    shape_torch.requires_grad = True
    transformation.requires_grad = True

    lr = 2e-3
    optim_params = [{'params': shape_torch, 'lr': lr},
                    {'params': pose_torch, 'lr': lr * 0},
                    {'params': transformation, 'lr': lr * 10},]
    optimizer = torch.optim.Adam(optim_params)

    # TODO: maybe add transfromation term as well
    loss_init = None
    bar = tqdm(range(epochs))
    for _ in bar:
        loss_chamfer = 0
        loss_smooth = 0
        loss_det = 0
        for idx in range(pose_params.shape[0]):
            pcd_torch = torch.from_numpy(
                pcds[idx]).unsqueeze(0).float().to(device)
            
            pcd_torch = torch.cat((
                pcd_torch,
                torch.ones(
                    pcd_torch.shape[0], pcd_torch.shape[1], 1,
                    device=device)), dim=2)
            pcd_torch = (torch.matmul(pcd_torch, transformation))
            pcd_torch = pcd_torch[:, :, :3] / pcd_torch[:, :, -1:]

            verts, _ = smpl_layer(
                pose_torch[idx].unsqueeze(0),
                th_betas=shape_torch.unsqueeze(0))

            loss_chamfer += chamfer_distance(verts, pcd_torch)[0]

            pose_prev = pose_torch[max(0, idx - 1)]
            pose_next = pose_torch[min(pose_params.shape[0] - 1, idx + 1)]
            loss_smooth += torch.sum(
                torch.abs(
                    (pose_torch[idx] - pose_prev) +
                    (pose_torch[idx] - pose_next))
            ) / pose_torch.shape[0]

            loss_det += torch.abs(1 - torch.det(transformation))

        loss = loss_chamfer + \
            loss_smooth * PARAM_COEFF_POSE + \
            loss_det * PARAM_COEFF_DET

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss = loss.detach().cpu().item()
        if loss_init is None:
            loss_init = loss

        bar.set_description(
            "L: {:.4f} C: {:.4f} S: {:.4f} D: {:.4f} Dv:{:.4f}".format(
            loss,
            loss_chamfer,
            loss_smooth * PARAM_COEFF_POSE,
            loss_det * PARAM_COEFF_DET,
            torch.det(transformation))
        )

    print('Loss went from {:.4f} to {:.4f}'.format(loss_init, loss))

    return pose_torch.detach().cpu().numpy(), \
        shape_torch.detach().cpu().numpy(), \
        transformation.detach().cpu().numpy()


# TODO: Shorten
def visualize_poses(alpha, beta, transformation, faces, pcds):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    pose_torch = torch.from_numpy(alpha).to(device)
    shape_torch = torch.from_numpy(beta).to(device)
    verts = smpl_layer(
        pose_torch,
        th_betas=shape_torch.repeat(pose_torch.shape[0], 1)
        )[0].detach().cpu().numpy()

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window()
    vis.get_render_option().background_color = data_loader.COLOR_SPACE_GRAY
    vis.get_render_option().show_coordinate_frame = True

    geometry_combined = o3d.geometry.PointCloud()
    mesh = o3d.geometry.TriangleMesh()
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    mesh_line = o3d.geometry.LineSet()
    for idx in range(len(verts)):
        pcd_combined = pcds[idx]

        pcd_combined = np.concatenate(
            (pcd_combined,
                np.ones((pcd_combined.shape[0], 1))
            ), axis=1)
        pcd_combined = np.matmul(pcd_combined, transformation)
        pcd_combined = pcd_combined[:, :3] / pcd_combined[:, -1:]

        geometry_combined.points = o3d.utility.Vector3dVector(pcd_combined)
        if idx == 0:
            vis.add_geometry(geometry_combined)
        else:
            vis.update_geometry(geometry_combined)

        mesh.vertices = o3d.utility.Vector3dVector(
            verts[idx].squeeze())
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


def get_corresponding_files(path, subject):
    file_name = path.split('/')[-1].split('.')[0]

    idx = (subject + 1) % 2

    files = (file_name + f'_{idx}_normalized_params.pkl',
             file_name + f'_{idx}_params.pkl')

    return files


def load_smpl(experiment, subject):
    path = f'keypoints3d_{experiment}_ba'
    file_smpl = get_corresponding_files(path, subject)
    
    # Load SMPL data
    path_smpl = os.path.join(DIR_SMPL, file_smpl[0])
    with open(path_smpl, 'rb') as handle:
        smpl = pickle.load(handle)
    pose_params = np.array(smpl['pose_params'])
    shape_params = np.array(smpl['shape_params'])
    faces = np.array(smpl['th_faces'])

    return pose_params, shape_params, faces


def load_pointcloud(experiment, subject):
    path = os.path.join(DIR_POINTCLOUD,
                        f'pointcloud_{experiment}_{subject}.pkl')
    with open(path, 'rb') as handle:
        pcds = pickle.load(handle)

    return pcds


def store_smpl_parameters(alpha, beta, faces, experiment, subject):
    if not os.path.exists(DIR_OUTPUT):
        os.mkdir(DIR_OUTPUT)

    path = os.path.join(
        DIR_OUTPUT,
        f'params_smpl_{experiment}_{subject}.pkl')
    
    params = {
        'alpha': alpha,
        'beta': beta,
        'faces': faces,
    }

    with open(path, 'wb') as handle:
        pickle.dump(params, handle,
                    protocol=pickle.HIGHEST_PROTOCOL)
        
    print(f'Stored results: {path}')


if __name__ == '__main__':
    smpl_layer = SMPL_Layer(
        center_idx=0,
        gender="neutral",
        model_root='smplpytorch/native/models')

    for experiment in EXPERIMENTS:
        for subject in SUBJECTS:
            pose_params, shape_params, faces = \
                load_smpl(experiment, subject)

            pcds = load_pointcloud(experiment, subject)

            print(f'Optimizing {experiment} {subject}')
            alpha, beta, transformation = optimize_beta(
                smpl_layer, pose_params, shape_params, pcds, EPOCHS)

            if VISUALIZE:
                visualize_poses(alpha, beta, transformation, faces, pcds)

            store_smpl_parameters(alpha, beta, faces, experiment, subject)
