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
from smplpytorch.pytorch.smpl_layer import SMPL_Layer


DIR_SMPL = '/home/hamid/Documents/phd/footefield/Pose_to_SMPL/fit/output/HALPE/'
DIR_POINTCLOUD = './pointcloud_normalized'


def visualize_poses(verts, faces, pcds):
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

        print(f"Update {idx}: {time.time()}")


def get_corresponding_files(path, subject):
    file_name = path.split('/')[-1].split('.')[0]

    idx = (subject + 1) % 2

    files = (file_name + f'_{idx}_normalized_params.pkl',
             file_name + f'_{idx}_params.pkl')

    return files


def load_smpl(file_org, subject):
    file_smpl = get_corresponding_files(file_org, subject)
    
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


SUBJECT = 0
EXPERIMENT = 'a1'

if __name__ == '__main__':
    smpl_layer = SMPL_Layer(
        center_idx=0,
        gender="neutral",
        model_root='smplpytorch/native/models')

    pose_params, shape_params, faces = \
        load_smpl('keypoints3d_a1_ba', SUBJECT)
    
    verts_all = []
    for idx in tqdm(range(pose_params.shape[0])):
        pose_torch = torch.from_numpy(
            pose_params[idx]).unsqueeze(0).float()
        shape_torch = torch.from_numpy(
            shape_params[idx]).unsqueeze(0).float()

        verts, Jtr = smpl_layer(pose_torch, th_betas=shape_torch)

        verts_all.append(verts.detach().cpu().numpy().astype(float))

    pcds = load_pointcloud(EXPERIMENT, SUBJECT)

    visualize_poses(verts_all, faces, pcds)
