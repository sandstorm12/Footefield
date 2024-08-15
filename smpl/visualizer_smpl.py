from re import sub
import sys
sys.path.append('../')

import time
import yaml
import torch
import argparse
import numpy as np
import open3d as o3d

from utils import data_loader
from smplpytorch.pytorch.smpl_layer import SMPL_Layer


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


def visualize_poses(verts, faces):
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window()
    vis.get_render_option().background_color = data_loader.COLOR_SPACE_GRAY
    vis.get_render_option().show_coordinate_frame = True

    geometry_combined = o3d.geometry.PointCloud()
    mesh = o3d.geometry.TriangleMesh()
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    mesh_line = o3d.geometry.LineSet()
    for idx in range(len(verts)):
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


if __name__ == '__main__':
    args = _get_arguments()
    configs = _load_configs(args.config)

    print(f"Config loaded: {configs}")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    smpl_layer = SMPL_Layer(
        center_idx=0,
        gender="neutral",
        model_root='models').to(device)

    with open(configs['params_smpl'], 'rb') as handle:
        params_smpl = yaml.safe_load(handle)
    
    for person in params_smpl:
        alphas = torch.from_numpy(
            np.array(person['alphas'], np.float32)
        ).to(device)
        betas = torch.from_numpy(
            np.array(person['betas'], np.float32)
        ).to(device)
        batch_tensor = torch.ones((alphas.shape[0], 1)).to(device)

        verts, Jtr = smpl_layer(alphas,
                                th_betas=betas * batch_tensor)

        verts = verts.detach().cpu().numpy().astype(float)

        faces = smpl_layer.th_faces.detach().cpu().numpy()
        visualize_poses(verts, faces)
