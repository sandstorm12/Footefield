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


def visualize_poses(poses, joints, verts, faces, configs):
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window()
    vis.get_render_option().background_color = data_loader.COLOR_SPACE_GRAY
    vis.get_render_option().show_coordinate_frame = True

    geometry_poses = o3d.geometry.PointCloud()
    lines_poses = o3d.geometry.LineSet()
    geometry_joints = o3d.geometry.PointCloud()
    lines_joints = o3d.geometry.LineSet()
    mesh = o3d.geometry.TriangleMesh()
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    mesh_line = o3d.geometry.LineSet()
    for idx in range(len(verts)):
        if configs['visualize_poses']:
            keypoints = poses[idx].reshape(-1, 3)
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(keypoints)
            pcd.paint_uniform_color([0, 1, 0]) # Blue points

            connections = np.array(data_loader.HALPE_EDGES)
            
            lines_poses.points = o3d.utility.Vector3dVector(keypoints)
            lines_poses.lines = o3d.utility.Vector2iVector(connections)
            lines_poses.paint_uniform_color([1, 1, 1]) # White lines

            geometry_poses.points = pcd.points
            geometry_poses.colors = pcd.colors
            if idx == 0:
                vis.add_geometry(geometry_poses)
                vis.add_geometry(lines_poses)
            else:
                vis.update_geometry(geometry_poses)
                vis.update_geometry(lines_poses)

        if configs['visualize_joints']:
            keypoints = joints[idx].reshape(-1, 3)
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(keypoints)
            pcd.paint_uniform_color([0, 1, 0]) # Blue points

            connections = np.array(data_loader.SMPL_EDGES)
            
            lines_joints.points = o3d.utility.Vector3dVector(keypoints)
            lines_joints.lines = o3d.utility.Vector2iVector(connections)
            lines_joints.paint_uniform_color([1, 1, 1]) # White lines

            geometry_joints.points = pcd.points
            geometry_joints.colors = pcd.colors
            if idx == 0:
                vis.add_geometry(geometry_joints)
                vis.add_geometry(lines_joints)
            else:
                vis.update_geometry(geometry_joints)
                vis.update_geometry(lines_joints)

        if configs['visualize_mesh']:
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
            
        delay = .1
        control_gaps = 10
        for _ in range(control_gaps):
            vis.poll_events()
            vis.update_renderer()
            time.sleep(delay / control_gaps)


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

    with open(configs['skeletons_norm'], 'rb') as handle:
        poses = np.array(yaml.safe_load(handle))

    print(poses[0].keys())
    
    for idx_person, person in enumerate(params_smpl):
        alphas = torch.from_numpy(
            np.array(person['alphas'], np.float32)
        ).to(device)
        betas = torch.from_numpy(
            np.array(person['betas'], np.float32)
        ).to(device)
        translation = np.array(person['translation'], np.float32)    
        scale = person['scale']
        batch_tensor = torch.ones((alphas.shape[0], 1)).to(device)

        # scale = poses[idx_person]['scale']
        # translation = poses[idx_person]['translation']

        verts, joints = smpl_layer(alphas,
                                th_betas=betas * batch_tensor)

        verts = verts.detach().cpu().numpy().astype(float)
        verts = (verts + np.expand_dims(translation, axis=1)) * scale
        joints = joints.detach().cpu().numpy().astype(float)
        joints = (joints + np.expand_dims(translation, axis=1)) * scale
        faces = smpl_layer.th_faces.detach().cpu().numpy()

        poses_person = (np.array(poses[idx_person]['pose_normalized']))
        
        visualize_poses(poses_person, joints, verts, faces, configs)
