import sys
sys.path.append('../')

import time
import yaml
import torch
import smplx
import argparse
import numpy as np
import open3d as o3d

from utils import data_loader


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


# TODO: Shorten
def visualize_poses(global_orient, jaw_pose, leye_pose,
                    reye_pose, body, left_hand_pose,
                    right_hand_pose, betas, expression,
                    translation, scale, faces):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    global_orient = torch.from_numpy(global_orient).to(device)
    jaw_pose = torch.from_numpy(jaw_pose).to(device)
    leye_pose = torch.from_numpy(leye_pose).to(device)
    reye_pose = torch.from_numpy(reye_pose).to(device)
    body = torch.from_numpy(body).to(device)
    left_hand_pose = torch.from_numpy(left_hand_pose).to(device)
    right_hand_pose = torch.from_numpy(right_hand_pose).to(device)
    betas = torch.from_numpy(betas).to(device)
    expression = torch.from_numpy(expression).to(device)

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window()
    vis.get_render_option().background_color = data_loader.COLOR_SPACE_GRAY
    vis.get_render_option().show_coordinate_frame = True

    output = model(
        global_orient=global_orient,
        jaw_pose=jaw_pose,
        leye_pose=leye_pose,
        reye_pose=reye_pose,
        body_pose=body,
        left_hand_pose=left_hand_pose,
        right_hand_pose=right_hand_pose,
        betas=betas,
        expression=expression,
        return_verts=True)
    
    verts = output.vertices.detach().cpu().numpy().squeeze()
    joints = output.joints.detach().cpu().numpy().squeeze()

    verts = verts - joints[0, 0] + translation
    verts = verts * scale
    verts = verts.squeeze()

    geometry_combined = o3d.geometry.PointCloud()
    mesh = o3d.geometry.TriangleMesh()
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    mesh_line = o3d.geometry.LineSet()
    for idx in range(len(verts)):
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


def load_smplx_params(smplx_params):
    global_orient = np.array(smplx_params['global_orient'], np.float32)
    jaw_pose = np.array(smplx_params['jaw_pose'], np.float32)
    leye_pose = np.array(smplx_params['leye_pose'], np.float32)
    reye_pose = np.array(smplx_params['reye_pose'], np.float32)
    body = np.array(smplx_params['body'], np.float32)
    left_hand_pose = np.array(smplx_params['left_hand_pose'], np.float32)
    right_hand_pose = np.array(smplx_params['right_hand_pose'], np.float32)
    betas = np.array(smplx_params['betas'], np.float32)
    expression = np.array(smplx_params['expression'], np.float32)
    translation = np.array(smplx_params['translation'], np.float32)
    scale = np.array(smplx_params['scale'], np.float32)

    return global_orient, jaw_pose, leye_pose, reye_pose, body, \
        left_hand_pose, right_hand_pose, betas, expression, \
        translation, scale


if __name__ == '__main__':
    args = _get_arguments()
    configs = _load_configs(args.config)

    print(f"Config loaded: {configs}")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = smplx.create(
        configs['models_root'], model_type='smplx',
        gender=configs['gender'], use_face_contour=False,
        num_betas=10, use_pca=False,
        num_expression_coeffs=10,
        ext='npz').to(device)
    
    with open(configs['params_smplx'], 'rb') as handle:
        params_smpl = yaml.safe_load(handle)

    for person in params_smpl:
        global_orient, jaw_pose, leye_pose, reye_pose, body, \
            left_hand_pose, right_hand_pose, betas, expression, \
            translation, scale = \
                load_smplx_params(person)

        visualize_poses(
            global_orient, jaw_pose, leye_pose,
            reye_pose,body,left_hand_pose,
            right_hand_pose, betas, expression,
            translation, scale, model.faces)
