import sys
sys.path.append('../')

import os
import time
import torch
import smplx
import pickle
import numpy as np
import open3d as o3d

from tqdm import tqdm
from utils import data_loader


# DIR_POINTCLOUD = './pointcloud_normalized'
DIR_OPTIMIZED = './params_smplx'
PATH_MODEL = 'models'


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

    print(verts.shape, joints.shape)

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


def load_smplx(experiment, subject):
    path = f'params_smplx_{experiment}_{subject}.pkl'
    
    # Load SMPL data
    path_smplx = os.path.join(DIR_OPTIMIZED, path)
    with open(path_smplx, 'rb') as handle:
        smplx_params = pickle.load(handle)

    global_orient = smplx_params['global_orient']
    jaw_pose = smplx_params['jaw_pose']
    leye_pose = smplx_params['leye_pose']
    reye_pose = smplx_params['reye_pose']
    body = smplx_params['body']
    left_hand_pose = smplx_params['left_hand_pose']
    right_hand_pose = smplx_params['right_hand_pose']
    betas = smplx_params['betas']
    expression = smplx_params['expression']
    translation = smplx_params['translation']
    scale = smplx_params['scale']

    return global_orient, jaw_pose, leye_pose, reye_pose, body, \
        left_hand_pose, right_hand_pose, betas, expression, \
        translation, scale


SUBJECT = 0
EXPERIMENT = 'a1'
OPTIMIZED = False

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = smplx.create(
        PATH_MODEL, model_type='smplx',
        gender='neutral', use_face_contour=False,
        num_betas=10, use_pca=False,
        num_expression_coeffs=10,
        ext='npz').to(device)

    global_orient, jaw_pose, leye_pose, reye_pose, body, \
        left_hand_pose, right_hand_pose, betas, expression, \
        translation, scale = \
            load_smplx(EXPERIMENT, SUBJECT)

    visualize_poses(
        global_orient, jaw_pose, leye_pose,
        reye_pose,body,left_hand_pose,
        right_hand_pose, betas, expression,
        translation, scale, model.faces)
