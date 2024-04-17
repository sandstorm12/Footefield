import sys
sys.path.append('../')

import os
import time
import smplx
import torch
import pickle
import numpy as np
import open3d as o3d
import torch.nn.functional as F

from tqdm import tqdm
from utils import data_loader


DIR_SKELETONS = '../pose_estimation/keypoints_3d_pose2smpl_x'
DIR_OUTPUT = './params_smplx'

EXPERIMENTS = ['a1', 'a2']
SUBJECTS = [0, 1]
EPOCHS = 500
VISUALIZE = False
PARAM_COEFF_POSE = .1
PARAM_COEFF_DET = .01
PATH_MODEL = 'models'

COEFF_NORM = 1
COEFF_MINI = 1
SMPLX_SKELETON_MAP = np.array([ # (SMPLX, SKELETON)
    [13, 5, COEFF_NORM],
    [14, 6, COEFF_NORM],
    [1, 11, COEFF_NORM],
    [2, 12, COEFF_NORM],
    [4, 13, COEFF_NORM],
    [5, 14, COEFF_NORM],
    [7, 15, COEFF_NORM],
    [8, 16, COEFF_NORM], # Body
    [19, 8, COEFF_NORM],
    [21, 10, COEFF_NORM],
    [18, 7, COEFF_NORM],
    [20, 9, COEFF_NORM], # Arms
    [58, 24, COEFF_NORM], # Right ear
    [59, 38, COEFF_NORM], # Left ear
    [86, 50, COEFF_MINI],
    [87, 51, COEFF_MINI],
    [88, 52, COEFF_MINI],
    [89, 53, COEFF_MINI], # Nose
    [63, 20, COEFF_MINI],
    [64, 21, COEFF_MINI],
    [65, 22, COEFF_MINI], # Right foot
    [60, 17, COEFF_MINI],
    [61, 18, COEFF_MINI],
    [62, 19, COEFF_MINI], # Left foot
    [107, 71, COEFF_MINI],
    [108, 72, COEFF_MINI],
    [109, 73, COEFF_MINI],
    [110, 74, COEFF_MINI],
    [111, 75, COEFF_MINI],
    [112, 76, COEFF_MINI],
    [113, 77, COEFF_MINI],
    [114, 78, COEFF_MINI],
    [115, 79, COEFF_MINI],
    [116, 80, COEFF_MINI],
    [117, 81, COEFF_MINI],
    [118, 82, COEFF_MINI], # Outer lip
    [21, 112, COEFF_MINI],
    [52, 113, COEFF_MINI],
    [53, 114, COEFF_MINI],
    [54, 115, COEFF_MINI],
    [71, 116, COEFF_MINI], # Right hand
    [40, 117, COEFF_MINI],
    [41, 118, COEFF_MINI],
    [42, 119, COEFF_MINI],
    [72, 120, COEFF_MINI], # Right hand
    [43, 121, COEFF_MINI],
    [44, 122, COEFF_MINI],
    [45, 123, COEFF_MINI],
    [73, 124, COEFF_MINI], # Right hand
    [49, 125, COEFF_MINI],
    [50, 126, COEFF_MINI],
    [51, 127, COEFF_MINI],
    [74, 128, COEFF_MINI], # Right hand
    [46, 129, COEFF_MINI],
    [47, 130, COEFF_MINI],
    [48, 131, COEFF_MINI],
    [75, 132, COEFF_MINI], # Right hand
    [20, 91, COEFF_MINI],
    [37, 92, COEFF_MINI],
    [38, 93, COEFF_MINI],
    [39, 94, COEFF_MINI],
    [66, 95, COEFF_MINI], # Left hand
    [25, 96, COEFF_MINI],
    [26, 97, COEFF_MINI],
    [27, 98, COEFF_MINI],
    [67, 99, COEFF_MINI], # Left hand
    [28, 100, COEFF_MINI],
    [29, 101, COEFF_MINI],
    [30, 102, COEFF_MINI],
    [68, 103, COEFF_MINI], # Left hand
    [34, 104, COEFF_MINI],
    [35, 105, COEFF_MINI],
    [36, 106, COEFF_MINI],
    [69, 107, COEFF_MINI], # Left hand
    [31, 108, COEFF_MINI],
    [32, 109, COEFF_MINI],
    [33, 110, COEFF_MINI],
    [70, 111, COEFF_MINI], # Left hand
    [95, 59, COEFF_MINI],
    [96, 60, COEFF_MINI],
    [97, 61, COEFF_MINI],
    [98, 62, COEFF_MINI],
    [99, 63, COEFF_MINI],
    [100, 64, COEFF_MINI], # Right eye
    [101, 65, COEFF_MINI],
    [102, 66, COEFF_MINI],
    [103, 67, COEFF_MINI],
    [104, 68, COEFF_MINI],
    [105, 69, COEFF_MINI],
    [106, 70, COEFF_MINI], # Right eye
])


def calc_distance(joints, skeleton, SMPLX_SKELETON_MAP_torch):
    skeleton_selected = skeleton[:, SMPLX_SKELETON_MAP[:, 1]]
    output_selected = joints[:, SMPLX_SKELETON_MAP[:, 0]]

    loss = F.smooth_l1_loss(output_selected, skeleton_selected, reduction='mean')

    return loss


# TODO: Shorten
def optimize_beta(smpl_layer, skeletons, epochs):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    smpl_layer.to(device)

    global_orient = (torch.zeros([skeletons.shape[0], 1, 3], dtype=torch.float32)).to(device)
    jaw_pose = (torch.zeros([skeletons.shape[0], 1, 3], dtype=torch.float32)).to(device)
    leye_pose = (torch.zeros([skeletons.shape[0], 1, 3], dtype=torch.float32)).to(device)
    reye_pose = (torch.zeros([skeletons.shape[0], 1, 3], dtype=torch.float32)).to(device)
    body = (torch.rand([skeletons.shape[0], 21, 3], dtype=torch.float32) * .1).to(device)
    left_hand_pose = (torch.rand([skeletons.shape[0], 15, 3], dtype=torch.float32) * .1).to(device)
    right_hand_pose = (torch.rand([skeletons.shape[0], 15, 3], dtype=torch.float32) * .1).to(device)
    betas = (torch.zeros([1, 10], dtype=torch.float32)).to(device)
    expression = (torch.rand([skeletons.shape[0], 10], dtype=torch.float32) * .1).to(device)
    translation = torch.zeros(3).to(device)
    scale = torch.ones([1]).to(device)

    batch_tensor = torch.ones((skeletons.shape[0], 1)).to(device)

    global_orient.requires_grad = True
    jaw_pose.requires_grad = True
    leye_pose.requires_grad = True
    reye_pose.requires_grad = True
    body.requires_grad = True
    left_hand_pose.requires_grad = True
    right_hand_pose.requires_grad = True
    betas.requires_grad = True
    expression.requires_grad = True
    translation.requires_grad = True
    scale.requires_grad = True

    lr = 2e-2
    optim_params = [{'params': global_orient, 'lr': lr},
                    {'params': jaw_pose, 'lr': lr},
                    {'params': leye_pose, 'lr': lr},
                    {'params': reye_pose, 'lr': lr},
                    {'params': body, 'lr': lr},
                    {'params': left_hand_pose, 'lr': lr},
                    {'params': right_hand_pose, 'lr': lr},
                    {'params': betas, 'lr': lr},
                    {'params': expression, 'lr': lr},
                    {'params': scale, 'lr': lr},
                    {'params': translation, 'lr': lr},]
    optimizer = torch.optim.Adam(optim_params)

    SMPLX_SKELETON_MAP_torch = torch.from_numpy(SMPLX_SKELETON_MAP).to(device)
    SMPLX_SKELETON_MAP_torch.requires_grad = False

    skeletons_torch = torch.from_numpy(skeletons).float().to(device)

    # TODO: maybe add transfromation term as well
    loss_init = None
    bar = tqdm(range(epochs))
    for _ in bar:
        output = model(
            global_orient=global_orient,
            jaw_pose=jaw_pose,
            leye_pose=leye_pose,
            reye_pose=reye_pose,
            body_pose=body,
            left_hand_pose=left_hand_pose,
            right_hand_pose=right_hand_pose,
            betas=betas * batch_tensor,
            expression=expression,
            return_verts=True)

        joints = output.joints
        joints = joints - joints[0, 0] + translation
        joints = joints * scale

        loss_distance = calc_distance(joints, skeletons_torch, SMPLX_SKELETON_MAP_torch)

        loss_smooth = torch.nn.functional.mse_loss(joints[1:], joints[:-1])

        loss = loss_distance + loss_smooth * PARAM_COEFF_POSE

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss = loss.detach().cpu().item()
        if loss_init is None:
            loss_init = loss

        bar.set_description(
            "L: {:.4f} D: {:.4f} S: {:.4f} Si:{:.2f}".format(
                loss,
                loss_distance,
                loss_smooth * PARAM_COEFF_POSE,
                scale.item(),
            )
        )

    print('Loss went from {:.4f} to {:.4f}'.format(loss_init, loss))

    return global_orient.detach().cpu().numpy(), \
        jaw_pose.detach().cpu().numpy(), \
        leye_pose.detach().cpu().numpy(), \
        reye_pose.detach().cpu().numpy(), \
        body.detach().cpu().numpy(), \
        left_hand_pose.detach().cpu().numpy(), \
        right_hand_pose.detach().cpu().numpy(), \
        betas.detach().cpu().numpy(), \
        expression.detach().cpu().numpy(), \
        translation.detach().cpu().numpy(), \
        scale.detach().cpu().numpy(), \


# TODO: Shorten
def visualize_poses(global_orient, jaw_pose, leye_pose,
                    reye_pose, body, left_hand_pose,
                    right_hand_pose, betas, expression,
                    translation, scale,
                    faces, skeletons):
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
    for idx in range(len(skeletons)):
        pcd_combined = skeletons[idx]
        
        geometry_combined.points = o3d.utility.Vector3dVector(pcd_combined)
        geometry_combined.paint_uniform_color([1, 1, 1])
        if idx == 0:
            vis.add_geometry(geometry_combined)
        else:
            vis.update_geometry(geometry_combined)

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


def store_smplx_parameters(global_orient, jaw_pose, leye_pose,
        reye_pose, body, left_hand_pose, right_hand_pose, betas,
        expression, translation, scale, experiment, subject):
    if not os.path.exists(DIR_OUTPUT):
        os.mkdir(DIR_OUTPUT)

    path = os.path.join(
        DIR_OUTPUT,
        f'params_smplx_{experiment}_{subject}.pkl')
    
    params = {
        'global_orient': global_orient,
        'jaw_pose': jaw_pose,
        'leye_pose': leye_pose,
        'reye_pose': reye_pose,
        'body': body,
        'left_hand_pose': left_hand_pose,
        'right_hand_pose': right_hand_pose,
        'betas': betas,
        'expression': expression,
        'translation': translation,
        'scale': scale,
    }

    with open(path, 'wb') as handle:
        pickle.dump(params, handle,
                    protocol=pickle.HIGHEST_PROTOCOL)
        
    print(f'Stored results: {path}')


if __name__ == '__main__':
    model = smplx.create(
        PATH_MODEL, model_type='smplx',
        gender='neutral', use_face_contour=False,
        num_betas=10, use_pca=False,
        num_expression_coeffs=10,
        ext='npz')

    for experiment in EXPERIMENTS:
        for subject in SUBJECTS:
            skeletons = load_skeletons(experiment, subject)

            print(f'Optimizing {experiment} {subject}')
            global_orient, jaw_pose, leye_pose, \
                reye_pose,body,left_hand_pose, \
                right_hand_pose, betas, expression, \
                translation, scale = optimize_beta(
                    model, skeletons, EPOCHS)

            if VISUALIZE:
                visualize_poses(
                    global_orient, jaw_pose, leye_pose,
                    reye_pose,body,left_hand_pose,
                    right_hand_pose, betas, expression,
                    translation, scale, model.faces, skeletons)

            store_smplx_parameters(
                global_orient, jaw_pose, leye_pose,
                reye_pose,body,left_hand_pose,
                right_hand_pose, betas, expression,
                translation, scale,
                experiment, subject)
