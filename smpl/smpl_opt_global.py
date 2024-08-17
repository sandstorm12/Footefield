import sys
sys.path.append('../')

import os
import cv2
import time
import yaml
import torch
import argparse
import numpy as np
import open3d as o3d
import torch.nn.functional as F

from tqdm import tqdm
from utils import data_loader
from pytorch3d.loss import chamfer_distance
from smplpytorch.pytorch.smpl_layer import SMPL_Layer


SMPL_SKELETON_MAP = np.array([ # (SMPL, HALPE+AUG)
    [0, 19, 1.],
    [1, 11, 1.],
    [2, 12, 1.],
    [4, 13, 10.],
    [5, 14, 10.],
    [6, 26, .25],
    [7, 15, 10.],
    [8, 16, 10.],
    [10, 27, 1.],
    [11, 28, 1.],
    [12, 18, 1.],
    [16, 5, 1.],
    [17, 6, 1.],
    [15, 0, .25],
    [18, 7, 10.],
    [19, 8, 10.],
    [20, 9, 10.],
    [21, 10, 10.],
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


def project_points_to_camera_plane(points_3d, mtx, R, T):
    transformation = torch.eye(4).cuda()
    transformation[:3, :3] = R
    transformation[:3, 3] = T
    transformation = transformation.unsqueeze(0)

    points_3d = torch.cat((
        points_3d,
        torch.ones(
            points_3d.shape[0], points_3d.shape[1], 1,
            device="cuda")), dim=2)
    points_3d = (torch.bmm(transformation, points_3d.transpose(1, 2)))
    points_3d = points_3d[:, :3, :] / points_3d[:, 3:, :]    

    points_3d = torch.bmm(mtx, points_3d)

    points_3d = points_3d[:, :2, :] / points_3d[:, 2:, :]
    points_3d = points_3d.transpose(1, 2)

    return points_3d[:, :, :2]


def calc_distance(joints, skeleton, skeleton_weights):
    skeleton_selected = skeleton[:, SMPL_SKELETON_MAP[:, 1]]
    output_selected = joints[:, SMPL_SKELETON_MAP[:, 0]]

    loss = F.smooth_l1_loss(output_selected, skeleton_selected, reduction='none')
    
    # Just for test, optimize
    loss = torch.mean(loss, dim=(0, 2))
    loss = torch.mean(loss * skeleton_weights)

    return loss


def calc_chamfer(verts, masks, params, configs):
    loss = 0
    for cam in range(len(masks)):
        for idx_mask in range(len(masks[cam])):
            mask_torch = masks[cam][idx_mask]
            mtx = params[cam]['mtx']
            rotation = params[cam]['rotation']
            translation = params[cam]['translation']
            pcd_proj = project_points_to_camera_plane(
                verts[idx_mask].unsqueeze(0), mtx,
                rotation, translation,)
            if configs['visualize_chamfer_projection'] and cam == 4:
                visualize_chamfer(
                    mask_torch.squeeze().detach().cpu().numpy(),
                    pcd_proj.squeeze().detach().cpu().numpy())
            distances = chamfer_distance(
                pcd_proj, mask_torch,
                single_directional=True, norm=2,
                point_reduction=None, batch_reduction=None)[0]
            loss_verts = torch.mean(distances[0])
            # loss_mask = torch.mean(distances[1][distances[1] \
            #                                     < torch.max(distances[0])])
            loss += loss_verts # + loss_mask

    return loss


def visualize_chamfer(mask, vert):
    img = np.zeros((1080, 1920, 3), np.uint8)
    
    for point in mask:
        x = int(point[0])
        y = int(point[1])
        if 0 < x < 1920 and 0 < y < 1080:
            img[y, x] = (255, 255, 255)

    for point in vert:
        x = int(point[0])
        y = int(point[1])
        if 0 < x < 1920 and 0 < y < 1080:
            img[y, x] = (0, 255, 0)

    cv2.imshow("frame", cv2.resize(img, (960, 540)))
    cv2.waitKey(1)


def denormalize(verts, denormalize_params):
    rotation_inverted, scale, translation = denormalize_params

    verts = torch.matmul(verts, rotation_inverted)
    verts = verts * scale
    verts = verts + translation

    return verts


def load_denormalize_params(subject, device, configs):
    with open(configs['skeletons'], 'rb') as handle:
        params = yaml.safe_load(handle)

    rotation = np.array(params[subject]['rotation'])
    scale = np.array(params[subject]['scale'])
    translation = np.array(params[subject]['translation'])

    rotation_inverted = np.linalg.inv(rotation).T

    rotation_inverted = torch.from_numpy(rotation_inverted).float().to(device)
    scale = torch.from_numpy(scale).float().to(device)
    translation = torch.from_numpy(translation).float().to(device)

    return rotation_inverted, scale, translation


def masks_params_torch(masks, params):
    masks_torch = []
    params_torch = []
    for cam in masks.keys():
        masks_torch_cam = []
        for idx_mask in range(len(masks[cam])):
            masks_torch_cam.append(
                torch.from_numpy(
                    np.array(masks[cam][idx_mask])
                ).float().unsqueeze(0).cuda())
        masks_torch.append(masks_torch_cam)
        
        mtx = torch.from_numpy(
            np.array(params[cam]['mtx'])
        ).float().cuda().unsqueeze(0)
        rotation = torch.from_numpy(
            np.array(params[cam]['rotation'])
        ).float().cuda().unsqueeze(0)
        translation = torch.from_numpy(
            np.array(params[cam]['translation'])
        ).float().cuda().unsqueeze(0)
        params_torch.append(
            {
                'mtx': mtx,
                'rotation': rotation,
                'translation': translation,
            }
        )

    return masks_torch, params_torch


def optimize(smpl_layer, masks, skeletons, alphas, betas, scale, translation, params, subject, configs):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    smpl_layer.to(device)

    denormalize_params = load_denormalize_params(subject, device, configs)

    skeleton_weights = torch.from_numpy(SMPL_SKELETON_MAP[:, 2]).float().to(device)

    length = len(alphas)

    alphas = torch.from_numpy(alphas).to(device)
    betas = torch.from_numpy(betas).to(device)
    scale = torch.tensor(scale).to(device)
    translation = torch.from_numpy(translation).to(device)
    skeletons_torch = torch.from_numpy(skeletons).float().to(device)

    batch_tensor = torch.ones((length, 1), dtype=torch.float32).to(device)

    alphas.requires_grad = True
    betas.requires_grad = True
    scale.requires_grad = True
    translation.requires_grad = True

    masks_torch, params_torch = masks_params_torch(masks, params)

    lr = 2e-3
    optim_params = [
        {'params': alphas, 'lr': lr},
        {'params': betas, 'lr': lr},
        {'params': scale, 'lr': lr},
        {'params': translation, 'lr': lr},]
    optimizer = torch.optim.Adam(optim_params)

    # TODO: maybe add transfromation term as well
    loss_init = None
    loss_distance_init = None
    loss_chamfer_init = None
    bar = tqdm(range(configs['epochs']))
    for _ in bar:
        verts, joints = model(
            alphas,
            th_betas=betas * batch_tensor)

        verts = verts * scale + translation
        joints = joints * scale + translation

        loss_distance = calc_distance(joints, skeletons_torch, skeleton_weights)
        
        verts = denormalize(verts, denormalize_params)
        loss_chamfer = calc_chamfer(verts, masks_torch, params_torch, configs)

        loss = loss_distance * configs['weight_distance'] + loss_chamfer * configs['weight_chamfer']

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss = loss.detach().cpu().item()
        if loss_init is None:
            loss_init = loss
            loss_distance_init = loss_distance * configs['weight_distance']
            loss_chamfer_init = loss_chamfer * configs['weight_chamfer']

        bar.set_description(
            "L: {:.2E} LD: {:.2E} LC: {:.2E} S:{:.2f}".format(
                loss,
                loss_distance * configs['weight_distance'],
                loss_chamfer * configs['weight_chamfer'],
                scale.item(),
            )
        )

    print('L {:.5f} to {:.5f}\n D {:.5f} to {:.5f}\nCH {:.5f} to {:.5f}'.format(
            loss_init, loss,
            loss_distance_init, loss_distance  * configs['weight_distance'],
            loss_chamfer_init, loss_chamfer * configs['weight_chamfer'],
        )
    )

    return alphas.detach().cpu().numpy(), \
        betas.detach().cpu().numpy(), \
        translation.detach().cpu().numpy(), \
        scale.detach().cpu().numpy(), \


# TODO: Shorten
def visualize_poses(alphas, betas,
                    translation, scale,
                    faces, configs):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    alphas = torch.from_numpy(alphas).to(device)
    betas = torch.from_numpy(betas).to(device)
    batch_tensor = torch.ones((alphas.shape[0], 1)).to(device)

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

    pcd_joints = o3d.geometry.PointCloud()
    mesh = o3d.geometry.TriangleMesh()
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    mesh_line = o3d.geometry.LineSet()
    for idx in range(alphas.shape[0]):
        pcd_joints.points = o3d.utility.Vector3dVector(joints[idx])
        pcd_joints.paint_uniform_color([1, 0, 1])
        if idx == 0:
            vis.add_geometry(pcd_joints)
        else:
            vis.update_geometry(pcd_joints)

        if configs['visualize_mesh']:
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


def get_mask_image(camera, idx, configs):
    img_depth = os.path.join(
        configs['images_mask'][camera],
        'mask{:05d}.jpg'.format(idx)
    )

    return img_depth


def get_mask(cam, idx, params, configs):
    img_mask = get_mask_image(cam, idx, configs)
    
    img_mask = cv2.imread(img_mask, -1)
    kernel = np.ones((5, 5), np.uint8)
    img_mask = cv2.erode(img_mask, kernel, iterations=2)

    mtx = np.array(params[cam]['mtx'], np.float32)
    dist = np.array(params[cam]['dist'], np.float32)
    img_mask = cv2.undistort(img_mask, mtx, dist, None, None)

    img_mask = cv2.resize(
        img_mask,
        (img_mask.shape[1] // configs['scale_mask'],
         img_mask.shape[0] // configs['scale_mask']))

    points = np.argwhere(img_mask > 0.7) * configs['scale_mask']
    points = np.flip(points, axis=1).copy()

    return points


def get_masks(cameras, params, length, configs):
    masks = {camera: [] for camera in cameras}

    print("Loading masks...")
    for idx in tqdm(range(length)):
        for camera in cameras:
            mask = get_mask(camera, idx, params, configs)
            masks[camera].append(
                mask
            )

    return masks


def load_smpl(subject, configs):
    with open(configs['params_smpl'], 'rb') as handle:
        params_smpl = yaml.safe_load(handle)

    alphas = np.array(params_smpl[subject]['alphas'], np.float32)
    betas = np.array(params_smpl[subject]['betas'], np.float32)
    translation = np.array(params_smpl[subject]['translation'], np.float32)    
    scale = params_smpl[subject]['scale']
    
    return alphas, betas, scale, translation


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

    with open(configs['params'], 'rb') as handle:
        params = yaml.safe_load(handle)

    cameras = list(params.keys())

    params_smpl = []
    for subject, bundle in enumerate(bundles):
        poses = np.array(bundle['pose_normalized'])

        alphas, betas, scale, translation = \
            load_smpl(subject, configs)

        masks = get_masks(cameras, params, poses.shape[0], configs)

        alphas, betas, translation, scale = optimize(
            model, masks, poses, alphas, betas, scale, translation,
            params, subject, configs)
        
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
                translation, scale, model.th_faces, configs)

        _store_artifacts(params_smpl, configs['output'])
