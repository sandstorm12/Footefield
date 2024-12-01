import sys
sys.path.append('../')

import os
import cv2
import time
import yaml
import torch
import smplx
import argparse
import numpy as np
import open3d as o3d
import torch.nn.functional as F

from tqdm import tqdm
from utils import data_loader
from pytorch3d.loss import chamfer_distance


COEFF_HGIH = 10
COEFF_NORM = 1
COEFF_MINI = 1
SMPLX_SKELETON_MAP = np.array([ # (SMPLX, HALPE)
    [16, 5, COEFF_NORM],
    [17, 6, COEFF_NORM],
    [1, 11, COEFF_NORM],
    [2, 12, COEFF_NORM],
    [4, 13, COEFF_HGIH],
    [5, 14, COEFF_HGIH],
    [7, 15, COEFF_HGIH],
    [8, 16, COEFF_HGIH], # Body
    [19, 8, COEFF_HGIH],
    [21, 10, COEFF_HGIH],
    [18, 7, COEFF_HGIH],
    [20, 9, COEFF_HGIH], # Arms
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


def calc_distance(joints, skeleton, skeleton_weights):
    skeleton_selected = skeleton[:, SMPLX_SKELETON_MAP[:, 1]]
    output_selected = joints[:, SMPLX_SKELETON_MAP[:, 0]]

    loss = F.smooth_l1_loss(
        output_selected, skeleton_selected, reduction='none')
    
    # Just for test, optimize
    loss = torch.mean(loss, dim=(0, 2))
    loss = torch.mean(loss * skeleton_weights)

    return loss


def get_mask_image(camera, idx, configs):
    img_depth = os.path.join(
        configs['images_mask'][camera],
        'mask{:05d}.jpg'.format(idx)
    )

    return img_depth


def load_smplx_params(smplx_params, device):
    global_orient = torch.from_numpy(
        np.array(smplx_params['global_orient'], np.float32)).to(device)
    jaw_pose = torch.from_numpy(
        np.array(smplx_params['jaw_pose'], np.float32)).to(device)
    leye_pose = torch.from_numpy(
        np.array(smplx_params['leye_pose'], np.float32)).to(device)
    reye_pose = torch.from_numpy(
        np.array(smplx_params['reye_pose'], np.float32)).to(device)
    body = torch.from_numpy(
        np.array(smplx_params['body'], np.float32)).to(device)
    left_hand_pose = torch.from_numpy(
        np.array(smplx_params['left_hand_pose'], np.float32)).to(device)
    right_hand_pose = torch.from_numpy(
        np.array(smplx_params['right_hand_pose'], np.float32)).to(device)
    betas = torch.from_numpy(
        np.array(smplx_params['betas'], np.float32)).to(device)
    expression = torch.from_numpy(
        np.array(smplx_params['expression'], np.float32)).to(device)
    translation = torch.from_numpy(
        np.array(smplx_params['translation'], np.float32)).to(device)
    scale = torch.from_numpy(
        np.array(smplx_params['scale'], np.float32)
    )

    global_orient.requires_grad = False
    jaw_pose.requires_grad = False
    leye_pose.requires_grad = False
    reye_pose.requires_grad = False
    body.requires_grad = False
    left_hand_pose.requires_grad = False
    right_hand_pose.requires_grad = False
    betas.requires_grad = True
    expression.requires_grad = False
    translation.requires_grad = False
    scale.requires_grad = False

    return global_orient, jaw_pose, leye_pose, reye_pose, body, \
        left_hand_pose, right_hand_pose, betas, expression, \
        translation, scale


def load_denormalize_params(subject, device, configs):
    with open(configs['skeletons'], 'rb') as handle:
        params = yaml.safe_load(handle)

    rotation = np.array(params[subject]['rotation'])
    scale = np.array(params[subject]['scale'])
    translation = np.array(params[subject]['translation'])

    rotation_inverted = np.transpose(np.linalg.inv(rotation), (0, 2, 1))

    rotation_inverted = torch.from_numpy(rotation_inverted).float().to(device)
    scale = torch.from_numpy(scale).float().to(device)
    translation = torch.from_numpy(translation).float().to(device)

    return rotation_inverted, scale, translation


def denormalize(verts, denormalize_params):
    rotation_inverted, scale, translation = denormalize_params

    verts = torch.bmm(verts, rotation_inverted)
    verts = verts * scale
    verts += translation.unsqueeze(1)

    return verts


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


def calc_chamfer(verts, masks, params):
    loss = []
    for cam in range(len(masks)):
        for idx_mask in range(len(masks[cam])):
            mask_torch = masks[cam][idx_mask]
            mtx = params[cam]['mtx']
            rotation = params[cam]['rotation']
            translation = params[cam]['translation']
            pcd_proj = project_points_to_camera_plane(
                verts[idx_mask].unsqueeze(0), mtx,
                rotation, translation,)
            if configs['visualize_chamfer_projection']:
                visualize_chamfer(
                    mask_torch.squeeze().detach().cpu().numpy(),
                    pcd_proj.squeeze().detach().cpu().numpy())
            distances = chamfer_distance(
                pcd_proj, mask_torch,
                single_directional=False, norm=2,
                point_reduction=None, batch_reduction=None)[0]
            loss_verts = torch.mean(distances[0])
            loss_mask = torch.mean(
                distances[1][distances[1] < torch.max(distances[0])])
            loss.append(loss_verts + loss_mask)

    loss = torch.stack(loss) 
    loss = torch.mean(loss)

    return loss


# TODO: Shorten
def optimize_beta(model, poses, masks, subject, params, params_smpl, configs):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    denormalize_params = load_denormalize_params(subject, device, configs)

    skeleton_weights = torch.from_numpy(SMPLX_SKELETON_MAP[:, 2]).float().to(device)

    batch_tensor = torch.ones((poses.shape[0], 1)).to(device)

    global_orient, jaw_pose, leye_pose, reye_pose, \
        body, left_hand_pose, right_hand_pose, betas, expression, \
        translation, scale = load_smplx_params(params_smpl[subject], device)

    lr = configs['learning_rate']
    optim_params = [
        {'params': global_orient, 'lr': lr},
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

    masks_torch, params_torch = masks_params_torch(masks, params)

    skeletons_torch = torch.from_numpy(poses).float().to(device)

    # TODO: maybe add transfromation term as well
    loss_init = None
    bar = tqdm(range(configs['epochs']))
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
        verts = output.vertices
        
        verts = verts - joints[0, 0] + translation.unsqueeze(1)
        verts = verts * scale
        joints = joints - joints[0, 0] + translation.unsqueeze(1)
        joints = joints * scale

        loss_distance = calc_distance(joints, skeletons_torch, skeleton_weights)

        verts = denormalize(verts, denormalize_params)
        loss_chamfer = calc_chamfer(verts, masks_torch, params_torch)

        loss = loss_distance * configs['weight_distance'] \
            + loss_chamfer * configs['weight_chamfer'] \
            + scale * configs['weight_scale']

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss = loss.detach().cpu().item()
        if loss_init is None:
            loss_init = loss
            loss_distance_init = loss_distance * configs['weight_distance']
            loss_chamfer_init = loss_chamfer * configs['weight_chamfer']

        bar.set_description(
            "L: {:.2E} D: {:.2E} C: {:.2E} S:{:.2f}".format(
                loss,
                loss_distance * configs['weight_distance'],
                loss_chamfer * configs['weight_chamfer'],
                scale.item(),
            )
        )

    print('L {:.2E} to {:.2E}\n D {:.2E} to {:.2E}\nCH {:.2E} to {:.2E}'.format(
            loss_init, loss,
            loss_distance_init, loss_distance  * configs['weight_distance'],
            loss_chamfer_init, loss_chamfer * configs['weight_chamfer'],
        )
    )

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


def get_masks(cameras, params, length, configs):
    masks = {camera: [] for camera in cameras}

    kernel = np.ones((5, 5), np.uint8)

    print("Loading masks...")
    for camera in tqdm(cameras):
        mtx = np.array(params[camera]['mtx'], np.float32)
        dist = np.array(params[camera]['dist'], np.float32)

        dir = configs['videos_mask'][camera]['path']
        offset = configs['videos_mask'][camera]['offset']
        cap = cv2.VideoCapture(dir)
        for _ in range(offset):
            cap.grab()
        video_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - offset
        for _ in range(video_frame_count):
            _, img_mask = cap.read()
            img_mask = cv2.cvtColor(img_mask, cv2.COLOR_BGR2GRAY)
            # img_mask = cv2.erode(img_mask, kernel, iterations=3)
            img_mask = cv2.undistort(img_mask, mtx, dist, None, None)
            img_mask = cv2.resize(
                img_mask,
                (img_mask.shape[1] // configs['scale_mask'],
                img_mask.shape[0] // configs['scale_mask']))

            mask = np.argwhere(img_mask > 0.7) * configs['scale_mask']
            mask = np.flip(mask, axis=1).copy()
            masks[camera].append(
                mask
            )

    return masks



def _store_artifacts(artifact, output):
    with open(output, 'w') as handle:
        yaml.dump(artifact, handle)


if __name__ == '__main__':
    args = _get_arguments()
    configs = _load_configs(args.config)

    print(f"Config loaded: {configs}")

    model = smplx.create(
        configs['models_root'], model_type='smplx',
        gender=configs['gender'], use_face_contour=False,
        num_betas=10, use_pca=False,
        num_expression_coeffs=10,
        ext='npz')
    
    with open(configs['skeletons'], 'rb') as handle:
        bundles = yaml.safe_load(handle)

    with open(configs['params'], 'rb') as handle:
        params = yaml.safe_load(handle)

    with open(configs['params_smplx'], 'rb') as handle:
        params_smpl = yaml.safe_load(handle)

    cameras = list(params.keys())

    params_smplx = []
    for subject, bundle in enumerate(bundles):
        poses = np.array(bundle['pose_normalized'])

        masks = get_masks(cameras, params, poses.shape[0], configs)

        global_orient, jaw_pose, leye_pose, \
            reye_pose,body,left_hand_pose, \
            right_hand_pose, betas, expression, \
            translation, scale = optimize_beta(
                model, poses, masks, subject, params, params_smpl, configs)
        
        # Do we need to add a rotation parameter as well?
        params_smplx_person = {
            'global_orient': global_orient.tolist(),
            'jaw_pose': jaw_pose.tolist(),
            'leye_pose': leye_pose.tolist(),
            'reye_pose': reye_pose.tolist(),
            'body': body.tolist(),
            'left_hand_pose': left_hand_pose.tolist(),
            'right_hand_pose': right_hand_pose.tolist(),
            'betas': betas.tolist(),
            'expression': expression.tolist(),
            'translation': translation.tolist(),
            'scale': scale.item(),
        }
        params_smplx.append(params_smplx_person)

        if configs['visualize']:
            visualize_poses(
                global_orient, jaw_pose, leye_pose,
                reye_pose,body,left_hand_pose,
                right_hand_pose, betas, expression,
                translation, scale, model.faces, poses)

        _store_artifacts(params_smplx, configs['output'])
