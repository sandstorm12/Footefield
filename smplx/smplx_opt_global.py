import sys
sys.path.append('../')

import os
import cv2
import time
import smplx
import torch
import pickle
import numpy as np
import open3d as o3d
import torch.nn.functional as F

from tqdm import tqdm
from torch import fft
from utils import data_loader
from pytorch3d.loss import chamfer_distance


DIR_SKELETONS = '../pose_estimation/keypoints_3d_pose2smpl_x'
DIR_OUTPUT = './params_smplx_mask'
DIR_ORG = '../pose_estimation/keypoints_3d_ba_x'
DIR_SMPLX = './params_smplx'

EXPERIMENTS = ['a1', 'a2']
SUBJECTS = [0, 1]
EPOCHS = 500
VISUALIZE = False
VISUALIZE_PROJ = False
PATH_MODEL = 'models'
PARAM_SCALE_MASK = 8

PARAM_WEIGHT_CHMF = 1e-8
PARAM_WEIGHT_SMTH = 1e-4
PARAM_WEIGHT_DIST = 1

COEFF_NORM = 1
COEFF_MINI = .5
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


def calc_distance(joints, skeleton):
    skeleton_selected = skeleton[:, SMPLX_SKELETON_MAP[:, 1]]
    output_selected = joints[:, SMPLX_SKELETON_MAP[:, 0]]

    loss = F.smooth_l1_loss(output_selected, skeleton_selected, reduction='none')
    
    # Just for test, optimize
    loss = torch.mean(loss, dim=(0, 2))
    loss = torch.mean(loss * torch.from_numpy(SMPLX_SKELETON_MAP[:, 2]).float().cuda())

    return loss


def calc_smooth(joints):
    target = joints[:, SMPLX_SKELETON_MAP[:, 0]]

    target = target.transpose(1, 2).reshape((target.shape[0] * target.shape[2], target.shape[1]))

    fft_target = fft.rfft(target)
    
    loss = torch.mean(torch.abs(fft_target[:, 10:]) ** 2)
    
    return loss


def get_mask_image(cam_name, experiment, idx):
    cam_num = cam_name[12:15]
    img_depth = '/home/hamid/Documents/phd/footefield/data/AzureKinectRecord_0729/{}/azure_kinect{}/mask/mask{:05d}.jpg'.format(experiment, cam_num, idx)

    return img_depth


def load_smplx_params(experiment, subject, device):
    path_smplx = os.path.join(
        DIR_SMPLX,
        f"params_smplx_{experiment}_{subject}.pkl")
    with open(path_smplx, 'rb') as handle:
        smplx_params = pickle.load(handle)
    
    global_orient = torch.from_numpy(smplx_params['global_orient']).to(device)
    jaw_pose = torch.from_numpy(smplx_params['jaw_pose']).to(device)
    leye_pose = torch.from_numpy(smplx_params['leye_pose']).to(device)
    reye_pose = torch.from_numpy(smplx_params['reye_pose']).to(device)
    body = torch.from_numpy(smplx_params['body']).to(device)
    left_hand_pose = torch.from_numpy(smplx_params['left_hand_pose']).to(device)
    right_hand_pose = torch.from_numpy(smplx_params['right_hand_pose']).to(device)
    betas = torch.from_numpy(smplx_params['betas']).to(device)
    expression = torch.from_numpy(smplx_params['expression']).to(device)
    translation_smplx = torch.from_numpy(smplx_params['translation']).to(device)
    scale_smplx = torch.from_numpy(smplx_params['scale']).to(device)

    global_orient.requires_grad = True
    jaw_pose.requires_grad = True
    leye_pose.requires_grad = True
    reye_pose.requires_grad = True
    body.requires_grad = True
    left_hand_pose.requires_grad = True
    right_hand_pose.requires_grad = True
    betas.requires_grad = True
    expression.requires_grad = True
    translation_smplx.requires_grad = True
    scale_smplx.requires_grad = True

    return global_orient, jaw_pose, leye_pose, reye_pose, \
        body, left_hand_pose, right_hand_pose, betas, expression, \
        translation_smplx, scale_smplx


def load_denormalize_params(experiment, subject, device):
    # Load alignment params
    path_params = os.path.join(
        DIR_SKELETONS,
        f"keypoints3d_{experiment}_ba_{subject}_params.pkl")
    with open(path_params, 'rb') as handle:
        params = pickle.load(handle)
    rotation = params['rotation']
    scale = params['scale']
    translation = params['translation']
    rotation_inverted = np.linalg.inv(rotation).T

    translation = torch.from_numpy(translation).float().to(device)
    rotation_inverted = torch.from_numpy(rotation_inverted).float().to(device)

    return rotation_inverted, scale, translation


def denormalize(verts, denormalize_params):
    rotation_inverted, scale, translation = denormalize_params

    verts = torch.matmul(verts, rotation_inverted)
    verts = verts * scale
    verts = verts + translation

    return verts


def masks_params_torch(masks, params):
    masks_torch = []
    params_torch = []
    for cam in range(len(masks)):
        masks_torch_cam = []
        for idx_mask in range(len(masks[cam])):
            masks_torch_cam.append(
                torch.from_numpy(masks[cam][idx_mask]).float().unsqueeze(0).cuda())
        masks_torch.append(masks_torch_cam)
        
        mtx = torch.from_numpy(params[cam]['mtx']).float().cuda().unsqueeze(0)
        rotation = torch.from_numpy(params[cam]['rotation']).float().cuda().unsqueeze(0)
        translation = torch.from_numpy(
            params[cam]['translation']).float().cuda().unsqueeze(0)
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
            if VISUALIZE_PROJ and cam == 0:
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
            loss += loss_verts + loss_mask

    return loss


# TODO: Shorten
def optimize_beta(smpl_layer, skeletons, masks, experiment, subject, epochs):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    smpl_layer.to(device)

    denormalize_params = load_denormalize_params(experiment, subject, device)

    batch_tensor = torch.ones((skeletons.shape[0], 1)).to(device)

    global_orient, jaw_pose, leye_pose, reye_pose, \
        body, left_hand_pose, right_hand_pose, betas, expression, \
        translation, scale = load_smplx_params(experiment, subject, device)

    lr = 2e-3
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

    masks_torch, params_torch = masks_params_torch(masks, params)

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
        verts = output.vertices
        
        verts = verts - joints[0, 0] + translation
        verts = verts * scale
        joints = joints - joints[0, 0] + translation
        joints = joints * scale

        loss_distance = calc_distance(joints, skeletons_torch)

        verts = denormalize(verts, denormalize_params)
        loss_chamfer = calc_chamfer(verts, masks_torch, params_torch)

        loss_smooth = calc_smooth(joints)

        loss = loss_distance * PARAM_WEIGHT_DIST + loss_chamfer * PARAM_WEIGHT_CHMF + loss_smooth * PARAM_WEIGHT_SMTH

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss = loss.detach().cpu().item()
        if loss_init is None:
            loss_init = loss
            loss_distance_init = loss_distance * PARAM_WEIGHT_DIST
            loss_chamfer_init = loss_chamfer * PARAM_WEIGHT_CHMF
            loss_smooth_init = loss_smooth * PARAM_WEIGHT_SMTH

        bar.set_description(
            "L: {:.2E} LD: {:.2E} LC: {:.2E} STH: {:.2E} S:{:.2f}".format(
                loss,
                loss_distance * PARAM_WEIGHT_DIST,
                loss_chamfer * PARAM_WEIGHT_CHMF,
                loss_smooth * PARAM_WEIGHT_SMTH,
                scale.item(),
            )
        )

    print('L {:.2E} to {:.2E}\n D {:.2E} to {:.2E}\nCH {:.2E} to {:.2E}\nSTH {:.2E} to {:.2E}'.format(
            loss_init, loss,
            loss_distance_init, loss_distance  * PARAM_WEIGHT_DIST,
            loss_chamfer_init, loss_chamfer * PARAM_WEIGHT_CHMF,
            loss_smooth_init, loss_smooth * PARAM_WEIGHT_SMTH,
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


def get_mask(cam, experiment, idx, params):
    img_mask = get_mask_image(cam, experiment, idx)
    
    img_mask = cv2.imread(img_mask, -1)
    kernel = np.ones((5, 5), np.uint8)
    img_mask = cv2.erode(img_mask, kernel, iterations=2)

    mtx = params[cameras.index(cam)]['mtx']
    dist = params[cameras.index(cam)]['dist']
    img_mask = cv2.undistort(img_mask, mtx, dist, None, None)

    img_mask = cv2.resize(
        img_mask,
        (img_mask.shape[1] // PARAM_SCALE_MASK,
         img_mask.shape[0] // PARAM_SCALE_MASK))

    points = np.argwhere(img_mask > 0.7) * PARAM_SCALE_MASK
    points = np.flip(points, axis=1).copy()

    return points


def get_masks(experiment, params, depth):
    masks = [[], [], [], []]

    print("Loading masks...")
    for idx in tqdm(range(depth)):
        mask24 = get_mask(cam24, experiment, idx, params)
        mask15 = get_mask(cam15, experiment, idx, params)
        # mask24 = get_mask(cam24, experiment, idx, params)
        mask34 = get_mask(cam34, experiment, idx, params)
        mask35 = get_mask(cam35, experiment, idx, params)

        masks[0].append(
            mask24
        )
        masks[1].append(
            mask15
        )
        masks[2].append(
            mask34
        )
        masks[3].append(
            mask35
        )

    return masks


def get_params_color(expriment):
    file = f"keypoints3d_{expriment}_ba.pkl"
    file_path = os.path.join(DIR_ORG, file)
    with open(file_path, 'rb') as handle:
        output = pickle.load(handle)

    params = output['params']

    return params


cam24 = 'azure_kinect2_4_calib_snap'
cam15 = 'azure_kinect1_5_calib_snap'
cam14 = 'azure_kinect1_4_calib_snap'
cam34 = 'azure_kinect3_4_calib_snap'
cam35 = 'azure_kinect3_5_calib_snap'
cameras = [
    cam24,
    cam15,
    # cam14,
    cam34,
    cam35,   
]
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

            params = get_params_color(experiment)
            masks = get_masks(experiment, params, skeletons.shape[0])

            print(f'Optimizing {experiment} {subject}')
            global_orient, jaw_pose, leye_pose, \
                reye_pose,body,left_hand_pose, \
                right_hand_pose, betas, expression, \
                translation, scale = optimize_beta(
                    model, skeletons, masks, experiment, subject, EPOCHS)

            store_smplx_parameters(
                global_orient, jaw_pose, leye_pose,
                reye_pose,body,left_hand_pose,
                right_hand_pose, betas, expression,
                translation, scale,
                experiment, subject)
