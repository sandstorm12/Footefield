import dis
import sys
sys.path.append('../')

import os
import cv2
import time
import torch
import pickle
import numpy as np
import open3d as o3d
import torch.nn.functional as F

from tqdm import tqdm
from utils import data_loader
from pytorch3d.loss import chamfer_distance
from smplpytorch.pytorch.smpl_layer import SMPL_Layer


DIR_SKELETONS = '../pose_estimation/keypoints_3d_pose2smpl'
DIR_ORG = '../pose_estimation/keypoints_3d_ba'
DIR_OUTPUT = './params_smpl_mask'
DIR_SMPL = 'params_smpl'
DIR_PARAMS = '../pose_estimation/keypoints_3d_pose2smpl/'

EXPERIMENTS = ['a1', 'a2']
SUBJECTS = [0, 1]
EPOCHS = 100
VISUALIZE = False
VISUALIZE_MESH = True
VISUALIZE_PROJ = False
PARAM_SCALE_MASK = 8
PATH_MODEL = 'models'

PARAM_WEIGHT_CHMF = 1e-8
PARAM_WEIGHT_DIST = 1


SMPL_SKELETON_MAP = np.array([ # (SMPL, SKELETON)
    [0, 19, 1.],
    [1, 11, 1.],
    [2, 12, 1.],
    [4, 13, 1.],
    [5, 14, 1.],
    [6, 26, .25],
    [7, 15, 1.],
    [8, 16, 1.],
    [10, 27, 1.],
    [11, 28, 1.],
    [12, 18, 1.],
    [16, 5, 1.],
    [17, 6, 1.],
    [15, 0, .25],
    [18, 7, 1.],
    [19, 8, 1.],
    [20, 9, 1.],
    [21, 10, 1.],
])


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


def calc_distance(joints, skeleton):
    skeleton_selected = skeleton[:, SMPL_SKELETON_MAP[:, 1]]
    output_selected = joints[:, SMPL_SKELETON_MAP[:, 0]]

    loss = F.smooth_l1_loss(output_selected, skeleton_selected, reduction='none')
    
    # Just for test, optimize
    loss = torch.mean(loss, dim=(0, 2))
    loss = torch.mean(loss * torch.from_numpy(SMPL_SKELETON_MAP[:, 2]).float().cuda())

    return loss


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
            loss_mask = torch.mean(distances[1][distances[1] < torch.max(distances[0])])
            loss += loss_verts + loss_mask

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


def load_smpl(experiment, subject):
    name = "params_smpl_{}_{}.pkl".format(
        experiment, subject
    )
    path_smpl = os.path.join(DIR_SMPL, name)
    with open(path_smpl, 'rb') as handle:
        smpl = pickle.load(handle)

    alphas = np.array(smpl['alphas'])
    betas = np.array(smpl['betas'])
    scale = np.array(smpl['scale'])
    translation = np.array(smpl['translation'])

    return alphas, betas, scale, translation


def denormalize(verts, denormalize_params):
    rotation_inverted, scale, translation = denormalize_params

    verts = torch.matmul(verts, rotation_inverted)
    verts = verts * scale
    verts = verts + translation

    return verts


def load_denormalize_params():
    path_params = os.path.join(DIR_PARAMS, f"keypoints3d_{experiment}_ba_{subject}_params.pkl")
    with open(path_params, 'rb') as handle:
        params = pickle.load(handle)
    rotation = params['rotation']
    scale = params['scale']
    translation = params['translation']

    rotation_inverted = np.linalg.inv(rotation).T

    rotation_inverted = torch.from_numpy(rotation_inverted).float().to('cuda')
    translation = torch.from_numpy(translation).float().to('cuda')

    return rotation_inverted, scale, translation


def masks_params_torch(masks, params):
    masks_torch = []
    params_torch = []
    for cam in range(len(masks)):
        masks_torch_cam = []
        for idx_mask in range(len(masks[cam])):
            masks_torch_cam.append(torch.from_numpy(masks[cam][idx_mask]).float().unsqueeze(0).cuda())
        masks_torch.append(masks_torch_cam)
        
        mtx = torch.from_numpy(params[cam]['mtx']).float().cuda().unsqueeze(0)
        rotation = torch.from_numpy(params[cam]['rotation']).float().cuda().unsqueeze(0)
        translation = torch.from_numpy(params[cam]['translation']).float().cuda().unsqueeze(0)
        params_torch.append(
            {
                'mtx': mtx,
                'rotation': rotation,
                'translation': translation,
            }
        )

    return masks_torch, params_torch


# TODO: Shorten
def optimize(smpl_layer, masks, skeletons, params_smpl, params, experiment, subject, epochs):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    smpl_layer.to(device)

    denormalize_params = load_denormalize_params()

    depth = len(masks[0])

    alphas = torch.from_numpy(params_smpl[0]).to(device)
    betas = torch.from_numpy(params_smpl[1]).to(device)
    scale = torch.from_numpy(params_smpl[2]).to(device)
    translation = torch.from_numpy(params_smpl[3]).to(device)
    skeletons_torch = torch.from_numpy(skeletons).float().to(device)

    batch_tensor = torch.ones((depth, 1)).to(device)

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
    bar = tqdm(range(epochs))
    for _ in bar:
        verts, joints = model(
            alphas,
            th_betas=betas * batch_tensor)

        verts = verts * scale + translation
        joints = joints * scale + translation

        loss_distance = calc_distance(joints, skeletons_torch)
        
        verts = denormalize(verts, denormalize_params)
        loss_chamfer = calc_chamfer(verts, masks_torch, params_torch)

        loss = loss_distance * PARAM_WEIGHT_DIST + loss_chamfer * PARAM_WEIGHT_CHMF

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss = loss.detach().cpu().item()
        if loss_init is None:
            loss_init = loss
            loss_distance_init = loss_distance * PARAM_WEIGHT_DIST
            loss_chamfer_init = loss_chamfer * PARAM_WEIGHT_CHMF

        bar.set_description(
            "L: {:.2E} LD: {:.2E} LC: {:.2E} S:{:.2f}".format(
                loss,
                loss_distance * PARAM_WEIGHT_DIST,
                loss_chamfer * PARAM_WEIGHT_CHMF,
                scale.item(),
            )
        )

    print('L {:.4f} to {:.4f}\n D {:.4f} to {:.4f}\nCH {:.4f} to {:.4f}'.format(
            loss_init, loss,
            loss_distance_init, loss_distance  * PARAM_WEIGHT_DIST,
            loss_chamfer_init, loss_chamfer * PARAM_WEIGHT_CHMF,
        )
    )

    return alphas.detach().cpu().numpy(), \
        betas.detach().cpu().numpy(), \
        translation.detach().cpu().numpy(), \
        scale.detach().cpu().numpy(), \


# TODO: Shorten
def visualize_poses(alphas, betas,
                    translation, scale,
                    faces):
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

    pcd_skeleton_org = o3d.geometry.PointCloud()
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

        if VISUALIZE_MESH:
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


def get_mask_image(cam_name, experiment, idx):
    cam_num = cam_name[12:15]
    img_depth = '/home/hamid/Documents/phd/footefield/data/AzureKinectRecord_0729/{}/azure_kinect{}/mask/mask{:05d}.jpg'.format(experiment, cam_num, idx)

    return img_depth


def get_params_color(expriment):
    file = f"keypoints3d_{expriment}_ba.pkl"
    file_path = os.path.join(DIR_ORG, file)
    with open(file_path, 'rb') as handle:
        output = pickle.load(handle)

    params = output['params']

    return params


def get_mask(cam, experiment, idx, params):
    img_mask = get_mask_image(cam, experiment, idx)
    
    img_mask = cv2.imread(img_mask, -1)
    kernel = np.ones((5, 5), np.uint8)
    img_mask = cv2.erode(img_mask, kernel, iterations=5)

    # cv2.imshow("before", cv2.resize(img_mask, (1280, 720)))
    # cv2.imshow("after", cv2.resize(image, (1280, 720)))
    # cv2.waitKey(0)

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


def store_smpl_parameters(alphas, betas,
        translation, scale, experiment, subject):
    if not os.path.exists(DIR_OUTPUT):
        os.mkdir(DIR_OUTPUT)

    path = os.path.join(
        DIR_OUTPUT,
        f'params_smpl_{experiment}_{subject}.pkl')
    
    params = {
        'alphas': alphas,
        'betas': betas,
        'translation': translation,
        'scale': scale,
    }

    with open(path, 'wb') as handle:
        pickle.dump(params, handle,
                    protocol=pickle.HIGHEST_PROTOCOL)
        
    print(f'Stored results: {path}')


def load_smpl_params(experiment, subject):
    name = "params_smpl_{}_{}.pkl".format(
        experiment, subject
    )
    path_smpl = os.path.join(DIR_SMPL, name)
    with open(path_smpl, 'rb') as handle:
        smpl = pickle.load(handle)

    alphas = np.array(smpl['alphas'])
    betas = np.array(smpl['betas'])
    scale = np.array(smpl['scale'])
    translation = np.array(smpl['translation'])

    return alphas, betas, scale, translation


def load_skeletons(experiment, subject):
    path = os.path.join(DIR_SKELETONS,
                        f'keypoints3d_{experiment}_ba_{subject}_normalized.npy')
    skeletons = np.load(path)

    return skeletons



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
    model = SMPL_Layer(
        center_idx=0,
        gender='neutral',
        model_root='models')

    for experiment in EXPERIMENTS:
        for subject in SUBJECTS:
            print(f'Optimizing {experiment} {subject}')

            skeletons = load_skeletons(experiment, subject)
            params_smpl = load_smpl_params(experiment, subject)
            params = get_params_color(experiment)
            masks = get_masks(experiment, params, params_smpl[0].shape[0])

            alphas, betas, translation, scale = optimize(
                model, masks, skeletons, params_smpl, params, experiment, subject, EPOCHS)

            if VISUALIZE:
                visualize_poses(
                    alphas, betas,
                    translation, scale, model.th_faces)

            store_smpl_parameters(
                alphas, betas,
                translation, scale,
                experiment, subject)
