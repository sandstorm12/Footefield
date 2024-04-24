from calendar import EPOCH
import os
import sys
sys.path.append('../')

import cv2
import time
import glob
import pickle
import torch
import diskcache
import numpy as np
import open3d as o3d

from pytorch3d.loss import chamfer_distance
from utils import data_loader
from sklearn.cluster import KMeans
from tqdm import tqdm
from smplpytorch.pytorch.smpl_layer import SMPL_Layer


DIR_OUTPUT = './extrinsics_mask'
DIR_ORG = '../pose_estimation/keypoints_3d_ba'
DIR_PARAMS_FINETUNED = "./extrinsics_finetuned"
DIR_PARAMS_TRANSFORM = '../pose_estimation/keypoints_3d_pose2smpl/'
DIR_SMPL = '/home/hamid/Documents/phd/footefield/Pose_to_SMPL/fit/output/HALPE/'
DIR_SMPL_OPT = './params_smpl'
PARAM_DEPTH = 20
PARAM_SCALE_MASK = 2
PARAM_EPOCHS = 100


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
    mtx = params[cameras.index(cam)]['mtx']
    dist = params[cameras.index(cam)]['dist']
    img_mask = cv2.undistort(img_mask, mtx, dist, None, None)

    img_mask = cv2.resize(
        img_mask,
        (img_mask.shape[1] // PARAM_SCALE_MASK,
         img_mask.shape[0] // PARAM_SCALE_MASK))
    
    # img_mask[img_mask < .7] = 0
    # cv2.imshow("mask", img_mask)
    # cv2.waitKey(20)

    # points = np.empty(shape=((img_mask > .7).sum(), 2))
    # idx = 0
    # for y in range(img_mask.shape[0]):
    #     for x in range(img_mask.shape[1]):
    #         if img_mask[y, x] > .7:
    #             points[idx] = (x * 4, y * 4)
    #             idx += 1

    points = np.argwhere(img_mask > 0.7) * PARAM_SCALE_MASK
    points = np.flip(points, axis=1).copy()

    return points


def load_finetuned_extrinsics():
    extrinsics_finetuned = {}
    for path in glob.glob(os.path.join(DIR_PARAMS_FINETUNED, '*')):
        experiment = path.split('.')[-2].split('_')[-2]
        subject = path.split('.')[-2].split('_')[-1]
        with open(path, 'rb') as handle:
            params = pickle.load(handle)

        extrinsics_finetuned[experiment + '_' + subject] = params

    return extrinsics_finetuned


def get_corresponding_files(path, experiment):
    file_name = path.split('/')[-1].split('.')[0]

    files = [
        (
            file_name + '_0_normalized_params.pkl',
            file_name + '_0_params.pkl',
            f'params_smpl_{experiment}_1.pkl',
        ),(
            file_name + '_1_normalized_params.pkl',
            file_name + '_1_params.pkl',
            f'params_smpl_{experiment}_0.pkl'
        ),
    ]

    return files


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

    # # Project to image plane using intrinsic matrix
    # projected_points = torch.matmul(points_3d, mtx)

    # Return only the first two elements (x, y) for pixel coordinates
    return points_3d[:, :, :2]


def project_3d_to_2d(camera_matrix, dist_coeffs, rvec, tvec, object_points):
    image_points, _ = cv2.projectPoints(object_points, rvec, tvec,
                                        camera_matrix, None)

    image_points = image_points.squeeze()

    return image_points


def get_masks(experiment, params):
    masks = ([], [], [], [])

    print("Loading masks...")
    for idx in tqdm(range(PARAM_DEPTH)):
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


def get_smpl_parameters(smpl_layer, file_org):
    experiment = file.split('.')[-2].split('_')[-2]
    files_smpl = get_corresponding_files(file_org, experiment)
        
    verts_all = []
    faces_all = []
    for file_smpl in files_smpl:
        # Load SMPL data
        path_smpl = os.path.join(DIR_SMPL, file_smpl[0])
        with open(path_smpl, 'rb') as handle:
            smpl = pickle.load(handle)
        scale_smpl = smpl['scale']
        transformation = smpl['transformation']

        # Load SMPL params and get verts
        path_smpl_opt = os.path.join(DIR_SMPL_OPT, file_smpl[2])
        with open(path_smpl_opt, 'rb') as handle:
            smpl_params = pickle.load(handle)
        pose_params = smpl_params['alpha']
        shape_params = np.tile(smpl_params['beta'], (pose_params.shape[0], 1))
        faces = smpl_params['faces']
        verts = []
        for idx in tqdm(range(pose_params.shape[0])):
            pose_torch = torch.from_numpy(
                pose_params[idx]).unsqueeze(0).float()
            shape_torch = torch.from_numpy(
                shape_params[idx]).unsqueeze(0).float()

            verts_single, _ = smpl_layer(pose_torch, th_betas=shape_torch)

            verts.append(verts_single.detach().cpu().numpy().astype(float))
        verts = np.array(verts).squeeze()

        # Load alignment params
        path_params = os.path.join(DIR_PARAMS_TRANSFORM, file_smpl[1])
        with open(path_params, 'rb') as handle:
            params = pickle.load(handle)
        rotation = params['rotation']
        scale = params['scale'] * scale_smpl
        translation = params['translation']

        rotation_inverted = np.linalg.inv(rotation)
        verts = np.concatenate(
            (verts,
                np.ones((verts.shape[0], verts.shape[1], 1))
            ), axis=2)
        verts = np.matmul(verts, transformation)
        verts = verts[:, :, :3] / verts[:, :, -1:]
        verts = verts.dot(rotation_inverted.T)
        verts = verts * scale
        verts = verts + translation

        verts_all.append(verts)
        faces_all.append(faces)

    verts_all = np.array(verts_all).squeeze()
    faces_all = np.array(faces_all)

    verts_all = np.transpose(verts_all, (1, 0, 2, 3))

    return verts_all, faces_all


def visualize_points(points, points2, name):
    img = np.zeros((1080, 1920, 3))
    
    points = points
    for point in points:
        x = int(point[0])
        y = int(point[1])
        if 0 < x < 1920 and 0 < y < 1080:
            img[y, x, 0] = 255

    points2 = points2
    for point in points2:
        x = int(point[0])
        y = int(point[1])
        if 0 < x < 1920 and 0 < y < 1080:
            img[y, x, 1] = 255

    cv2.imshow(name, cv2.resize(img, (960, 540)))
    cv2.waitKey(10)


def optimize(subject, experiment):
    params = get_params_color(experiment)

    smpl_layer = SMPL_Layer(
        center_idx=0,
        gender="neutral",
        model_root='models/')
    verts_all, faces_all = get_smpl_parameters(smpl_layer, file_path)
    verts_all = verts_all[:, (subject + 1) % 2].squeeze()
    masks24, masks15, masks34, masks35 = get_masks(experiment, params)

    # transformation = torch.eye(4).to("cuda").unsqueeze(0)
    transformation = torch.zeros(3).to("cuda").unsqueeze(0)
    transformation.requires_grad = True
    lr = 2e-0
    optim_params = [{'params': transformation, 'lr': lr},]
    optimizer = torch.optim.Adam(optim_params)

    bar = tqdm(range(PARAM_EPOCHS))

    # masks24_tensor = torch.from_numpy(masks24).float().cuda()
    loss_init = None
    for epoch in bar:
        loss = 0
        for idx in range(PARAM_DEPTH):
            pcd_np = verts_all[idx]
            mask24 = masks24[idx]
            mask15 = masks15[idx]
            mask34 = masks34[idx]
            mask35 = masks35[idx]

            pcd_torch = torch.from_numpy(pcd_np).float().unsqueeze(0).cuda()
            mask24_tensor = torch.from_numpy(mask24).float().unsqueeze(0).cuda()
            mask15_tensor = torch.from_numpy(mask15).float().unsqueeze(0).cuda()
            mask34_tensor = torch.from_numpy(mask34).float().unsqueeze(0).cuda()
            mask35_tensor = torch.from_numpy(mask35).float().unsqueeze(0).cuda()

            pcd_torch = pcd_torch + transformation
            # pcd_torch = torch.cat((
            #     pcd_torch,
            #     torch.ones(
            #         pcd_torch.shape[0], pcd_torch.shape[1], 1,
            #         device="cuda")), dim=2)
            # pcd_torch = (torch.bmm(pcd_torch, transformation))
            # pcd_torch = pcd_torch[:, :, :3] / pcd_torch[:, :, -1:]

            mtx = torch.from_numpy(params[0]['mtx']).float().cuda().unsqueeze(0)
            rotation = torch.from_numpy(params[0]['rotation']).float().cuda().unsqueeze(0)
            translation = torch.from_numpy(params[0]['translation']).float().cuda().unsqueeze(0)
            pcd_proj_24 = project_points_to_camera_plane(
                pcd_torch, mtx,
                rotation, translation,)
            
            mtx = torch.from_numpy(params[1]['mtx']).float().cuda().unsqueeze(0)
            rotation = torch.from_numpy(params[1]['rotation']).float().cuda().unsqueeze(0)
            translation = torch.from_numpy(params[1]['translation']).float().cuda().unsqueeze(0)
            pcd_proj_15 = project_points_to_camera_plane(
                pcd_torch, mtx,
                rotation, translation,)
            
            mtx = torch.from_numpy(params[2]['mtx']).float().cuda().unsqueeze(0)
            rotation = torch.from_numpy(params[2]['rotation']).float().cuda().unsqueeze(0)
            translation = torch.from_numpy(params[2]['translation']).float().cuda().unsqueeze(0)
            pcd_proj_34 = project_points_to_camera_plane(
                pcd_torch, mtx,
                rotation, translation,)
            
            mtx = torch.from_numpy(params[3]['mtx']).float().cuda().unsqueeze(0)
            rotation = torch.from_numpy(params[3]['rotation']).float().cuda().unsqueeze(0)
            translation = torch.from_numpy(params[3]['translation']).float().cuda().unsqueeze(0)
            pcd_proj_35 = project_points_to_camera_plane(
                pcd_torch, mtx,
                rotation, translation,)

            # pcd_proj_24 = project_3d_to_2d(
            #     params[0]['mtx'], params[0]['dist'],
            #     params[0]['rotation'], params[0]['translation'],
            #     pcd_np)
            
            # visualize_points(
            #     mask24_tensor.squeeze().detach().cpu().numpy(),
            #     pcd_proj_24.squeeze().detach().cpu().numpy())

            loss += chamfer_distance(pcd_proj_24, mask24_tensor, single_directional=True)[0]
            loss += chamfer_distance(pcd_proj_15, mask15_tensor, single_directional=True)[0]
            loss += chamfer_distance(pcd_proj_34, mask34_tensor, single_directional=True)[0]
            loss += chamfer_distance(pcd_proj_35, mask35_tensor, single_directional=True)[0]

        visualize_points(
            mask24_tensor.squeeze().detach().cpu().numpy(),
            pcd_proj_24.squeeze().detach().cpu().numpy(), name="24")
        visualize_points(
            mask15_tensor.squeeze().detach().cpu().numpy(),
            pcd_proj_15.squeeze().detach().cpu().numpy(), name="15")
        visualize_points(
            mask34_tensor.squeeze().detach().cpu().numpy(),
            pcd_proj_34.squeeze().detach().cpu().numpy(), name="34")
        visualize_points(
            mask35_tensor.squeeze().detach().cpu().numpy(),
            pcd_proj_35.squeeze().detach().cpu().numpy(), name="35")

        # loss += torch.abs(1 - torch.det(transformation))[0]

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss = loss.detach().cpu().item()
        if loss_init is None:
            loss_init = loss

        bar.set_description(
            "E: {} L: {:.4f}".format(
                epoch,
                loss,
                # torch.det(transformation).detach().item()
                # transformation.detach()
            )
        )

    print('Loss went from {:.4f} to {:.4f}'.format(loss_init, loss))

    return transformation.squeeze().detach().cpu().numpy()


def store_smpl_parameters(transformation, experiment, subject):
    if not os.path.exists(DIR_OUTPUT):
        os.mkdir(DIR_OUTPUT)

    path = os.path.join(
        DIR_OUTPUT,
        f'trans_mask_{experiment}_{subject}.pkl')
    
    params = {
        'transformation': transformation,
    }

    with open(path, 'wb') as handle:
        pickle.dump(params, handle,
                    protocol=pickle.HIGHEST_PROTOCOL)
        
    print(f'Stored results: {path}')


EXPERIMENT = 'a1'
SUBJECT = 1
VOXEL_SIZE = .005


# TODO: Move the cameras somewhere else
cam24 = 'azure_kinect2_4_calib_snap'
cam15 = 'azure_kinect1_5_calib_snap'
cam14 = 'azure_kinect1_4_calib_snap'
cam34 = 'azure_kinect3_4_calib_snap'
cam35 = 'azure_kinect3_5_calib_snap'
if __name__ == "__main__":
    cache = diskcache.Cache('../calibration/cache')

    cameras = [
        cam24,
        cam15,
        # cam14,
        cam34,
        cam35,   
    ]

    finetuned_extrinsics = load_finetuned_extrinsics()

    for file in sorted(os.listdir(DIR_ORG), reverse=False):
        experiment = file.split('.')[0].split('_')[1]
        
        if experiment != EXPERIMENT:
            continue

        file_path = os.path.join(DIR_ORG, file)
        print(f"Visualizing {file_path}")

        transformation = optimize(
            SUBJECT,
            EXPERIMENT,
        )

        store_smpl_parameters(transformation, experiment, SUBJECT)
