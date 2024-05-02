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


DIR_OUTPUT = './extrinsics_mask'
DIR_ORG = '../pose_estimation/keypoints_3d_ba'
DIR_PARMAS_GLOBAL = "./extrinsics_global"
DIR_PARAMS_TRANSFORM = '../pose_estimation/keypoints_3d_pose2smpl/'
DIR_SMPL = '/home/hamid/Documents/phd/footefield/Pose_to_SMPL/fit/output/HALPE/'
PARAM_DEPTH = 20
PARAM_SCALE_MASK = 3
PARAM_EPOCHS = 100


# TODO: Refactor
def get_depth_image(cam_name, experiment, idx):
    cam_num = cam_name[12:15]
    img_depth = '/home/hamid/Documents/phd/footefield/data/AzureKinectRecord_0729/{}/azure_kinect{}/depth/depth{:05d}.png'.format(experiment, cam_num, idx)

    return img_depth


def get_mask_image(cam_name, experiment, idx):
    cam_num = cam_name[12:15]
    img_depth = '/home/hamid/Documents/phd/footefield/data/AzureKinectRecord_0729/{}/azure_kinect{}/mask/mask{:05d}.jpg'.format(experiment, cam_num, idx)

    return img_depth


def get_params_depth(cam, cache):
    mtx = cache['depth_matching'][cam]['mtx_r']
    dist = cache['depth_matching'][cam]['dist_r']
    R = cache['depth_matching'][cam]['rotation']
    T = cache['depth_matching'][cam]['transition']

    extrinsics = np.identity(4, dtype=float)
    extrinsics[:3, :3] = R
    extrinsics[:3, 3] = T.ravel() / 1000

    return mtx, dist, extrinsics


def get_params_color(expriment):
    file = f"keypoints3d_{expriment}_ba.pkl"
    file_path = os.path.join(DIR_ORG, file)
    with open(file_path, 'rb') as handle:
        output = pickle.load(handle)

    params = output['params']

    return params


def get_pcd(cam, subject, experiment, idx, extrinsics, cache):
    img_depth = get_depth_image(cam, experiment, idx)
    mtx, dist, _ = get_params_depth(cam, cache)

    img_depth = cv2.imread(img_depth, -1)
    mapx, mapy = cv2.initUndistortRectifyMap(
        mtx, dist, None, mtx, (640, 576), cv2.CV_32FC2)
    img_depth = cv2.remap(img_depth, mapx, mapy, cv2.INTER_NEAREST)

    depth = o3d.geometry.Image(img_depth)

    intrinsics = o3d.camera.PinholeCameraIntrinsic(
        640, 576, mtx[0, 0], mtx[1, 1], mtx[0, 2], mtx[1, 2])
    pcd = o3d.geometry.PointCloud.create_from_depth_image(
        depth, intrinsics, extrinsics['extrinsics'][cam]['base'])
    pcd = pcd.transform(extrinsics['extrinsics'][cam]['offset'])

    pcd_np = np.asarray(pcd.points)

    start_pts = np.array([[0, 0, 0], [1, -1, 3]])
    pcd_np = np.asarray(pcd.points)
    kmeans = KMeans(n_clusters=2, random_state=47,
                    init=start_pts, n_init=1).fit(pcd_np)
    # TODO: Explain what (subject + 1 % 2) is
    pcd.points = o3d.utility.Vector3dVector(
        pcd_np[kmeans.labels_ == (subject + 1) % 2])

    return pcd


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


def remove_outliers(pointcloud):
    _, ind = pointcloud.remove_statistical_outlier(
        nb_neighbors=16,
        std_ratio=.05)
    pointcloud = pointcloud.select_by_index(ind)
    
    return pointcloud


def filter_area(pcd, volume=1):
    points = np.asarray(pcd.points)
    center = np.mean(points, axis=0)
    points = points[points[:, 0] < center[0] + volume]
    points = points[points[:, 1] < center[1] + volume]
    points = points[points[:, 2] < center[2] + volume]
    points = points[points[:, 0] > center[0] - volume]
    points = points[points[:, 1] > center[1] - volume]
    points = points[points[:, 2] > center[2] - volume]

    pcd.points = o3d.utility.Vector3dVector(points)

    return pcd


def preprocess(pointcloud, voxel_size):
    pointcloud = filter_area(pointcloud)
    
    # pointcloud = remove_outliers(pointcloud)
    
    pointcloud = pointcloud.voxel_down_sample(voxel_size=voxel_size)

    return pointcloud


def load_finetuned_extrinsics():
    extrinsics_finetuned = {}
    for path in glob.glob(os.path.join(DIR_PARMAS_GLOBAL, '*')):
        experiment = path.split('.')[-2].split('_')[-2]
        subject = path.split('.')[-2].split('_')[-1]
        with open(path, 'rb') as handle:
            params = pickle.load(handle)

        extrinsics_finetuned[experiment + '_' + subject] = params

    return extrinsics_finetuned


def get_corresponding_files(path):
    file_name = path.split('/')[-1].split('.')[0]

    files = [
        (file_name + '_0_normalized_params.pkl', file_name + '_0_params.pkl'),
        (file_name + '_1_normalized_params.pkl', file_name + '_1_params.pkl'),
    ]

    return files


def project_points_to_camera_plane(points_3d, mtx, R, T):
    transformation = torch.eye(4).cuda()
    transformation[:3, :3] = R
    transformation[:3, 3] = T / 1000
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


def get_pcds(subject, experiment, extrinsics, voxel_size, cache):
    pcds = []

    print("Loading pcds...")
    for idx in tqdm(range(PARAM_DEPTH)):
        pcd = get_pcd(cam24, subject, experiment,
                      idx, extrinsics[experiment + '_' + str(subject)], cache)
        pcd += get_pcd(cam15, subject, experiment,
                       idx, extrinsics[experiment + '_' + str(subject)], cache)
        # pcd14 = get_pcd(cam14, subject, experiment,
        #                 idx, extrinsics[?], params)
        pcd += get_pcd(cam34, subject, experiment,
                       idx, extrinsics[experiment + '_' + str(subject)], cache)
        pcd += get_pcd(cam35, subject, experiment,
                       idx, extrinsics[experiment + '_' + str(subject)], cache)
        
        pcd = pcd.transform(extrinsics[experiment + '_' + str(subject)]['global'])

        pcd = preprocess(pcd, voxel_size=voxel_size)

        pcd_np = np.asarray(pcd.points)
        pcds.append(pcd_np)

    return pcds


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


def visualize_points(points, points2, name="frame", writer=None):
    img = np.zeros((1080, 1920, 3), np.uint8)
    
    for point in points:
        x = int(point[0])
        y = int(point[1])
        if 0 < x < 1920 and 0 < y < 1080:
            img[y, x] = (255, 255, 255)

    for point in points2:
        x = int(point[0])
        y = int(point[1])
        if 0 < x < 1920 and 0 < y < 1080:
            img[y, x] = (0, 255, 0)

    # cv2.imshow(name, cv2.resize(img, (960, 540)))
    # cv2.waitKey(10)

    writer.write(img)


def optimize(subject, experiment, extrinsics, voxel_size, cache):
    params = get_params_color(experiment)
    pcds = get_pcds(subject, experiment, extrinsics, voxel_size, cache)
    masks24, masks15, masks34, masks35 = get_masks(experiment, params)

    # transformation = torch.eye(4).to("cuda").unsqueeze(0)
    transformation = torch.zeros(3).to("cuda").unsqueeze(0)
    transformation.requires_grad = True
    lr = 1e-3
    optim_params = [{'params': transformation, 'lr': lr},]
    optimizer = torch.optim.Adam(optim_params)

    bar = tqdm(range(PARAM_EPOCHS))

    writer24 = cv2.VideoWriter(
        "./output_videos_register_mask/24.mp4",
        cv2.VideoWriter_fourcc(*"mp4v"),
        25,
        (data_loader.IMAGE_RGB_WIDTH, data_loader.IMAGE_RGB_HEIGHT))
    writer15 = cv2.VideoWriter(
        "./output_videos_register_mask/15.mp4",
        cv2.VideoWriter_fourcc(*"mp4v"),
        25,
        (data_loader.IMAGE_RGB_WIDTH, data_loader.IMAGE_RGB_HEIGHT))
    writer34 = cv2.VideoWriter(
        "./output_videos_register_mask/34.mp4",
        cv2.VideoWriter_fourcc(*"mp4v"),
        25,
        (data_loader.IMAGE_RGB_WIDTH, data_loader.IMAGE_RGB_HEIGHT))
    writer35 = cv2.VideoWriter(
        "./output_videos_register_mask/35.mp4",
        cv2.VideoWriter_fourcc(*"mp4v"),
        25,
        (data_loader.IMAGE_RGB_WIDTH, data_loader.IMAGE_RGB_HEIGHT))

    # masks24_tensor = torch.from_numpy(masks24).float().cuda()
    loss_init = None
    for epoch in bar:
        loss = 0
        for idx in range(PARAM_DEPTH):
            pcd_np = pcds[idx]
            mask24 = masks24[idx]
            mask15 = masks15[idx]
            mask34 = masks34[idx]
            mask35 = masks35[idx]

            pcd_torch = torch.from_numpy(pcd_np).float().unsqueeze(0).cuda()
            mask24_tensor = torch.from_numpy(mask24).float().unsqueeze(0).cuda()
            mask15_tensor = torch.from_numpy(mask15).float().unsqueeze(0).cuda()
            mask34_tensor = torch.from_numpy(mask34).float().unsqueeze(0).cuda()
            mask35_tensor = torch.from_numpy(mask35).float().unsqueeze(0).cuda()

            pcd_torch = torch.from_numpy(pcd_np).float().unsqueeze(0).cuda()
            mask24_tensor = torch.from_numpy(mask24).float().unsqueeze(0).cuda()

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
            #     params[0]['rotation'], params[0]['translation'] / 1000,
            #     pcd_np)
            
            # visualize_points(
            #     mask24_tensor.squeeze().detach().cpu().numpy(),
            #     pcd_proj_24.squeeze().detach().cpu().numpy())

            loss += chamfer_distance(pcd_proj_24, mask24_tensor, single_directional=True, norm=1)[0]
            loss += chamfer_distance(pcd_proj_15, mask15_tensor, single_directional=True, norm=1)[0]
            loss += chamfer_distance(pcd_proj_34, mask34_tensor, single_directional=True, norm=1)[0]
            loss += chamfer_distance(pcd_proj_35, mask35_tensor, single_directional=True, norm=1)[0]

        visualize_points(
            mask24_tensor.squeeze().detach().cpu().numpy(),
            pcd_proj_24.squeeze().detach().cpu().numpy(), name="24",
            writer=writer24)
        visualize_points(
            mask15_tensor.squeeze().detach().cpu().numpy(),
            pcd_proj_15.squeeze().detach().cpu().numpy(), name="15",
            writer=writer15)
        visualize_points(
            mask34_tensor.squeeze().detach().cpu().numpy(),
            pcd_proj_34.squeeze().detach().cpu().numpy(), name="34",
            writer=writer34)
        visualize_points(
            mask35_tensor.squeeze().detach().cpu().numpy(),
            pcd_proj_35.squeeze().detach().cpu().numpy(), name="35",
            writer=writer35)

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


EXPERIMENT = 'a2'
SUBJECT = 0
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
            finetuned_extrinsics,
            VOXEL_SIZE,
            cache
        )

        store_smpl_parameters(transformation, experiment, SUBJECT)

