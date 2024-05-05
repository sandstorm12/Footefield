import sys
sys.path.append('../')

import os
import cv2
import torch
import smplx
import pickle
import diskcache
import numpy as np

from tqdm import tqdm
from utils import data_loader


VIS_MESH = True
VIS_ORG = True
VIS_JTR = True

DIR_SMPLX = './params_smplx_mask'
DIR_PARAMS = '../pose_estimation/keypoints_3d_pose2smpl_x/'
DIR_STORE_ORG = '../pose_estimation/keypoints_3d_ba_x'
DIR_OUTPUT = "./output_videos_mask"
PATH_MODEL = 'models'

PARAM_OUTPUT_SIZE = (1920, 1080)
PARAM_OUTPUT_FPS = 5.0

TYPE_ORG = "org"
TYPE_JTR = "jtr"
TYPE_MESH = "mesh"

body_foot_skeleton = [
    (16, 14), (14, 12), (17, 15), (15, 13), (12, 13), (6, 12), (7, 13),
    (6, 7), (6, 8), (7, 9), (8, 10), (9, 11), (2, 3), (1, 2), (1, 3),
    (2, 4), (3, 5), (4, 6), (5, 7),
    (16, 20), (16, 19), (16, 18),    # left foot
    (17, 23), (17, 21), (17, 22)     # right foot
]
face_skeleton = [
    (25,5), (39,4), # ear to ear body
    (54, 1), #nose to nose body
    (60, 3), (3, 63), (66, 2), (2, 69), # eyes to eyes body 
    ] + [(x,x+1) for x in range(24, 40)] + [ #face outline
    (24,41), (41,42), (42,43), (43,44), (44,45), (45,51), #right eyebrow
    (40,50), (50,49), (49,48), (48,47), (47,46), (46,51), #left eyebrow
    (24,60), (60,61), (61,62), (62,63), (63,51), (63,64), (64,65), (65,60), #right eye
    (40,69), (69,68), (68,67), (67,66), (66,51), (66,71), (71,70), (70,69), #left eye
    ] + [(x,x+1) for x in range(51, 59)] + [ (59, 54), #nose
    (57, 75), (78,36), (72, 28), (72,83)] + [(x,x+1) for x in range(72, 83)] + [ # mouth outline
    (72, 84), (84,85), (85,86), (86,87), (87,88), (88,78), #upper lip
    (72, 91), (91,90), (90,89), (89,78) #lower lip
    ]
                                                                                
lefthand_skeleton = [
    (92, 10), #connect to wrist
    (92,93), (92, 97), (92,101), (92,105), (92, 109) #connect to finger starts
    ] + [(x,x+1) for s in [93,97,101,105,109] for x in range(s, s+3)] #four finger                                                                         

righthand_skeleton = [
    (113, 11), #connect to wrist
    (113,114), (113, 118), (113,122), (113,126), (113, 130) #connect to finger starts
    ] + [(x,x+1) for s in [114,118,122,126,130] for x in range(s, s+3)] #four finger                                                                      

WHOLEBODY_SKELETON = body_foot_skeleton + face_skeleton + lefthand_skeleton + righthand_skeleton
HALPE_LINES = np.array(WHOLEBODY_SKELETON) - 1

JOINTS_SMPLX = np.array([
    (0, 1), (0, 2), (0, 3), (1, 4), (2, 5), (3, 6), (4, 7),
    (5, 8), (6, 9), (7, 10), (8, 11), (9, 12), (9, 13),
    (9, 14), (12, 15), (13, 16), (14, 17), (16, 18), (17, 19),
    (18, 20),(19, 21), # Body
    (21, 52), (52, 53), (53, 54), (54, 71),
    (21, 40), (40, 41), (41, 42), (42, 72),
    (21, 43), (43, 44), (44, 45), (45, 73),
    (21, 49), (49, 50), (50, 51), (51, 74),
    (21, 46), (46, 47), (47, 48), (48, 75), # Right hand
    (20, 37), (37, 38), (38, 39), (39, 66),
    (20, 25), (25, 26), (26, 27), (27, 67),
    (20, 28), (28, 29), (29, 30), (30, 68),
    (20, 34), (34, 35), (35, 36), (36, 69),
    (20, 31), (31, 32), (32, 33), (33, 70), # Left hand
    (8, 65), (11, 63), (11, 64), # Right foot
    (7, 62), (10, 60), (10, 61), # Left foot
    (15, 58), (15, 59), # Head
    (86, 87), (87, 88), (88, 89), (89, 55),
    (90, 91), (91, 92), (92, 93), (93, 94), # Nose
    (95, 96), (96, 97), (97, 98), (98, 99), (99, 100),
    (101, 102), (102, 103), (103, 104), (104, 105), (105, 106), # Eyes
    (76, 77), (77, 78), (78, 79), (79, 80),
    (81, 82), (82, 83), (83, 84), (84, 85), # Eyebrows
    (107, 108), (108, 109), (109, 110), (110, 111), (111, 112),
    (112, 113), (113, 114), (114, 115), (115, 116), (116, 117),
    (117, 118), # Outer lips
    (119, 120), (120, 121), (121, 122), (122, 123), (123, 124),
    (124, 125), (125, 126), # Inner lips
])


def get_parameters(params):
    mtx = params['mtx']
    dist = params['dist']
    rotation = params['rotation']
    translation = params['translation']

    extrinsics = np.zeros((3, 4), dtype=float)
    extrinsics[:3, :3] = rotation
    extrinsics[:3, 3] = translation

    return mtx, dist, extrinsics


# Implemented by Gemini
def project_3d_to_2d(camera_matrix, dist_coeffs, rvec, tvec, object_points):
    image_points, _ = cv2.projectPoints(object_points, rvec, tvec,
                                        camera_matrix, dist_coeffs)

    image_points = image_points.squeeze()

    return image_points


def get_video_writer(experiment, camera):
    if not os.path.exists(DIR_OUTPUT):
        os.mkdir(DIR_OUTPUT)

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    writer = cv2.VideoWriter(
        os.path.join(
            DIR_OUTPUT,
            f'visualizer_skeleton_video_{experiment}_{camera}.avi'
        ),
        fourcc,
        PARAM_OUTPUT_FPS,
        PARAM_OUTPUT_SIZE
    )
    
    return writer


# TODO: Make types constant
def get_connections_by_type(type):
    if type == TYPE_ORG:
        connections = np.concatenate(
            (np.array(HALPE_LINES),
             np.array(HALPE_LINES) + 133))
    elif type == TYPE_JTR:
        connections = np.concatenate(
            (np.array(JOINTS_SMPLX),
             np.array(JOINTS_SMPLX) + 127))
    elif type == TYPE_MESH:
        connections = None
    else:
        raise Exception("Unknown type.")
        
    return connections


def get_point_size_by_type(type):
    if type == TYPE_ORG:
        point_size = 3
    elif type == TYPE_JTR:
        point_size = 3
    elif type == TYPE_MESH:
        point_size = 1
    else:
        raise Exception("Unknown type.")
        
    return point_size
    

def write_frame(img_rgb, poses_2d, type):
    point_size = get_point_size_by_type(type)
    for point in poses_2d:
        cv2.circle(img_rgb, (int(point[0]), int(point[1])),
                    point_size, (0, 255, 0), -1)

    connections = get_connections_by_type(type)
    if connections is not None:
        for connection in connections:
            cv2.line(img_rgb,
                    (int(poses_2d[connection[0]][0]),
                     int(poses_2d[connection[0]][1])),
                    (int(poses_2d[connection[1]][0]),
                     int(poses_2d[connection[1]][1])),
                    (255, 255, 255), 1)


def poses_3d_2_2d(poses_3d, params):
    poses_shape = list(poses_3d.shape)
    poses_shape[-1] = 2
    
    mtx = params['mtx']
    dist = params['dist']
    rotation = params['rotation']
    translation = params['translation']
    poses_2d = project_3d_to_2d(
        mtx, None,
        rotation,
        translation,
        poses_3d.reshape(-1, 3))
    poses_2d[:, 1] = poses_2d[:, 1]
    poses_2d = poses_2d.reshape(poses_shape)

    return poses_2d


def get_corresponding_files(experiment):
    files = [
        (f'params_smplx_{experiment}_0.pkl', f'keypoints3d_{experiment}_ba_0_params.pkl'),
        (f'params_smplx_{experiment}_1.pkl', f'keypoints3d_{experiment}_ba_1_params.pkl'),
    ]

    return files


# TODO: Shorten
def get_smplx_parameters(experiment, smplx_model, device):
    files_smpl = get_corresponding_files(experiment)
        
    poses_smpl_all = []
    verts_all = []
    faces_all = []
    for file_smpl in files_smpl:
        # Load SMPL data
        path_smplx = os.path.join(DIR_SMPLX, file_smpl[0])
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
        translation_smplx = smplx_params['translation']
        scale_smplx = smplx_params['scale']

        output = smplx_model(
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
        origin = np.copy(joints[0, 0])
        faces = smplx_model.faces

        # Load alignment params
        path_params = os.path.join(DIR_PARAMS, file_smpl[1])
        with open(path_params, 'rb') as handle:
            params = pickle.load(handle)
        rotation = params['rotation']
        scale = params['scale'] * scale_smplx
        translation = params['translation']

        rotation_inverted = np.linalg.inv(rotation)
        
        joints = joints - origin + translation_smplx
        joints = joints.dot(rotation_inverted.T)
        joints = joints * scale
        joints = joints + translation

        verts = verts - origin + translation_smplx
        verts = verts.dot(rotation_inverted.T)
        verts = verts * scale
        verts = verts + translation

        poses_smpl_all.append(joints)
        verts_all.append(verts)
        faces_all.append(faces)

    poses_smpl_all = np.array(poses_smpl_all)
    verts_all = np.array(verts_all)
    faces_all = np.array(faces_all)

    poses_smpl_all = np.transpose(poses_smpl_all, (1, 0, 2, 3))
    verts_all = np.transpose(verts_all, (1, 0, 2, 3))

    return poses_smpl_all, verts_all, faces_all


# TODO: Move the cameras somewhere else
cam24 = 'azure_kinect2_4_calib_snap'
cam15 = 'azure_kinect1_5_calib_snap'
cam14 = 'azure_kinect1_4_calib_snap'
cam34 = 'azure_kinect3_4_calib_snap'
cam35 = 'azure_kinect3_5_calib_snap'

# TODO: Too long
if __name__ == "__main__":
    cache = diskcache.Cache('../calibration/cache')

    cameras = [
        cam24,
        cam15,
        # cam14,
        cam34,
        cam35
    ]

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = smplx.create(
        PATH_MODEL, model_type='smplx',
        gender='neutral', use_face_contour=False,
        num_betas=10, use_pca=False,
        num_expression_coeffs=10,
        ext='npz').to(device)

    for file in os.listdir(DIR_STORE_ORG):
        experiment = file.split('.')[-2].split('_')[-2]

        file_path = os.path.join(DIR_STORE_ORG, file)
        print(f"Visualizing {file_path}")
        
        with open(file_path, 'rb') as handle:
            output = pickle.load(handle)

        poses = output['points_3d'].reshape(-1, 2, 133, 3)
        params = output['params']

        poses_smpl_all, verts_all, faces_all = get_smplx_parameters(experiment, model, device)

        for idx_cam, camera in enumerate(tqdm(cameras)):
            dir = data_loader.EXPERIMENTS[experiment][camera]

            img_rgb_paths = data_loader.list_rgb_images(
                os.path.join(dir, "color"))
            
            poses_2d = poses_3d_2_2d(
                poses,
                params[idx_cam]).reshape(poses.shape[0], -1, 2)
            poses_2d_smpl = poses_3d_2_2d(
                poses_smpl_all,
                params[idx_cam]).reshape(poses.shape[0], -1, 2)
            poses_2d_verts = poses_3d_2_2d(
                verts_all,
                params[idx_cam]).reshape(poses.shape[0], -1, 2)

            writer = get_video_writer(experiment, camera)
            for idx, t in enumerate(poses_2d.reshape(poses_2d.shape[0], -1, 2)):
                img_rgb = cv2.imread(img_rgb_paths[idx])
                mtx, dist, _ = get_parameters(params[idx_cam])
                img_rgb = cv2.undistort(img_rgb, mtx, dist, None, None)
                # if VIS_ORG:
                #     write_frame(img_rgb, poses_2d[idx],
                #                 TYPE_ORG)

                if VIS_JTR:
                    write_frame(img_rgb, poses_2d_smpl[idx],
                                TYPE_JTR)

                if VIS_MESH:
                    write_frame(img_rgb, poses_2d_verts[idx],
                                TYPE_MESH)
                    
                writer.write(img_rgb)
