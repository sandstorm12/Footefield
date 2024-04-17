import sys
sys.path.append('../')

import os
import cv2
import pickle
import diskcache
import numpy as np

from tqdm import tqdm
from utils import data_loader


DIR_INPUT = "./keypoints_3d_ba_x"
DIR_OUTPUT = "./output_videos_x"
PARAM_OUTPUT_SIZE = (1920, 1080)
PARAM_OUTPUT_FPS = 5.0

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


def get_parameters(params):
    mtx = params['mtx']
    dist = params['dist']
    rotation = params['rotation']
    translation = params['translation']

    extrinsics = np.identity(4, dtype=float)
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


def write_video(poses_2d, experiment, camera, params, cache):
    img_rgb_paths = data_loader.list_rgb_images(os.path.join(dir, "color"))

    mtx, dist, _ = get_parameters(params)

    writer = get_video_writer(experiment, camera)
    for idx, t in enumerate(poses_2d.reshape(poses_2d.shape[0], -1, 2)):
        img_rgb = cv2.imread(img_rgb_paths[idx])

        img_rgb = cv2.undistort(img_rgb, mtx, dist, None, None)
        for point in t:
            cv2.circle(img_rgb, (int(point[0]), int(point[1])),
                       3, (0, 255, 0), -1)

        connections = np.concatenate(
            (np.array(HALPE_LINES),
             np.array(HALPE_LINES) + 133))
        for connection in connections:
            cv2.line(img_rgb,
                    (int(t[connection[0]][0]), int(t[connection[0]][1])),
                    (int(t[connection[1]][0]), int(t[connection[1]][1])),
                    (255, 255, 255), 1)

        writer.write(img_rgb)


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

    for file in os.listdir(DIR_INPUT):
        experiment = file.split('.')[-2].split('_')[-2]

        file_path = os.path.join(DIR_INPUT, file)
        print(f"Visualizing {file_path}")
        
        with open(file_path, 'rb') as handle:
            output = pickle.load(handle)

        poses = output['points_3d'].reshape(-1, 2, 133, 3)
        params = output['params']
        for idx_cam, camera in enumerate(tqdm(cameras)):
            dir = data_loader.EXPERIMENTS[experiment][camera]

            poses_2d = poses_3d_2_2d(
                poses,
                params[idx_cam])

            write_video(poses_2d, experiment, camera, params[idx_cam], cache)
