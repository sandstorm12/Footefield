import sys
sys.path.append('../')

import os
import cv2
import yaml
import argparse
import numpy as np

from tqdm import tqdm
from utils import data_loader


# TODO: Move to configs
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


def _get_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-c', '--config',
        help='Path to the config file',
        type=str,
        default='configs/visualizer_skeleton_2d_video_wb.yml',
    )

    args = parser.parse_args()

    return args


def _load_configs(path):
    with open(path, 'r') as yaml_file:
        configs = yaml.safe_load(yaml_file)

    return configs


def get_video_writer(camera, dir, fps, size):
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    writer = cv2.VideoWriter(
        os.path.join(
            dir,
            f'visualizer_skeleton_2d_wb_{camera}.avi'
        ),
        fourcc,
        5,
        (1920, 1080)
    )
    
    return writer


def write_video(poses, camera, intrinsics, configs):
    dir = configs['calibration_folders'][camera]
    img_rgb_paths = data_loader.list_rgb_images(dir)

    if not os.path.exists(configs['output_dir']):
        os.makedirs(configs['output_dir'])

    mtx = np.array(intrinsics[camera]['mtx'], np.float32)
    dist = np.array(intrinsics[camera]['dist'], np.float32)

    writer = get_video_writer(camera, configs['output_dir'], configs['fps'], configs['size'])
    for idx, t in enumerate(poses.reshape(poses.shape[0], -1, 2)):
        img_rgb = cv2.imread(img_rgb_paths[idx])

        img_rgb = cv2.undistort(img_rgb, mtx, dist, None, None)
        for point in t:
            cv2.circle(img_rgb, (int(point[0]), int(point[1])),
                       3, (0, 255, 0), -1)

        connections = np.concatenate(
            [np.array(HALPE_LINES) + i * 133
             for i in range(poses.shape[1])]
        )
        for connection in connections:
            cv2.line(img_rgb,
                    (int(t[connection[0]][0]), int(t[connection[0]][1])),
                    (int(t[connection[1]][0]), int(t[connection[1]][1])),
                    (255, 255, 255), 1)

        writer.write(img_rgb)


# TODO: Too long
if __name__ == "__main__":
    args = _get_arguments()
    configs = _load_configs(args.config)

    print(f"Config loaded: {configs}")
    
    with open(configs['skeletons']) as handler:
        poses = yaml.safe_load(handler)

    cameras = poses.keys()
    for idx_cam, camera in enumerate(tqdm(cameras)):
        with open(configs['intrinsics']) as handler:
            intrinsics = yaml.safe_load(handler)
        
        poses_cam = np.array(poses[camera]['pose'])

        write_video(poses_cam, camera, intrinsics, configs)
