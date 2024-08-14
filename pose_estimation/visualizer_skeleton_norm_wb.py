import sys
sys.path.append('../')

import time
import yaml
import argparse
import numpy as np
import open3d as o3d

from utils import data_loader


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
    )

    args = parser.parse_args()

    return args


def _load_configs(path):
    with open(path, 'r') as yaml_file:
        configs = yaml.safe_load(yaml_file)

    return configs


def visualize_poses(poses):
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window()
    vis.get_render_option().show_coordinate_frame = True
    vis.get_render_option().background_color = data_loader.COLOR_SPACE_GRAY
    
    origin = o3d.geometry.TriangleMesh().create_coordinate_frame(.10)
    geometry = o3d.geometry.PointCloud()
    lines = o3d.geometry.LineSet()
    for idx in range(len(poses)):
        keypoints = poses[idx].reshape(-1, 3)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(keypoints)
        pcd.paint_uniform_color([0, 1, 0]) # Blue points
        
        lines.points = o3d.utility.Vector3dVector(keypoints)
        lines.lines = o3d.utility.Vector2iVector(HALPE_LINES)
        lines.paint_uniform_color([1, 1, 1]) # White lines

        geometry.points = pcd.points
        geometry.colors = pcd.colors
        if idx == 0:
            vis.add_geometry(geometry)
            vis.add_geometry(lines)
            vis.add_geometry(origin)
        else:
            vis.update_geometry(geometry)
            vis.update_geometry(lines)
            
        delay_ms = 50
        for _ in range(delay_ms // 10):
            vis.poll_events()
            vis.update_renderer()
            time.sleep(.01)
            

if __name__ == "__main__":
    args = _get_arguments()
    configs = _load_configs(args.config)

    print(f"Config loaded: {configs}")

    with open(configs['skeletons'], 'rb') as handle:
        bundles = yaml.safe_load(handle)

    for bundle in bundles:
        poses = np.array(bundle['pose_normalized'])

        visualize_poses(poses)
