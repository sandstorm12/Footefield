import sys
sys.path.append('../')

import time
import yaml
import torch
import smplx
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


# TODO: Shorten
def visualize_poses(poses, global_orient, jaw_pose, leye_pose,
                    reye_pose, body, left_hand_pose,
                    right_hand_pose, betas, expression,
                    translation, scale, faces, configs):
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

    verts = verts - joints[0, 0] + np.expand_dims(translation, axis=1)
    verts = verts * scale
    verts = verts.squeeze()

    joints = joints - joints[0, 0] + np.expand_dims(translation, axis=1)
    joints = joints * scale
    joints = joints.squeeze()

    geometry_poses = o3d.geometry.PointCloud()
    lines_poses = o3d.geometry.LineSet()
    geometry_joints = o3d.geometry.PointCloud()
    lines_joints = o3d.geometry.LineSet()
    mesh = o3d.geometry.TriangleMesh()
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    mesh_line = o3d.geometry.LineSet()
    for idx in range(len(verts)):
        if configs['visualize_poses']:
            keypoints = poses[idx].reshape(-1, 3)
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(keypoints)
            pcd.paint_uniform_color([0, 1, 0]) # Blue points

            connections = np.array(HALPE_LINES)
            
            lines_poses.points = o3d.utility.Vector3dVector(keypoints)
            lines_poses.lines = o3d.utility.Vector2iVector(connections)
            lines_poses.paint_uniform_color([1, 1, 1]) # White lines

            geometry_poses.points = pcd.points
            geometry_poses.colors = pcd.colors
            if idx == 0:
                vis.add_geometry(geometry_poses)
                vis.add_geometry(lines_poses)
            else:
                vis.update_geometry(geometry_poses)
                vis.update_geometry(lines_poses)

        if configs['visualize_joints']:
            keypoints = joints[idx].reshape(-1, 3)
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(keypoints)
            pcd.paint_uniform_color([0, 1, 0]) # Blue points

            connections = np.array(JOINTS_SMPLX)
            
            lines_joints.points = o3d.utility.Vector3dVector(keypoints)
            lines_joints.lines = o3d.utility.Vector2iVector(connections)
            lines_joints.paint_uniform_color([1, 1, 1]) # White lines

            geometry_joints.points = pcd.points
            geometry_joints.colors = pcd.colors
            if idx == 0:
                vis.add_geometry(geometry_joints)
                vis.add_geometry(lines_joints)
            else:
                vis.update_geometry(geometry_joints)
                vis.update_geometry(lines_joints)

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
            
        delay = .1
        control_gaps = 10
        for _ in range(control_gaps):
            vis.poll_events()
            vis.update_renderer()
            time.sleep(delay / control_gaps)


def load_smplx_params(smplx_params):
    global_orient = np.array(smplx_params['global_orient'], np.float32)
    jaw_pose = np.array(smplx_params['jaw_pose'], np.float32)
    leye_pose = np.array(smplx_params['leye_pose'], np.float32)
    reye_pose = np.array(smplx_params['reye_pose'], np.float32)
    body = np.array(smplx_params['body'], np.float32)
    left_hand_pose = np.array(smplx_params['left_hand_pose'], np.float32)
    right_hand_pose = np.array(smplx_params['right_hand_pose'], np.float32)
    betas = np.array(smplx_params['betas'], np.float32)
    expression = np.array(smplx_params['expression'], np.float32)
    translation = np.array(smplx_params['translation'], np.float32)
    scale = np.array(smplx_params['scale'], np.float32)

    return global_orient, jaw_pose, leye_pose, reye_pose, body, \
        left_hand_pose, right_hand_pose, betas, expression, \
        translation, scale


if __name__ == '__main__':
    args = _get_arguments()
    configs = _load_configs(args.config)

    print(f"Config loaded: {configs}")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = smplx.create(
        configs['models_root'], model_type='smplx',
        gender=configs['gender'], use_face_contour=False,
        num_betas=10, use_pca=False,
        num_expression_coeffs=10,
        ext='npz').to(device)
    
    with open(configs['params_smplx'], 'rb') as handle:
        params_smpl = yaml.safe_load(handle)

    with open(configs['skeletons_norm'], 'rb') as handle:
        poses = np.array(yaml.safe_load(handle))

    for idx_person, person in enumerate(params_smpl):
        global_orient, jaw_pose, leye_pose, reye_pose, body, \
            left_hand_pose, right_hand_pose, betas, expression, \
            translation, scale = \
                load_smplx_params(person)
        
        poses_person = (np.array(poses[idx_person]['pose_normalized']))

        visualize_poses(
            poses_person, global_orient, jaw_pose, leye_pose,
            reye_pose,body,left_hand_pose,
            right_hand_pose, betas, expression,
            translation, scale, model.faces, configs)
