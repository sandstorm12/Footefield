import sys
sys.path.append('../')

import time
import torch
import smplx
import numpy as np
import open3d as o3d

from utils import data_loader


VIS_JTR = True
VIS_MESH = True


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


# TODO: Too complicated, refactor please
def visualize_poses(poses_smpl, verts, faces):
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window()
    vis.get_render_option().background_color = data_loader.COLOR_SPACE_GRAY
    origin = o3d.geometry.TriangleMesh().create_coordinate_frame(size=.1)
    vis.add_geometry(origin)
    
    geometry_jtr = o3d.geometry.PointCloud()
    lines_jtr = o3d.geometry.LineSet()
    mesh = o3d.geometry.TriangleMesh()
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    mesh_line = o3d.geometry.LineSet()
    
    if VIS_JTR:
        keypoints_smpl = poses_smpl.reshape(-1, 3)

        for idx, keypoint_smpl in enumerate(keypoints_smpl):
            label = o3d.t.geometry.TriangleMesh.create_text(
                str(idx), depth=1).to_legacy()
            label.paint_uniform_color((0, 1, 0))

            scale = .0001
            location = keypoint_smpl
            label.transform([[scale, 0, 0, location[0]],
                             [0, scale, 0, location[1]],
                             [0, 0, scale, location[2]],
                             [0, 0, 0, 1]])
            vis.add_geometry(label)

        pcd_smpl = o3d.geometry.PointCloud()
        pcd_smpl.points = o3d.utility.Vector3dVector(keypoints_smpl)
        pcd_smpl.paint_uniform_color([1, 1, 1])
        
        lines_jtr.points = pcd_smpl.points
        lines_jtr.lines = o3d.utility.Vector2iVector(
            JOINTS_SMPLX)
        lines_jtr.paint_uniform_color([1, 1, 1]) # White lines
        geometry_jtr.points = pcd_smpl.points
        geometry_jtr.colors = pcd_smpl.colors
        vis.add_geometry(geometry_jtr)
        vis.add_geometry(lines_jtr)

    if VIS_MESH:
        mesh.vertices = o3d.utility.Vector3dVector(verts)
        mesh_line_temp = o3d.geometry.LineSet.create_from_triangle_mesh(
            mesh)
        mesh_line.points = mesh_line_temp.points
        mesh_line.lines = mesh_line_temp.lines
        vis.add_geometry(mesh_line)

    vis.update_renderer()
    vis.run()
    vis.destroy_window()


# TODO: Shorten
if __name__ == "__main__":
    model_path = 'models'
    model = smplx.create(
        model_path, model_type='smplx',
        gender='neutral', use_face_contour=False,
        num_betas=10,
        num_expression_coeffs=10,
        ext='npz')

    body = torch.rand([1, 21, 3], dtype=torch.float32) * .1
    left_hand_pose = torch.rand([1, 6], dtype=torch.float32) * 1
    right_hand_pose = torch.rand([1, 6], dtype=torch.float32) * 1
    betas = torch.rand([1, 10], dtype=torch.float32) * 1
    expression = torch.rand([1, 10], dtype=torch.float32) * 1

    output = model(body_pose=body, left_hand_pose=left_hand_pose,
                   right_hand_pose=right_hand_pose,
                   betas=betas, expression=expression, return_verts=True)

    vertices = output.vertices.detach().cpu().numpy().squeeze()
    joints = output.joints.detach().cpu().numpy().squeeze()

    vertices -= joints[0]
    joints -= joints[0]
    

    print('Vertices shape =', vertices.shape)
    print('Joints shape =', joints.shape)

    poses_smpl = joints
    verts = vertices
    faces = model.faces

    # Visualize
    visualize_poses(poses_smpl, verts, faces)
