import sys
sys.path.append('../')

import torch
import numpy as np
import open3d as o3d

from utils import data_loader
from smplpytorch.pytorch.smpl_layer import SMPL_Layer


VIS_JTR = True
VIS_MESH = True


JOINTS_SMPL = np.array([
    (0, 1), (0, 2), (0, 3), (1, 4), (2, 5), (3, 6), (4, 7),
    (5, 8), (6, 9), (7, 10), (8, 11), (9, 12), (9, 13), (9, 14),
    (12, 15), (13, 16), (14, 17), (16, 18), (17, 19), (18, 20), (19, 21),
    (20, 22), (21, 23),
])


# TODO: Too complicated, refactor please
def visualize_poses(poses_smpl, verts, faces):
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window()
    vis.get_render_option().background_color = data_loader.COLOR_SPACE_GRAY
    vis.get_render_option().show_coordinate_frame = True
    
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

            scale = .005
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
            JOINTS_SMPL)
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
    model = SMPL_Layer(
        center_idx=0,
        gender="neutral",
        model_root='models')

    alphas = torch.zeros([1, 72], dtype=torch.float32)
    betas = torch.zeros([1, 10], dtype=torch.float32)

    vertices, joints = model(alphas, th_betas=betas)

    vertices = vertices.detach().cpu().squeeze().numpy()
    joints = joints.detach().cpu().squeeze().numpy()

    print('Vertices shape =', vertices.shape)
    print('Joints shape =', joints.shape)

    poses_smpl = joints
    verts = vertices
    faces = model.th_faces

    # Visualize
    visualize_poses(poses_smpl, verts, faces)
