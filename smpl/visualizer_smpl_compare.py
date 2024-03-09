import sys
sys.path.append('../')

import os
import glob
import time
import pickle
import numpy as np
import open3d as o3d

from utils import data_loader


VIS_MESH = True
VIS_ORG = True
VIS_JTR = True

DIR_STORE = '/home/hamid/Documents/phd/footefield/Pose_to_SMPL/fit/output/HALPE/'
DIR_PARAMS = '../pose_estimation/keypoints_3d_pose2smpl/'
DIR_STORE_ORG = '../pose_estimation/keypoints_3d_ba'

JOINTS_SMPL = np.array([
    (0, 1), (0, 2), (0, 3), (1, 4), (2, 5), (3, 6), (4, 7),
    (5, 8), (6, 9), (7, 10), (8, 11), (9, 12), (9, 13), (9, 14),
    (12, 15), (13, 16), (14, 17), (16, 18), (17, 19), (18, 20), (19, 21),
    (20, 22), (21, 23),
])

HALPE_LINES = np.array(
    [(0, 1), (0, 2), (1, 3), (2, 4), (5, 18), (6, 18), (5, 7),
     (7, 9), (6, 8), (8, 10), (17, 18), (18, 19), (19, 11),
     (19, 12), (11, 13), (12, 14), (13, 15), (14, 16), (20, 24),
     (21, 25), (23, 25), (22, 24), (15, 24), (16, 25)])


def visualize_poses(poses_org, poses_smpl, verts, faces):
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window()
    vis.get_render_option().background_color = data_loader.COLOR_SPACE_GRAY
    vis.get_render_option().show_coordinate_frame = True
    
    geometry_org = o3d.geometry.PointCloud()
    lines_org = o3d.geometry.LineSet()
    geometry_jtr = o3d.geometry.PointCloud()
    lines_jtr = o3d.geometry.LineSet()
    mesh = o3d.geometry.TriangleMesh()
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    mesh_line = o3d.geometry.LineSet()
    
    for idx in range(len(poses_org)):
        if VIS_ORG:
            keypoints_org = poses_org[idx].reshape(-1, 3)
            pcd_org = o3d.geometry.PointCloud()
            pcd_org.points = o3d.utility.Vector3dVector(keypoints_org)
            pcd_org.paint_uniform_color([0, 1, 0])
            lines_org.points = pcd_org.points
            lines_org.lines = o3d.utility.Vector2iVector(HALPE_LINES)
            lines_org.paint_uniform_color([0, 1, 0]) # White lines
            geometry_org.points = pcd_org.points
            geometry_org.colors = pcd_org.colors
            if idx == 0:
                vis.add_geometry(geometry_org)
                vis.add_geometry(lines_org)
            else:
                vis.update_geometry(geometry_org)
                vis.update_geometry(lines_org)

        if VIS_JTR:
            keypoints_smpl = poses_smpl[idx].reshape(-1, 3)
            pcd_smpl = o3d.geometry.PointCloud()
            pcd_smpl.points = o3d.utility.Vector3dVector(keypoints_smpl)
            pcd_smpl.paint_uniform_color([1, 0, 0])
            lines_jtr.points = pcd_smpl.points
            lines_jtr.lines = o3d.utility.Vector2iVector(JOINTS_SMPL)
            lines_jtr.paint_uniform_color([0, 0, 1]) # White lines
            geometry_jtr.points = pcd_smpl.points
            geometry_jtr.colors = pcd_smpl.colors
            if idx == 0:
                vis.add_geometry(geometry_jtr)
                vis.add_geometry(lines_jtr)
            else:
                vis.update_geometry(geometry_jtr)
                vis.update_geometry(lines_jtr)

        if VIS_MESH:
            mesh.vertices = o3d.utility.Vector3dVector(verts[idx])
            mesh_line_temp = o3d.geometry.LineSet.create_from_triangle_mesh(
                mesh)
            mesh_line.points = mesh_line_temp.points
            mesh_line.lines = mesh_line_temp.lines
            if idx == 0:
                vis.add_geometry(mesh_line)
            else:
                vis.update_geometry(mesh_line)
            
        delay_ms = 50
        for _ in range(delay_ms // 10):
            vis.poll_events()
            vis.update_renderer()
            time.sleep(.01)

        print(f"Update {idx}: {time.time()}")


def get_corresponding_files(path):
    file_name = path.split('/')[-1].split('.')[0]

    files = [
        (file_name + '_0_normalized_params.pkl', file_name + '_0_params.pkl'),
        (file_name + '_1_normalized_params.pkl', file_name + '_1_params.pkl'),
    ]

    return files


if __name__ == "__main__":
    files_org = glob.glob(os.path.join(DIR_STORE_ORG, "*.pkl"))
    for file_org in files_org:
        with open(file_org, 'rb') as handle:
            output = pickle.load(handle)

        poses_org = output['points_3d'].reshape(-1, 2, 26, 3)

        files_smpl = get_corresponding_files(file_org)
        for idx_smpl, file_smpl in enumerate(files_smpl):
            path_smpl = os.path.join(DIR_STORE, file_smpl[0])
            with open(path_smpl, 'rb') as handle:
                smpl = pickle.load(handle)
            poses_smpl = np.array(smpl['Jtr'])
            verts = np.array(smpl['verts'])
            faces = np.array(smpl['th_faces'])
            scale_smpl = smpl['scale']
            translation_smpl = smpl['translation']

            path_params = os.path.join(DIR_PARAMS, file_smpl[1])
            with open(path_params, 'rb') as handle:
                params = pickle.load(handle)
            rotation = params['rotation']
            scale = params['scale'] * scale_smpl
            translation = params['translation']

            rotation_inverted = np.linalg.inv(rotation)
            poses_smpl = poses_smpl + translation_smpl
            poses_smpl = poses_smpl.dot(rotation_inverted.T)
            poses_smpl = poses_smpl * scale
            poses_smpl = poses_smpl + translation

            verts = verts + translation_smpl
            verts = verts.dot(rotation_inverted.T)
            verts = verts * scale
            verts = verts + translation

            visualize_poses(poses_org[:, idx_smpl], poses_smpl, verts, faces)
