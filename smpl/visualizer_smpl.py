import sys
sys.path.append('../')

import cv2
import time
import pickle
import diskcache
import numpy as np
import open3d as o3d

from utils import data_loader
from calibration import rgb_depth_map


JOINTS_SMPL = np.array([
    (0, 1),
    (0, 2),
    (0, 3),
    (1, 4),
    (2, 5),
    (3, 6),
    (4, 7),
    (5, 8),
    (6, 9),
    (7, 10),
    (8, 11),
    (9, 12),
    (9, 13),
    (9, 14),
    (12, 15),
    (13, 16),
    (14, 17),
    (16, 18),
    (17, 19),
    (18, 20),
    (19, 21),
    (20, 22),
    (21, 23),
])


def visualize_poses(poses, verts, faces):
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window()
    vis.get_render_option().background_color = data_loader.COLOR_SPACE_GRAY
    vis.get_render_option().show_coordinate_frame = True
    
    geometry = o3d.geometry.PointCloud()
    lines = o3d.geometry.LineSet()
    mesh = o3d.geometry.TriangleMesh()
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    for idx in range(len(poses)):
        keypoints = poses[idx].reshape(-1, 3)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(keypoints)
        pcd.paint_uniform_color([0, 1, 0]) # Blue points

        connections = JOINTS_SMPL
        
        lines.points = o3d.utility.Vector3dVector(keypoints)
        lines.lines = o3d.utility.Vector2iVector(connections)
        lines.paint_uniform_color([1, 1, 1]) # White lines

        print(verts.shape)
        mesh.vertices = o3d.utility.Vector3dVector(verts[idx])
        mesh.compute_vertex_normals()
        mesh.compute_triangle_normals()


        geometry.points = pcd.points
        geometry.colors = pcd.colors
        if idx == 0:
            vis.add_geometry(geometry)
            vis.add_geometry(lines)
            vis.add_geometry(mesh)
        else:
            vis.update_geometry(geometry)
            vis.update_geometry(lines)
            vis.update_geometry(mesh)
            
        delay_ms = 100
        for _ in range(delay_ms // 10):
            vis.poll_events()
            vis.update_renderer()
            time.sleep(.01)

        print(f"Update {idx}: {time.time()}")


if __name__ == "__main__":
    path = '/home/hamid/Documents/phd/footefield/Pose_to_SMPL/fit/output/HALPE/keypoints3d_a1_ba_params.pkl'
    with open(path, 'rb') as handle:
        params = pickle.load(handle)

    print(params.keys())
    print(np.array(params['Jtr']).shape)
    print(np.array(params['th_faces']).shape)

    poses = np.array(params['Jtr'])
    verts = np.array(params['verts'])
    faces = np.array(params['th_faces'])

    visualize_poses(poses, verts, faces)