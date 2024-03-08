import sys
sys.path.append('../')

import os
import glob
import time
import numpy as np
import open3d as o3d

from utils import data_loader


STORE_DIR = './keypoints_3d_pose2smpl'
HALPE_LINES = np.array(
    [(0, 1), (0, 2), (1, 3), (2, 4), (5, 18), (6, 18), (5, 7),
     (7, 9), (6, 8), (8, 10), (17, 18), (18, 19), (19, 11),
     (19, 12), (11, 13), (12, 14), (13, 15), (14, 16), (20, 24),
     (21, 25), (23, 25), (22, 24), (15, 24), (16, 25)])

def visualize_poses(poses):
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window()
    vis.get_render_option().show_coordinate_frame = True
    vis.get_render_option().background_color = data_loader.COLOR_SPACE_GRAY
    
    origin = o3d.geometry.TriangleMesh().create_coordinate_frame(1.0)
    geometry = o3d.geometry.PointCloud()
    lines = o3d.geometry.LineSet()
    for idx in range(len(poses)):
        keypoints = poses[idx].reshape(-1, 3)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(keypoints)
        pcd.paint_uniform_color([0, 1, 0]) # Blue points

        connections = np.concatenate((HALPE_LINES, HALPE_LINES + 26))
        
        lines.points = o3d.utility.Vector3dVector(keypoints)
        lines.lines = o3d.utility.Vector2iVector(connections)
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

        print(f"Update {idx}: {time.time()}")
            

if __name__ == "__main__":
    files = glob.glob(os.path.join(STORE_DIR, "*.npy"))
    for file in files:
        print(f"Visualizing {file}")

        with open(file, 'rb') as handle:
            poses = np.load(handle)

        visualize_poses(poses)
