import sys
sys.path.append('../')

import os
import math
import glob
import time
import diskcache
import numpy as np
import open3d as o3d


HALPE_LINES = np.array(
    [(0, 1), (0, 2), (1, 3), (2, 4), (5, 18), (6, 18), (5, 7),
     (7, 9), (6, 8), (8, 10), (17, 18), (18, 19), (19, 11),
     (19, 12), (11, 13), (12, 14), (13, 15), (14, 16), (20, 24),
     (21, 25), (23, 25), (22, 24), (15, 24), (16, 25)])

def visualize_poses(poses):
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window()

    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=300)
    
    geometry = o3d.geometry.PointCloud()
    lines = o3d.geometry.LineSet()

    for i in range(len(poses)):
        keypoints = poses[i].reshape(-1, 3)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(keypoints)
        pcd.paint_uniform_color([1, 0, 0]) # red points

        connections = np.concatenate((HALPE_LINES, HALPE_LINES + 26))
        
        lines.points = o3d.utility.Vector3dVector(keypoints)
        lines.lines = o3d.utility.Vector2iVector(connections)

        geometry.points = pcd.points
        if i == 0:
            vis.add_geometry(geometry)
            vis.add_geometry(lines)
            vis.add_geometry(axis)
        else:
            vis.update_geometry(geometry)
            vis.update_geometry(lines)
            vis.update_geometry(axis)
        
        vis.poll_events()
        vis.update_renderer()

        print(f"Update {i}: {time.time()}")
        time.sleep(.2)




if __name__ == "__main__":
    cache = diskcache.Cache('../calibration/cache')

    cache_process = cache.get('process', {})

    for key in cache_process.keys():
        if "skeleton_3D_smooth" in key:
            print(f"Visualizing {key}")

            poses = cache_process[key]
            poses = np.array(poses)

            if poses.shape[0] == 200:
                visualize_poses(poses)
