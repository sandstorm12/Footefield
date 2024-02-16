import sys
sys.path.append('../')

import os
import time
import numpy as np
import open3d as o3d

from smplx import SMPL

from utils import data_loader


def visualize_poses(poses, joints, triangles):
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window()
    
    geometry = o3d.geometry.PointCloud()
    geometry_joints = o3d.geometry.PointCloud()
    geometry_joints.paint_uniform_color([1, 0, 0]) # red points
    lines = o3d.geometry.LineSet()
    geometry_mesh = o3d.geometry.TriangleMesh()

    for i in range(len(poses)):
        keypoints = poses[i].reshape(-1, 3)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(keypoints)

        keypoints_joints = joints[i].reshape(-1, 3)
        keypoints_joints[:, 2] += 1000
        pcd_joints = o3d.geometry.PointCloud()
        pcd_joints.points = o3d.utility.Vector3dVector(keypoints_joints)

        connections = np.array(data_loader.MBERT_EDGES)
        
        lines.points = o3d.utility.Vector3dVector(keypoints_joints)
        lines.lines = o3d.utility.Vector2iVector(connections)    

        geometry.points = pcd.points
        geometry_joints.points = pcd_joints.points
        geometry_mesh.vertices = pcd.points
        geometry_mesh.triangles = o3d.utility.Vector3iVector(triangles)
        geometry_mesh.compute_vertex_normals()
        geometry_mesh.compute_triangle_normals()

        if i == 0:
            # vis.add_geometry(geometry)
            vis.add_geometry(geometry_joints)
            vis.add_geometry(lines)
            vis.add_geometry(geometry_mesh)
            # vis.add_geometry(axis)
        else:
            # vis.update_geometry(geometry)
            vis.update_geometry(geometry_joints)
            vis.update_geometry(lines)
            vis.update_geometry(geometry_mesh)
            # vis.update_geometry(axis)
        
        vis.poll_events()
        vis.update_renderer()

        print(f"Update {i}: {time.time()}")
        time.sleep(.05)


def get_smpl_faces():
    dir = os.path.dirname(os.path.abspath(__file__))
    smpl_faces = np.load(os.path.join(dir, 'data/smpl_faces.npy'))

    return smpl_faces


if __name__ == "__main__":
    faces = get_smpl_faces()
    print("SMPL faces:", faces.shape)

    path = "/home/hamid/Documents/phd/footefield/MotionBERT/output/2_4_woman_1000/vertices.npy"
    poses = np.load(path)
    print("MoCap vertices:", poses.shape)

    path = "/home/hamid/Documents/phd/footefield/MotionBERT/output/2_4_woman_1000/joints.npy"
    joints = np.load(path)
    print("MoCap joints:", joints.shape)

    visualize_poses(poses, joints, faces)
