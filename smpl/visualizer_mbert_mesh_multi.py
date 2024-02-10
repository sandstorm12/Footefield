import sys
sys.path.append('../')

import os
import time
import numpy as np
import open3d as o3d

from smplx import SMPL

from utils import data_loader


def rotation_matrix_from_euler_angles(roll, pitch, yaw):
    """
        Creates a 3x3 rotation matrix from euler angles (roll, pitch, yaw).

        Args:
        roll (float): Roll angle in radians.
        pitch (float): Pitch angle in radians.
        yaw (float): Yaw angle in radians.

        Returns:
        rotation_matrix (np.ndarray): 3x3 rotation matrix.
    """

    rotation_matrix = np.identity(3)

    # Roll
    c_roll = np.cos(roll)
    s_roll = np.sin(roll)

    rotation_matrix = np.dot(rotation_matrix, np.array(
        [[1, 0, 0],
        [0, c_roll, -s_roll],
        [0, s_roll, c_roll]]))

    # Pitch
    c_pitch = np.cos(pitch)
    s_pitch = np.sin(pitch)

    rotation_matrix = np.dot(rotation_matrix, np.array(
        [[c_pitch, 0, s_pitch],
        [0, 1, 0],
        [-s_pitch, 0, c_pitch]]))

    # Yaw
    c_yaw = np.cos(yaw)
    s_yaw = np.sin(yaw)

    rotation_matrix = np.dot(rotation_matrix, np.array(
        [[c_yaw, s_yaw, 0],
        [-s_yaw, c_yaw, 0],
        [0, 0, 1]]))

    return rotation_matrix

def key_callback_w(vis):
    transition[1] += 100

    print(transition, angle)

def key_callback_s(vis):
    transition[1] += -100

    print(transition, angle)

def key_callback_a(vis):
    transition[0] += 100

    print(transition, angle)

def key_callback_d(vis):
    transition[0] += -100

    print(transition, angle)

def key_callback_e(vis):
    transition[2] += 100

    print(transition, angle)

def key_callback_r(vis):
    transition[2] += -100

    print(transition, angle)

def key_callback_t(vis):
    angle[1] += 1

    print(transition, angle)

def key_callback_g(vis):
    angle[1] += -1

    print(transition, angle)

def key_callback_f(vis):
    angle[0] += 1
    
    print(transition, angle)

def key_callback_h(vis):
    angle[0] += -1

    print(transition, angle)

def key_callback_y(vis):
    angle[2] += 1
    
    print(transition, angle)

def key_callback_u(vis):
    angle[2] += -1

    print(transition, angle)


def visualize_poses(poses, joints, triangles, control_idx):
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window()
    # vis.register_key_callback(87, key_callback_w)
    # vis.register_key_callback(83, key_callback_s)
    # vis.register_key_callback(65, key_callback_a)
    # vis.register_key_callback(68, key_callback_d)
    # vis.register_key_callback(69, key_callback_e)
    # vis.register_key_callback(82, key_callback_r)

    # vis.register_key_callback(84, key_callback_t)
    # vis.register_key_callback(71, key_callback_g)
    # vis.register_key_callback(70, key_callback_f)
    # vis.register_key_callback(72, key_callback_h)
    # vis.register_key_callback(89, key_callback_y)
    # vis.register_key_callback(85, key_callback_u)
    
    geometry = [o3d.geometry.PointCloud() for _ in range(len(poses))]
    geometry_joints = [o3d.geometry.PointCloud() for _ in range(len(poses))]
    lines = [o3d.geometry.LineSet() for _ in range(len(poses))]
    geometry_mesh = [o3d.geometry.TriangleMesh() for _ in range(len(poses))]

    for i in range(len(poses[0])):
        for j in range(len(poses)):
            keypoints = poses[j][i].reshape(-1, 3)
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(keypoints)

            if j == control_idx:
                print("PCD control", j)
                pcd = pcd.translate(np.array(transition))
                rm = rotation_matrix_from_euler_angles(
                    np.deg2rad(angle[0]),
                    np.deg2rad(angle[1]),
                    np.deg2rad(angle[2]))
                pcd = pcd.rotate(rm)

            keypoints_joints = joints[j][i].reshape(-1, 3)
            keypoints_joints[:, 2] += 1000
            pcd_joints = o3d.geometry.PointCloud()
            pcd_joints.points = o3d.utility.Vector3dVector(keypoints_joints)

            if j == control_idx:
                pcd_joints = pcd_joints.translate(np.array(transition))
                rm = rotation_matrix_from_euler_angles(
                    np.deg2rad(angle[0]),
                    np.deg2rad(angle[1]),
                    np.deg2rad(angle[2]))
                pcd_joints = pcd_joints.rotate(rm)

            connections = np.array(data_loader.MBERT_EDGES)
            
            lines[j].points = pcd_joints.points
            lines[j].lines = o3d.utility.Vector2iVector(connections)    

            geometry[j].points = pcd.points
            geometry_joints[j].points = pcd_joints.points
            geometry_mesh[j].vertices = pcd.points
            geometry_mesh[j].triangles = o3d.utility.Vector3iVector(triangles)
            geometry_mesh[j].compute_vertex_normals()
            geometry_mesh[j].compute_triangle_normals()

            if i == 0:
                # vis.add_geometry(geometry[j])
                # vis.add_geometry(geometry_joints[j])
                # vis.add_geometry(lines[j])
                vis.add_geometry(geometry_mesh[j])
                # vis.add_geometry(axis)
            else:
                # vis.update_geometry(geometry[j])
                # vis.update_geometry(geometry_joints[j])
                # vis.update_geometry(lines[j])
                vis.update_geometry(geometry_mesh[j])
                # vis.update_geometry(axis)
        
        vis.poll_events()
        vis.update_renderer()

        print(f"Update {i}: {time.time()}")
        time.sleep(.05)


def get_smpl_faces():
    dir = os.path.dirname(os.path.abspath(__file__))
    smpl_faces = np.load(os.path.join(dir, 'data/smpl_faces.npy'))

    return smpl_faces


def load_model(path):
    vertices = os.path.join(path, "vertices.npy")
    poses = np.load(vertices)
    print("MoCap vertices:", poses.shape)

    joints = os.path.join(path, "joints.npy")
    joints = np.load(joints)
    print("MoCap joints:", joints.shape)

    return poses, joints


if __name__ == "__main__":
    faces = get_smpl_faces()
    print("SMPL faces:", faces.shape)

    paths = [
        "/home/hamid/Documents/phd/footefield/MotionBERT/output/2_4_woman_1000",
        "/home/hamid/Documents/phd/footefield/MotionBERT/output/2_4_man_1000"
    ]

    poses = []
    joints = []
    for path in paths:
        pose, joint = load_model(path)
        print(np.min(pose), np.mean(pose), np.max(pose))
        poses.append(pose)
        joints.append(joint)

    transition = [1500.0, -100.0, 300.0]
    angle = [0.0, -20.0, 0.0]

    visualize_poses(poses, joints, faces, control_idx=1)
