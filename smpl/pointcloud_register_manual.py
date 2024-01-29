# Import Open3D and numpy
import cv2
import copy
import math
import open3d as o3d
import numpy as np
import diskcache

from probreg import cpd

INDEX = '00420'

# # Good pairs
# 2_4 & 1_5
# 3_4 & 3_5
# 1_4 & 3_4

cam_name_0 = '1_5'
cam_name_1 = '1_4'
cam_name_2 = '3_4'

cam0 = f'azure_kinect{cam_name_0}_calib_snap'
cam1 = f'azure_kinect{cam_name_1}_calib_snap'
cam2 = f'azure_kinect{cam_name_2}_calib_snap'
img_0_depth = f'/home/hamid/Documents/footefield/data/AzureKinectRecord_0729/a1/azure_kinect{cam_name_0}/depth/depth{INDEX}.png'
img_0_color = f'/home/hamid/Documents/footefield/data/AzureKinectRecord_0729/a1/azure_kinect{cam_name_0}/color/color{INDEX}.jpg'
img_1_depth = f'/home/hamid/Documents/footefield/data/AzureKinectRecord_0729/a1/azure_kinect{cam_name_1}/depth/depth{INDEX}.png'
img_1_color = f'/home/hamid/Documents/footefield/data/AzureKinectRecord_0729/a1/azure_kinect{cam_name_1}/color/color{INDEX}.jpg'
img_2_depth = f'/home/hamid/Documents/footefield/data/AzureKinectRecord_0729/a1/azure_kinect{cam_name_2}/depth/depth{INDEX}.png'
img_2_color = f'/home/hamid/Documents/footefield/data/AzureKinectRecord_0729/a1/azure_kinect{cam_name_2}/color/color{INDEX}.jpg'

color0 = cv2.imread(img_0_color)
color0 = cv2.resize(color0, (640, 576))
color1 = cv2.imread(img_1_color)
color1 = cv2.resize(color1, (640, 576))
color2 = cv2.imread(img_2_color)
color2 = cv2.resize(color2, (640, 576))

# Read the RGBD images
color0 = o3d.geometry.Image((color0).astype(np.uint8))
depth0 = o3d.io.read_image(img_0_depth)
color1 = o3d.geometry.Image((color1).astype(np.uint8))
depth1 = o3d.io.read_image(img_1_depth)
color2 = o3d.geometry.Image((color2).astype(np.uint8))
depth2 = o3d.io.read_image(img_2_depth)

cache = diskcache.Cache('../calibration/cache')

mtx0 = cache['extrinsics'][cam0 + 'infrared']['mtx_l']
dist0 = cache['extrinsics'][cam0 + 'infrared']['dist_l']
mtx1 = cache['extrinsics'][cam0 + 'infrared']['mtx_r']
dist1 = cache['extrinsics'][cam0 + 'infrared']['dist_r']
R = cache['extrinsics'][cam0 + 'infrared']['rotation']
T = cache['extrinsics'][cam0 + 'infrared']['transition']

mtx2 = cache['extrinsics'][cam1 + 'infrared']['mtx_l']
dist2 = cache['extrinsics'][cam1 + 'infrared']['dist_l']
mtx3 = cache['extrinsics'][cam1 + 'infrared']['mtx_r']
dist3 = cache['extrinsics'][cam1 + 'infrared']['dist_r']
R2 = cache['extrinsics'][cam1 + 'infrared']['rotation']
T2 = cache['extrinsics'][cam1 + 'infrared']['transition']

# Create RGBD images from color and depth images
rgbd0 = o3d.geometry.RGBDImage.create_from_color_and_depth(color0, depth0)
rgbd1 = o3d.geometry.RGBDImage.create_from_color_and_depth(color1, depth1)
rgbd2 = o3d.geometry.RGBDImage.create_from_color_and_depth(color2, depth2)


# Define the intrinsic parameters of the cameras
# You need to adjust these values according to your cameras
intrinsic0 = o3d.camera.PinholeCameraIntrinsic(640, 576, mtx0[0, 0], mtx0[1, 1], mtx0[0, 2], mtx0[1, 2])
intrinsic1 = o3d.camera.PinholeCameraIntrinsic(640, 576, mtx1[0, 0], mtx1[1, 1], mtx1[0, 2], mtx1[1, 2])
intrinsic2 = o3d.camera.PinholeCameraIntrinsic(640, 576, mtx3[0, 0], mtx3[1, 1], mtx3[0, 2], mtx3[1, 2])

# # Define the extrinsic parameters of the second camera
# # You can use the rotation and translation matrix that you have
extrinsic0 = np.identity(4) # Extrinsic matrix
r = np.array([[1, 0, 0], # Rotation matrix
              [0, 1, 0],
              [0, 0, 1]])
t = np.array([0, 0, 0]) # Translation vector
extrinsic0[:3, :3] = r
extrinsic0[:3, 3] = t.reshape(3)

extrinsic1 = np.identity(4) # Extrinsic matrix
extrinsic1[:3, :3] = R
extrinsic1[:3, 3] = T.reshape(3,) / 1000

extrinsic2 = np.identity(4) # Extrinsic matrix
R2_com = np.dot(R2, R)
T2_com = (np.dot(R2, T).reshape(3, 1) + T2).reshape(3,)
extrinsic2[:3, :3] = R2_com
extrinsic2[:3, 3] = (T2_com / 1000) + np.array([0, 0, 0])


# Create point clouds from RGBD images
pcd0 = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd0, intrinsic0, extrinsic0)
pcd0.paint_uniform_color([0, 0, 1])
pcd1 = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd1, intrinsic1, extrinsic1)
pcd1.paint_uniform_color([1, 0, 0])
pcd2 = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd2, intrinsic2, extrinsic2)
pcd2.paint_uniform_color([0, 1, 0])

# Combine the point clouds
pcd = pcd0 + pcd1 + pcd2

# Visualize the point cloud
vis = o3d.visualization.Visualizer()
vis.create_window(visible=True)
# Call only after creating visualizer window.
vis.get_render_option().background_color = [.9, .9, .9]
vis.add_geometry(pcd)
vis.run()


# ##################################################################################

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
    transition[1] += .01
    pcd2.translate(np.array([0, .01, 0]))
    vis.update_geometry(pcd2)

    print(transition, angle)

def key_callback_s(vis):
    transition[1] += -.01
    pcd2.translate(np.array([0, -.01, 0]))
    vis.update_geometry(pcd2)

    print(transition, angle)

def key_callback_a(vis):
    transition[0] += .01
    pcd2.translate(np.array([.01, 0, 0]))
    vis.update_geometry(pcd2)

    print(transition, angle)

def key_callback_d(vis):
    transition[0] += -.01
    pcd2.translate(np.array([-.01, 0, 0]))
    vis.update_geometry(pcd2)

    print(transition, angle)

def key_callback_e(vis):
    transition[2] += .01
    pcd2.translate(np.array([0, 0, 0.01]))
    vis.update_geometry(pcd2)

    print(transition, angle)

def key_callback_r(vis):
    transition[2] += -.01
    pcd2.translate(np.array([0, 0, -0.01]))
    vis.update_geometry(pcd2)

    print(transition, angle)

def key_callback_t(vis):
    angle[1] += .1
    rm = rotation_matrix_from_euler_angles(0, np.deg2rad(1), 0)
    pcd2.rotate(rm)
    vis.update_geometry(pcd2)

    print(transition, angle)

def key_callback_g(vis):
    angle[1] += -.1
    rm = rotation_matrix_from_euler_angles(0, np.deg2rad(-1), 0)
    pcd2.rotate(rm)
    vis.update_geometry(pcd2)

    print(transition, angle)

def key_callback_f(vis):
    angle[0] += .1
    rm = rotation_matrix_from_euler_angles(np.deg2rad(1), 0, 0)
    pcd2.rotate(rm)
    vis.update_geometry(pcd2)

    print(transition, angle)

def key_callback_h(vis):
    angle[0] += -.1
    rm = rotation_matrix_from_euler_angles(np.deg2rad(-1), 0, 0)
    pcd2.rotate(rm)
    vis.update_geometry(pcd2)

    print(transition, angle)

def key_callback_y(vis):
    angle[2] += .1
    rm = rotation_matrix_from_euler_angles(0, 0, np.deg2rad(1))
    pcd2.rotate(rm)
    vis.update_geometry(pcd2)

    print(transition, angle)

def key_callback_u(vis):
    angle[2] += -.1
    rm = rotation_matrix_from_euler_angles(0, 0, np.deg2rad(-1))
    pcd2.rotate(rm)
    vis.update_geometry(pcd2)

    print(transition, angle)


pcd0 = pcd0 + pcd1
pcd1 = pcd2

transition = [0.009999999999999997, 0.03, 0.03]
angle = [0.0, 2.1000000000000005, 0.0]
rm = rotation_matrix_from_euler_angles(np.deg2rad(angle[0]),
                                       np.deg2rad(angle[1]),
                                       np.deg2rad(angle[2]))
pcd1 = pcd1.rotate(rm)
pcd1 = pcd1.translate(transition)

vis = o3d.visualization.VisualizerWithKeyCallback()
vis.create_window()

vis.add_geometry(pcd0)
vis.add_geometry(pcd1)

opt = vis.get_render_option()
opt.show_coordinate_frame = True

vis.register_key_callback(87, key_callback_w)
vis.register_key_callback(83, key_callback_s)
vis.register_key_callback(65, key_callback_a)
vis.register_key_callback(68, key_callback_d)
vis.register_key_callback(69, key_callback_e)
vis.register_key_callback(82, key_callback_r)

vis.register_key_callback(84, key_callback_t)
vis.register_key_callback(71, key_callback_g)
vis.register_key_callback(70, key_callback_f)
vis.register_key_callback(72, key_callback_h)
vis.register_key_callback(89, key_callback_y)
vis.register_key_callback(85, key_callback_u)

vis.run()
vis.destroy_window()

# pcd1 to pcd0
# pcd2 to pcd0
# [0, 0, 0] [0.0, 0.6, 0.0]
# [-3.469446951953614e-18, 0.0, 0.019999999999999997] [0.0, 1.2, 0.0]
# [0.060000000000000005, 0.019999999999999997, 0.019999999999999997] [0.0, 1.8000000000000005, 0.0]
# [0.019999999999999997, 0.03, 0.09999999999999999] [0.0, 1.8000000000000005, 0.0]
# [0.009999999999999997, 0.03, 0.03] [0.0, 2.1000000000000005, 0.0]
# [0.009999999999999997, 0.03, 0.060000000000000005] [0.0, 2.1000000000000005, 0.0]