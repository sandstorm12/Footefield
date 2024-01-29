# Import Open3D and numpy
import cv2
import copy
import math
import open3d as o3d
import numpy as np
import diskcache

from probreg import cpd

INDEX = '00750'

# # Good pairs
# 2_4 & 1_5
# 3_4 & 3_5
# 1_4 & 3_4

cam_name_0 = '2_4'
cam_name_1 = '1_5'
cam_name_2 = '1_4'
cam_name_3 = '3_4'
cam_name_4 = '3_5'

cam0 = f'azure_kinect{cam_name_0}_calib_snap'
cam1 = f'azure_kinect{cam_name_1}_calib_snap'
cam2 = f'azure_kinect{cam_name_2}_calib_snap'
cam3 = f'azure_kinect{cam_name_3}_calib_snap'
cam4 = f'azure_kinect{cam_name_4}_calib_snap'
img_0_depth = f'/home/hamid/Documents/footefield/data/AzureKinectRecord_0729/a1/azure_kinect{cam_name_0}/depth/depth{INDEX}.png'
img_0_color = f'/home/hamid/Documents/footefield/data/AzureKinectRecord_0729/a1/azure_kinect{cam_name_0}/color/color{INDEX}.jpg'
img_1_depth = f'/home/hamid/Documents/footefield/data/AzureKinectRecord_0729/a1/azure_kinect{cam_name_1}/depth/depth{INDEX}.png'
img_1_color = f'/home/hamid/Documents/footefield/data/AzureKinectRecord_0729/a1/azure_kinect{cam_name_1}/color/color{INDEX}.jpg'
img_2_depth = f'/home/hamid/Documents/footefield/data/AzureKinectRecord_0729/a1/azure_kinect{cam_name_2}/depth/depth{INDEX}.png'
img_2_color = f'/home/hamid/Documents/footefield/data/AzureKinectRecord_0729/a1/azure_kinect{cam_name_2}/color/color{INDEX}.jpg'
img_3_depth = f'/home/hamid/Documents/footefield/data/AzureKinectRecord_0729/a1/azure_kinect{cam_name_3}/depth/depth{INDEX}.png'
img_3_color = f'/home/hamid/Documents/footefield/data/AzureKinectRecord_0729/a1/azure_kinect{cam_name_3}/color/color{INDEX}.jpg'
img_4_depth = f'/home/hamid/Documents/footefield/data/AzureKinectRecord_0729/a1/azure_kinect{cam_name_4}/depth/depth{INDEX}.png'
img_4_color = f'/home/hamid/Documents/footefield/data/AzureKinectRecord_0729/a1/azure_kinect{cam_name_4}/color/color{INDEX}.jpg'

color1 = cv2.imread(img_0_color)
color1 = cv2.resize(color1, (640, 576))
color2 = cv2.imread(img_1_color)
color2 = cv2.resize(color2, (640, 576))
color3 = cv2.imread(img_2_color)
color3 = cv2.resize(color3, (640, 576))
color4 = cv2.imread(img_3_color)
color4 = cv2.resize(color4, (640, 576))
color5 = cv2.imread(img_4_color)
color5 = cv2.resize(color5, (640, 576))

# Read the RGBD images
color1 = o3d.geometry.Image((color1).astype(np.uint8))
depth1 = o3d.io.read_image(img_0_depth)
color2 = o3d.geometry.Image((color2).astype(np.uint8))
depth2 = o3d.io.read_image(img_1_depth)
color3 = o3d.geometry.Image((color3).astype(np.uint8))
depth3 = o3d.io.read_image(img_2_depth)
color4 = o3d.geometry.Image((color4).astype(np.uint8))
depth4 = o3d.io.read_image(img_3_depth)
color5 = o3d.geometry.Image((color5).astype(np.uint8))
depth5 = o3d.io.read_image(img_4_depth)

cache = diskcache.Cache('../calibration/cache')

mtx0 = cache['extrinsics'][cam0]['mtx_l']
dist0 = cache['extrinsics'][cam0]['dist_l']
mtx1 = cache['extrinsics'][cam0]['mtx_r']
dist1 = cache['extrinsics'][cam0]['dist_r']
R = cache['extrinsics'][cam0]['rotation']
T = cache['extrinsics'][cam0]['transition']

mtx2 = cache['extrinsics'][cam1]['mtx_l']
dist2 = cache['extrinsics'][cam1]['dist_l']
mtx3 = cache['extrinsics'][cam1]['mtx_r']
dist3 = cache['extrinsics'][cam1]['dist_r']
R2 = cache['extrinsics'][cam1]['rotation']
T2 = cache['extrinsics'][cam1]['transition']

mtx4 = cache['extrinsics'][cam2]['mtx_l']
dist4 = cache['extrinsics'][cam2]['dist_l']
mtx5 = cache['extrinsics'][cam2]['mtx_r']
dist5 = cache['extrinsics'][cam2]['dist_r']
R3 = cache['extrinsics'][cam2]['rotation']
T3 = cache['extrinsics'][cam2]['transition']

mtx6 = cache['extrinsics'][cam3]['mtx_l']
dist6 = cache['extrinsics'][cam3]['dist_l']
mtx7 = cache['extrinsics'][cam3]['mtx_r']
dist7 = cache['extrinsics'][cam3]['dist_r']
R4 = cache['extrinsics'][cam3]['rotation']
T4 = cache['extrinsics'][cam3]['transition']

# Create RGBD images from color and depth images
rgbd1 = o3d.geometry.RGBDImage.create_from_color_and_depth(color1, depth1)
rgbd2 = o3d.geometry.RGBDImage.create_from_color_and_depth(color2, depth2)
rgbd3 = o3d.geometry.RGBDImage.create_from_color_and_depth(color3, depth3)
rgbd4 = o3d.geometry.RGBDImage.create_from_color_and_depth(color4, depth4)
rgbd5 = o3d.geometry.RGBDImage.create_from_color_and_depth(color5, depth5)


# Define the intrinsic parameters of the cameras
# You need to adjust these values according to your cameras
intrinsic1 = o3d.camera.PinholeCameraIntrinsic(640, 576, mtx0[0, 0], mtx0[1, 1], mtx0[0, 2], mtx0[1, 2])
intrinsic2 = o3d.camera.PinholeCameraIntrinsic(640, 576, mtx1[0, 0], mtx1[1, 1], mtx1[0, 2], mtx1[1, 2])
intrinsic3 = o3d.camera.PinholeCameraIntrinsic(640, 576, mtx3[0, 0], mtx3[1, 1], mtx3[0, 2], mtx3[1, 2])
intrinsic4 = o3d.camera.PinholeCameraIntrinsic(640, 576, mtx5[0, 0], mtx5[1, 1], mtx5[0, 2], mtx5[1, 2])
intrinsic5 = o3d.camera.PinholeCameraIntrinsic(640, 576, mtx7[0, 0], mtx7[1, 1], mtx7[0, 2], mtx7[1, 2])

# # Define the extrinsic parameters of the second camera
# # You can use the rotation and translation matrix that you have
extrinsic1 = np.identity(4) # Extrinsic matrix
r = np.array([[1, 0, 0], # Rotation matrix
              [0, 1, 0],
              [0, 0, 1]])
t = np.array([0, 0, 0]) # Translation vector
extrinsic1[:3, :3] = r
extrinsic1[:3, 3] = t.reshape(3)

extrinsic2 = np.identity(4) # Extrinsic matrix
extrinsic2[:3, :3] = R
extrinsic2[:3, 3] = T.reshape(3,) / 1000

extrinsic3 = np.identity(4) # Extrinsic matrix
R2_com = np.dot(R2, R)
T2_com = (np.dot(R2, T).reshape(3, 1) + T2).reshape(3,)
extrinsic3[:3, :3] = R2_com
extrinsic3[:3, 3] = (T2_com / 1000) + np.array([0, 0, 0])

extrinsic4 = np.identity(4) # Extrinsic matrix
R3_com = np.dot(R3, R2_com)
T3_com = (np.dot(R3, T2_com).reshape(3, 1) + T3).reshape(3,)
extrinsic4[:3, :3] = R3_com
extrinsic4[:3, 3] = T3_com / 1000

extrinsic5 = np.identity(4) # Extrinsic matrix
R4_com = np.dot(R4, R3_com)
T4_com = (np.dot(R4, T3_com).reshape(3, 1) + T4).reshape(3,)
extrinsic5[:3, :3] = R4_com
extrinsic5[:3, 3] = T4_com / 1000

# print(extrinsic1)
# print(extrinsic2)
# print(extrinsic3)
# print(extrinsic4)
# print(extrinsic5)

# # Transform the second RGBD image using the extrinsic matrix
# rgbd2.transform(extrinsic)

# Create point clouds from RGBD images
pcd1 = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd1, intrinsic1, extrinsic1)
pcd1.paint_uniform_color([1, 0, 0])
pcd2 = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd2, intrinsic2, extrinsic2)
pcd2.paint_uniform_color([0, 1, 0])
pcd3 = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd3, intrinsic3, extrinsic3)
pcd3.paint_uniform_color([0, 0, 1])
pcd4 = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd4, intrinsic4, extrinsic4)
pcd4.paint_uniform_color([0, 1, 0])
pcd5 = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd5, intrinsic5, extrinsic5)
pcd5.paint_uniform_color([0, 1, 0])

# Combine the point clouds
# pcd = pcd2
# pcd = pcd1 + pcd2 + pcd3 + pcd4 + pcd5
pcd = pcd1 + pcd2 + pcd5

# Visualize the point cloud
vis = o3d.visualization.Visualizer()
vis.create_window(visible=True)
# Call only after creating visualizer window.
vis.get_render_option().background_color = [.9, .9, .9]
vis.add_geometry(pcd)
vis.run()


##################################################################################

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

# transition = [0.07, 0.019999999999999997, 0.060000000000000005]
# angle = [9.099999999999984, 3.300000000000001, 3.3000000000000016]
# rm = rotation_matrix_from_euler_angles(np.deg2rad(angle[0]),
#                                        np.deg2rad(angle[1]),
#                                        np.deg2rad(angle[2]))
# pcd3.rotate(rm)
# pcd3.translate(transition)

pcd1 = pcd1
pcd2 = pcd2

transition = [-0.03500000000000001, -0.005, 0.006]
angle = [0.09999999999999981, -0.1, 0.10000000000000003]
rm = rotation_matrix_from_euler_angles(np.deg2rad(angle[0]),
                                       np.deg2rad(angle[1]),
                                       np.deg2rad(angle[2]))
pcd2.rotate(rm)
pcd2.translate(transition)

vis = o3d.visualization.VisualizerWithKeyCallback()
vis.create_window()

vis.add_geometry(pcd1)
vis.add_geometry(pcd2)

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

# pcd2 to pcd1 [-0.070000000000000005, -0.03, 0] [1.8, 0, 0]
# [-0.03500000000000001, -0.005, 0.006] [0.09999999999999981, -0.1, 0.10000000000000003]
# [-0.06500000000000002, -0.045000000000000005, -0.034] [0.3999999999999998, -0.1, 0.10000000000000003]
# pcd3 to pcd1 [0.07, 0.019999999999999997, 0.060000000000000005] [9.099999999999984, 3.300000000000001, 3.3000000000000016]
# pcd4 to pcd1 [0.07, -0.09999999999999999, 0.10999999999999996] [11.399999999999975, 6.699999999999992, -3.100000000000002]
# [0.08, -0.08, 0.10999999999999996] [6.999999999999991, 6.299999999999994, -3.100000000000002]
# pcd5 to pcd1 [0.10999999999999999, -0.09, 0.21] [12.499999999999972, 4.000000000000002, -6.499999999999994][12.499999999999972, 4.000000000000002, -6.499999999999994]
# [0.08, -0.08999999999999997, 0.24000000000000005] [12.499999999999972, 4.000000000000002, -6.499999999999994]
# [0.07999999999999997, -0.06999999999999998, 0.21000000000000002] [11.999999999999973, 3.100000000000001, -6.099999999999995]