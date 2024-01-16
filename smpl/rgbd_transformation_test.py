# Import Open3D and numpy
import cv2
import open3d as o3d
import numpy as np
import diskcache

cam_name_0 = '2_4'
cam_name_1 = '1_5'

cam0 = f'azure_kinect{cam_name_0}_calib_snap'
cam1 = f'azure_kinect{cam_name_1}_calib_snap'
img_0_depth = f'/home/hamid/Documents/footefield/data/AzureKinectRecord_0729/a1/azure_kinect{cam_name_0}/depth/depth00000.png'
img_0_color = f'/home/hamid/Documents/footefield/data/AzureKinectRecord_0729/a1/azure_kinect{cam_name_0}/color/color00000.jpg'
img_1_depth = f'/home/hamid/Documents/footefield/data/AzureKinectRecord_0729/a1/azure_kinect{cam_name_1}/depth/depth00000.png'
img_1_color = f'/home/hamid/Documents/footefield/data/AzureKinectRecord_0729/a1/azure_kinect{cam_name_1}/color/color00000.jpg'

color1 = cv2.imread(img_0_color)
color1 = cv2.resize(color1, (640, 576))

color2 = cv2.imread(img_1_color)
color2 = cv2.resize(color2, (640, 576))

# Read the RGBD images
color1 = o3d.geometry.Image((color1).astype(np.uint8))
depth1 = o3d.io.read_image(img_0_depth)
color2 = o3d.geometry.Image((color2).astype(np.uint8))
depth2 = o3d.io.read_image(img_1_depth)

cache = diskcache.Cache('../calibration/cache')

mtx0 = cache['extrinsics'][cam0]['mtx_l']
dist0 = cache['extrinsics'][cam0]['dist_l']
mtx1 = cache['extrinsics'][cam0]['mtx_r']
dist1 = cache['extrinsics'][cam0]['dist_r']
R = cache['extrinsics'][cam0]['rotation']
T = cache['extrinsics'][cam0]['transition']

# Create RGBD images from color and depth images
rgbd1 = o3d.geometry.RGBDImage.create_from_color_and_depth(color1, depth1)
rgbd2 = o3d.geometry.RGBDImage.create_from_color_and_depth(color2, depth2)

# Define the intrinsic parameters of the cameras
# You need to adjust these values according to your cameras
fx = 525.0 # Focal length x
fy = 525.0 # Focal length y
cx = 319.5 # Optical center x
cy = 239.5 # Optical center y
intrinsic1 = o3d.camera.PinholeCameraIntrinsic(640, 576, mtx0[0, 0], mtx0[1, 1], mtx0[0, 2], mtx0[1, 2])
intrinsic2 = o3d.camera.PinholeCameraIntrinsic(640, 576, mtx1[0, 0], mtx1[1, 1], mtx1[0, 2], mtx1[1, 2])

# # Define the extrinsic parameters of the second camera
# # You can use the rotation and translation matrix that you have
r = np.array([[1, 0, 0], # Rotation matrix
              [0, 1, 0],
              [0, 0, 1]])
t = np.array([0, 0, 0]) # Translation vector
extrinsic1 = np.identity(4) # Extrinsic matrix
extrinsic1[:3, :3] = r
extrinsic1[:3, 3] = t.reshape(3)

extrinsic2 = np.identity(4) # Extrinsic matrix
extrinsic2[:3, :3] = R
extrinsic2[:3, 3] = np.array([-1.61, -.2, 1.1])

print(extrinsic1)
print(extrinsic2)

# # Transform the second RGBD image using the extrinsic matrix
# rgbd2.transform(extrinsic)

# Create point clouds from RGBD images
pcd1 = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd1, intrinsic1, extrinsic1)
pcd1.paint_uniform_color([1, 0, 0])
pcd2 = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd2, intrinsic2, extrinsic2)
pcd2.paint_uniform_color([0, 1, 0])

pcd1_np = np.asarray(pcd1.points)
print(np.min(pcd1_np[:, 0]), np.max(pcd1_np[:, 0]))

pcd2_np = np.asarray(pcd2.points)
print(np.min(pcd2_np[:, 0]), np.max(pcd2_np[:, 0]))

# Combine the point clouds
# pcd = pcd2
pcd = pcd1 + pcd2

# Visualize the point cloud
vis = o3d.visualization.Visualizer()
vis.create_window(visible=True)
# Call only after creating visualizer window.
vis.get_render_option().background_color = [.9, .9, .9]
vis.add_geometry(pcd)
vis.run()