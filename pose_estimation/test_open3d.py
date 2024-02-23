import sys
sys.path.append('../')

import cv2
import diskcache
import numpy as np
import open3d as o3d

from utils import data_loader
from calibration import rgb_depth_map


cache = diskcache.Cache('../calibration/cache')

camera = 'azure_kinect1_4_calib_snap'

img_depth_path = '/home/hamid/Documents/phd/footefield/data/AzureKinectRecord_0729/a1/azure_kinect1_4/depth/depth00000.png'
img_color_path = '/home/hamid/Documents/phd/footefield/data/AzureKinectRecord_0729/a1/azure_kinect1_4/color/color00000.jpg'

img_depth = cv2.imread(img_depth_path, -1)
img_color = cv2.imread(img_color_path)
img_color = data_loader.downsample_keep_aspect_ratio(
    img_color,
    (
        data_loader.IMAGE_INFRARED_WIDTH,
        data_loader.IMAGE_INFRARED_HEIGHT
    )
)

img_color = rgb_depth_map.align_image_rgb(img_color, camera, cache)
img_depth = rgb_depth_map.align_image_depth(img_depth, camera, cache)

color0 = o3d.geometry.Image((img_color).astype(np.uint8))
depth0 = o3d.geometry.Image(img_depth)

mtx0 = cache['extrinsics'][camera + 'infrared']['mtx_l']

rgbd0 = o3d.geometry.RGBDImage.create_from_color_and_depth(color0, depth0)

intrinsic0 = o3d.camera.PinholeCameraIntrinsic(640, 576, mtx0[0, 0], mtx0[1, 1], mtx0[0, 2], mtx0[1, 2])

extrinsic0 = np.identity(4) # Extrinsic matrix
r = np.array([[1, 0, 0], # Rotation matrix
            [0, 1, 0],
            [0, 0, 1]])
t = np.array([0, 0, 0]) # Translation vector
extrinsic0[:3, :3] = r
extrinsic0[:3, 3] = t.reshape(3)

pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd0, intrinsic0, extrinsic0)

vis = o3d.visualization.Visualizer()
vis.create_window()

opt = vis.get_render_option()
opt.show_coordinate_frame = True
opt.background_color = np.asarray([0.5, 0.5, 0.5])

vis.add_geometry(pcd)
vis.update_renderer()
vis.run()
vis.destroy_window()
