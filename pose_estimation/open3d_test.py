import sys
sys.path.append('../')

import cv2
import diskcache
import numpy as np
import open3d as o3d

from utils import data_loader
from calibration import rgb_depth_map


cache = diskcache.Cache('../calibration/cache')

cam = '2_4'
camera = f'azure_kinect{cam}_calib_snap'
img_depth_path = f'/home/hamid/Documents/footefield/data/AzureKinectRecord_0729/a1/azure_kinect{cam}/depth/depth00000.png'
img_color_path = f'/home/hamid/Documents/footefield/data/AzureKinectRecord_0729/a1/azure_kinect{cam}/color/color00000.jpg'

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

point_cloud = []
colors = []
for i in range(img_depth.shape[1]):
    for j in range(img_depth.shape[0]):
        if img_depth[j, i] != 0:
            point_cloud.append((i / 640., 1 - (j / 576.), 1 - (img_depth[j, i] / 5000.)))
            colors.append((img_color[j, i, 0] / 255.,
                            img_color[j, i, 1] / 255.,
                            img_color[j, i, 2] / 255.))

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(point_cloud)
pcd.colors = o3d.utility.Vector3dVector(colors)
o3d.visualization.draw_geometries([pcd])
