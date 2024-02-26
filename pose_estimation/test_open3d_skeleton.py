import sys
sys.path.append('../')

import os
import cv2
import math
import glob
import time
import diskcache
import numpy as np
import open3d as o3d

from utils import data_loader
from calibration import rgb_depth_map


HALPE_LINES = np.array(
    [(0, 1), (0, 2), (1, 3), (2, 4), (5, 18), (6, 18), (5, 7),
     (7, 9), (6, 8), (8, 10), (17, 18), (18, 19), (19, 11),
     (19, 12), (11, 13), (12, 14), (13, 15), (14, 16), (20, 24),
     (21, 25), (23, 25), (22, 24), (15, 24), (16, 25)])

def visualize_poses(poses):
    cache = diskcache.Cache('../calibration/cache')

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window()

    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=300)
    
    geometry = o3d.geometry.PointCloud()
    lines = o3d.geometry.LineSet()

    geometry_image = o3d.geometry.PointCloud()

    for idx in range(len(poses)):
        camera = 'azure_kinect3_5_calib_snap'

        img_depth_path = '/home/hamid/Documents/phd/footefield/data/AzureKinectRecord_0729/a1/azure_kinect3_5/depth/depth{:05d}.png'.format(idx)
        img_color_path = '/home/hamid/Documents/phd/footefield/data/AzureKinectRecord_0729/a1/azure_kinect3_5/color/color{:05d}.jpg'.format(idx)

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
        color0 = o3d.geometry.Image((img_color).astype(np.uint8))
        depth0 = o3d.geometry.Image(img_depth)

        mtx0 = cache['extrinsics']['azure_kinect3_4_calib_snap' + 'infrared']['mtx_r']

        rgbd0 = o3d.geometry.RGBDImage.create_from_color_and_depth(color0, depth0, convert_rgb_to_intensity=False)

        intrinsic0 = o3d.camera.PinholeCameraIntrinsic(640, 576, mtx0[0, 0], mtx0[1, 1], mtx0[0, 2], mtx0[1, 2])

        extrinsic0 = np.identity(4) # Extrinsic matrix
        r = np.array([[1, 0, 0], # Rotation matrix
                    [0, 1, 0],
                    [0, 0, 1]])
        t = np.array([0, 0, 0]) # Translation vector
        extrinsic0[:3, :3] = r
        extrinsic0[:3, 3] = t.reshape(3)

        pcd_image = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd0, intrinsic0, extrinsic0)
        pcd_image.paint_uniform_color([0, 1, 0]) # red points

        poses[idx][:,:,2] /= 1000.
        poses[idx][:,:,0] = (poses[idx][:,:,0] - mtx0[0, 2]) * poses[idx][:,:,2] / mtx0[0, 0]
        poses[idx][:,:,1] = (poses[idx][:,:,1] - mtx0[1, 2]) * poses[idx][:,:,2] / mtx0[1, 1]
        keypoints = poses[idx].reshape(-1, 3)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(keypoints)
        pcd.paint_uniform_color([1, 0, 0]) # red points

        connections = np.concatenate((HALPE_LINES, HALPE_LINES + 26))
        
        lines.points = o3d.utility.Vector3dVector(keypoints)
        lines.lines = o3d.utility.Vector2iVector(connections)

        geometry.points = pcd.points
        geometry.colors = pcd.colors
        geometry_image.points = pcd_image.points
        geometry_image.colors = pcd_image.colors
        if idx == 0:
            vis.add_geometry(geometry)
            vis.add_geometry(lines)
            # vis.add_geometry(axis)
            vis.add_geometry(geometry_image)
        else:
            vis.update_geometry(geometry)
            vis.update_geometry(lines)
            # vis.update_geometry(axis)
            vis.update_geometry(geometry_image)
            
        for delay in range(20):
            vis.poll_events()
            vis.update_renderer()
            time.sleep(.01)

        print(f"Update {idx}: {time.time()}")
            

if __name__ == "__main__":
    cache = diskcache.Cache('../calibration/cache')

    cache_process = cache.get('process', {})

    for file in os.listdir('keypoints_3d'):
        file_path = os.path.join('keypoints_3d', file)
        print(f"Visualizing {file_path}")

        import pickle
        with open(file_path, 'rb') as handle:
            poses = np.array(pickle.load(handle))

        visualize_poses(poses)
