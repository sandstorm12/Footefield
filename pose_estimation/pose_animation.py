import sys
sys.path.append('../')

import os
import cv2
import diskcache
import numpy as np
import matplotlib.animation
import matplotlib.pyplot as plt

from utils import data_loader
from data_loader import *

from tqdm import tqdm
from calibration import rgb_depth_map
from mmpose.apis import MMPoseInferencer


def _get_skeleton(image, inferencer):
    result_generator = inferencer(image)
    
    detected_keypoints = []
    for result in result_generator:
        for predictions in result['predictions'][0]:
            keypoints = predictions['keypoints']
            detected_keypoints.append(keypoints)

    return np.array(detected_keypoints)

# Remove this and use the one in the rgb_depth_map
def points_to_depth(people_keypoints, image_depth):
    keypoints_3d = []
    for keypoints in people_keypoints:
        keypoints_3d.append([])
        for i in range(len(keypoints)):
            x, y = keypoints[i]
            x = int(x)
            y = int(y)
            roi = image_depth[
                y-rgb_depth_map.DEPTH_AREA:y+rgb_depth_map.DEPTH_AREA,
                x-rgb_depth_map.DEPTH_AREA:x+rgb_depth_map.DEPTH_AREA]
            roi = roi[np.logical_and(roi < 3000, roi > 100)]
            if len(roi) > 0:
                depth = np.max(roi)
                keypoints_3d[-1].append((x, y, depth))
            else:
                keypoints_3d[-1].append((x, y, 0))

    return keypoints_3d


def extract_poses(dir, camera):
    poses = []
    
    img_rgb_paths = data_loader.list_rgb_images(os.path.join(dir, "color"))
    img_dpt_paths = data_loader.list_depth_images(os.path.join(dir, "depth"))
    for idx in tqdm(range(len(img_dpt_paths[:100]))):
        img_rgb = cv2.imread(img_rgb_paths[idx])
        
        img_dpt = cv2.imread(img_dpt_paths[idx], -1)
        img_dpt = cv2.resize(img_dpt, (1920, 1080))

        img_rgb = rgb_depth_map.align_image_rgb(img_rgb, camera, cache)
        img_dpt = rgb_depth_map.align_image_depth(img_dpt, camera, cache)

        people_keypoints = _get_skeleton(img_rgb, mmpose)

        people_keypoints_3d = points_to_depth(
            people_keypoints, img_dpt, camera, cache)
        poses.append(people_keypoints_3d)

    return poses

# Its too long
def visualize_poses(poses):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    title = ax.set_title('3D Test')

    # Create the scatter plot
    people_keypoints = np.array(poses[0])

    num_people = len(people_keypoints)

    lines = []
    graphs = []
    for keypoints in people_keypoints:
        # Define the data for the scatter plot
        x = [point[0] for point in keypoints]
        y = [point[2] for point in keypoints]
        z = [1080 - point[1] for point in keypoints]

        graph = ax.scatter(x, y, z, c='r', marker='o')
        graphs.append(graph)
        lines.append([])
        for idx in range(len(MMPOSE_EDGES)):
            lines[-1].append(
                ax.plot(
                    (x[MMPOSE_EDGES[idx][0]], x[MMPOSE_EDGES[idx][1]]),
                    (y[MMPOSE_EDGES[idx][0]], y[MMPOSE_EDGES[idx][1]]),
                    (z[MMPOSE_EDGES[idx][0]], z[MMPOSE_EDGES[idx][1]])
                )[0]
            )

    ax.view_init(elev=1, azim=-89)

    # Remove the grid background
    ax.grid(False)

    # Set the labels for the axes
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    ax.axes.set_xlim3d(0, 1920)
    ax.axes.set_zlim3d(0, 1080)
    ax.axes.set_ylim3d(0, 3000)

    def update_graph(num):            
        people_keypoints = np.array(poses[num])
        for person, keypoints in enumerate(people_keypoints[:num_people]):
            # Define the data for the scatter plot
            x = [point[0] for point in keypoints]
            y = [point[2] for point in keypoints]
            z = [1080 - point[1] for point in keypoints]

            graphs[person]._offsets3d = (x, y, z)
            
            title.set_text('3D Test, time={}'.format(num))

            for idx, line in enumerate(lines[person]):
                line.set_data(
                    (x[MMPOSE_EDGES[idx][0]], x[MMPOSE_EDGES[idx][1]]),
                    (y[MMPOSE_EDGES[idx][0]], y[MMPOSE_EDGES[idx][1]]))
                line.set_3d_properties(
                    (z[MMPOSE_EDGES[idx][0]], z[MMPOSE_EDGES[idx][1]])
                )

    ani = matplotlib.animation.FuncAnimation(
        fig, update_graph, len(poses),
        interval=100, blit=False)

    plt.show()


# Just for test
# Clean and shorten the test
if __name__ == "__main__":
    cache = diskcache.Cache('../calibration/cache')

    cache_process = cache.get('process', {})

    mmpose = MMPoseInferencer('human')

    for expriment in EXPERIMENTS.keys():
        for dir in EXPERIMENTS[expriment]:
            camera = dir.split("/")[-1] + "_calib_snap"
            
            id_exp = f'{expriment}_{camera}'
            if not cache_process.__contains__(id_exp):
                poses = extract_poses(dir, camera)

                cache_process[id_exp] = poses
                cache['process'] = cache_process
            else:
                poses = cache_process[id_exp]

            visualize_poses(poses)
