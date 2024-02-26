import sys
sys.path.append('../')

import os
import cv2
import diskcache
import numpy as np
import matplotlib.animation
import matplotlib.pyplot as plt

from tqdm import tqdm
from utils import data_loader
from mmpose.apis import MMPoseInferencer


OVERWRITE = False
VISUALIZE = False
REMOVE_PREVIOUS = True
EXP_LENGTH = 50

def filter_sort(people_keypoints, num_select=2):
    heights = []
    for person in people_keypoints:
        person = person['keypoints']
        heights.append(person[16][1] - person[0][1])

    indecies = np.argsort(heights)[::-1]
    people_keypoints = [people_keypoints[indecies[idx]] for idx in range(num_select)]

    horizontal_position = []
    for person in people_keypoints:
        person = person['keypoints']
        horizontal_position.append(person[0][0])

    indecies = np.argsort(horizontal_position)
    people_keypoints = [people_keypoints[indecies[idx]] for idx in range(num_select)]

    return people_keypoints


def _get_skeleton(image, inferencer):
    result_generator = inferencer(image)
    
    detected_keypoints = []
    for result in result_generator:
        poeple_keypoints = filter_sort(result['predictions'][0])
        for predictions in poeple_keypoints:
            keypoints = predictions['keypoints']
            detected_keypoints.append(keypoints)

    return np.array(detected_keypoints)


def extract_poses(dir0, dir1, cam0, cam1, model, cache):
    poses = []
    
    img_rgb0_paths = data_loader.list_rgb_images(os.path.join(dir0, "color"))
    img_rgb1_paths = data_loader.list_rgb_images(os.path.join(dir1, "color"))
    
    for idx in tqdm(range(len(img_rgb0_paths[:EXP_LENGTH]))):
        img_rgb0 = cv2.imread(img_rgb0_paths[idx])
        img_rgb0 = data_loader.downsample_keep_aspect_ratio(
            img_rgb0,
            (data_loader.IMAGE_INFRARED_WIDTH, data_loader.IMAGE_INFRARED_HEIGHT))
        img_rgb1 = cv2.imread(img_rgb1_paths[idx])
        img_rgb1 = data_loader.downsample_keep_aspect_ratio(
            img_rgb1,
            (data_loader.IMAGE_INFRARED_WIDTH, data_loader.IMAGE_INFRARED_HEIGHT))

        people_keypoints0 = _get_skeleton(img_rgb0, model)[:2]
        people_keypoints1 = _get_skeleton(img_rgb1, model)[:2]

        num_points = people_keypoints1.shape[1]

        mtx0 = cache['extrinsics'][cam0]['mtx_l']
        dist0 = cache['extrinsics'][cam0]['dist_l']
        mtx1 = cache['extrinsics'][cam0]['mtx_r']
        dist1 = cache['extrinsics'][cam0]['dist_r']
        R = cache['extrinsics'][cam0]['rotation']
        T = cache['extrinsics'][cam0]['transition']

        I = np.eye(3)
        Z = np.zeros(3)
        RT = np.concatenate((I, Z.reshape(3, 1)), axis=1)
        P1 = np.dot(mtx0, RT)

        RT = np.concatenate((R, T.reshape(3, 1)), axis=1)
        P2 = np.dot(mtx1, RT)

        # Triangulate the 3D point
        point_3d_01 = cv2.triangulatePoints(P1, P2, people_keypoints0.reshape(-1, 1, 2), people_keypoints1.reshape(-1, 1, 2))

        # Convert from homogeneous to Euclidean coordinates
        point_3d_01 = cv2.convertPointsFromHomogeneous(point_3d_01.T)

        point_3d_01 = point_3d_01.reshape((-1, num_points, 3))

        poses.append(point_3d_01)

    return poses

# Its too long
# Make it also more robust
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
        z = [point[1] for point in keypoints]
        x.append(0)
        y.append(0)
        z.append(0)

        graph = ax.scatter(x, y, z, c='r', marker='o')
        graphs.append(graph)
        lines.append([])
        for idx in range(len(data_loader.MMPOSE_EDGES)):
            lines[-1].append(
                ax.plot(
                    (x[data_loader.MMPOSE_EDGES[idx][0]],
                     x[data_loader.MMPOSE_EDGES[idx][1]]),
                    (y[data_loader.MMPOSE_EDGES[idx][0]],
                     y[data_loader.MMPOSE_EDGES[idx][1]]),
                    (z[data_loader.MMPOSE_EDGES[idx][0]],
                     z[data_loader.MMPOSE_EDGES[idx][1]])
                )[0]
            )

    # ax.view_init(elev=1, azim=-89)

    # Remove the grid background
    ax.grid(False)

    # Set the labels for the axes
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    scale_min = int(np.min(keypoints))
    scale_max = int(np.max(keypoints))

    ax.axes.set_xlim3d(scale_min, scale_max)
    ax.axes.set_zlim3d(scale_min, scale_max)
    ax.axes.set_ylim3d(scale_min, scale_max)

    def update_graph(num):            
        people_keypoints = np.array(poses[num])
        for person, keypoints in enumerate(people_keypoints[:num_people]):
            # Define the data for the scatter plot
            x = [point[0] for point in keypoints]
            y = [point[2] for point in keypoints]
            z = [point[1] for point in keypoints]
            x.append(0)
            y.append(0)
            z.append(0)

            graphs[person]._offsets3d = (x, y, z)
            
            title.set_text('3D Test, time={}'.format(num))

            for idx, line in enumerate(lines[person]):
                line.set_data(
                    (x[data_loader.MMPOSE_EDGES[idx][0]],
                     x[data_loader.MMPOSE_EDGES[idx][1]]),
                    (y[data_loader.MMPOSE_EDGES[idx][0]],
                     y[data_loader.MMPOSE_EDGES[idx][1]]))
                line.set_3d_properties(
                    (z[data_loader.MMPOSE_EDGES[idx][0]],
                     z[data_loader.MMPOSE_EDGES[idx][1]])
                )

    ani = matplotlib.animation.FuncAnimation(
        fig, update_graph, len(poses),
        interval=100, blit=False)

    plt.show()


# Just for test
# Clean and shorten the test
if __name__ == "__main__":
    cache = diskcache.Cache('../calibration/cache')

    if REMOVE_PREVIOUS:
        cache['process'] = {}  # Removes all the caches processes
    
    cache_process = cache.get('process', {})

    mmpose = MMPoseInferencer('rtmpose-x_8xb256-700e_body8-halpe26-384x288')

    cam_pairs = [
        ("azure_kinect3_4_calib_snap", "azure_kinect3_5_calib_snap"),
        ("azure_kinect2_4_calib_snap", "azure_kinect1_5_calib_snap"),
    ]

    for expriment in data_loader.EXPERIMENTS.keys():
        for cam_pair in cam_pairs:
            cam0 = cam_pair[0]
            cam1 = cam_pair[1]
            dir0 = data_loader.EXPERIMENTS[expriment][cam0]
            dir1 = data_loader.EXPERIMENTS[expriment][cam1]
            
            id_exp = f'{expriment}_{cam0}_{cam1}_skeleton_3D'
            if not cache_process.__contains__(id_exp) or OVERWRITE:
                poses = extract_poses(dir0, dir1, cam0, cam1, mmpose, cache)

                cache_process[id_exp] = poses
                cache['process'] = cache_process
            else:
                poses = cache_process[id_exp]

            if VISUALIZE:
                visualize_poses(poses=poses)
