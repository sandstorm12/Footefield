import sys
sys.path.append('../')

import os
import cv2
import pickle
import diskcache
import numpy as np
import matplotlib.animation
import matplotlib.pyplot as plt

from tqdm import tqdm
from utils import data_loader
from calibration import rgb_depth_map
from mmpose.apis import MMPoseInferencer


VISUALIZE = False
EXP_LENGTH = 50


def filter_sort(people_keypoints, num_select=2, invert=False):
    heights = []
    for person in people_keypoints:
        person = person['keypoints']
        heights.append(person[16][1] - person[0][1])

    indecies = np.argsort(heights)[::-1]
    people_keypoints = [people_keypoints[indecies[idx]]
                        for idx in range(num_select)]

    horizontal_position = []
    for person in people_keypoints:
        person = person['keypoints']
        horizontal_position.append(person[0][0])

    indecies = np.argsort(horizontal_position)
    if invert:
        indecies = indecies[::-1]
    people_keypoints = [people_keypoints[indecies[idx]]
                        for idx in range(num_select)]

    return people_keypoints


def _get_skeleton(image, inferencer, max_people=2):
    result_generator = inferencer(image)
    
    detected_keypoints = []
    for result in result_generator:
        poeple_keypoints = filter_sort(result['predictions'][0],
                                       num_select=max_people)
        for predictions in poeple_keypoints:
            keypoints = predictions['keypoints']
            detected_keypoints.append(keypoints)

    return np.array(detected_keypoints)


def extract_poses(dir, camera, model, max_people=2):
    cache = diskcache.Cache('../calibration/cache')
    intrinsics = cache.get("intrinsics", None)
    mtx_dpt = intrinsics[camera + '_infrared']['mtx']
    dist_dpt = intrinsics[camera + '_infrared']['dist']

    poses = []

    img_rgb_paths = data_loader.list_rgb_images(os.path.join(dir, "color"))
    img_dpt_paths = data_loader.list_depth_images(os.path.join(dir, "depth"))
    for idx in tqdm(range(len(img_dpt_paths[:EXP_LENGTH]))):
        img_rgb = cv2.imread(img_rgb_paths[idx])
        img_rgb = data_loader.downsample_keep_aspect_ratio(
            img_rgb,
            (data_loader.IMAGE_INFRARED_WIDTH, data_loader.IMAGE_INFRARED_HEIGHT))
        img_dpt = cv2.imread(img_dpt_paths[idx], -1)

        img_rgb = rgb_depth_map.align_image_rgb(img_rgb, camera, cache)

        # # Image undistortion but with nearest interpolation
        # # for more accurate depth value
        # mapx, mapy = cv2.initUndistortRectifyMap(
        #     mtx_dpt, dist_dpt, None,
        #     mtx_dpt,
        #     (640, 576), cv2.CV_32FC1)
        # img_dpt = cv2.remap(img_dpt, mapx, mapy, cv2.INTER_NEAREST)

        # img_rgb = cv2.undistort(img_rgb, mtx_dpt, dist_dpt, None, None)

        people_keypoints = _get_skeleton(img_rgb, model, max_people)

        people_keypoints_3d = rgb_depth_map.points_to_depth(
            people_keypoints, img_dpt)
        poses.append(people_keypoints_3d)

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
        z = [1080 - point[1] for point in keypoints]

        graph = ax.scatter(x, y, z, c='r', marker='o')
        graphs.append(graph)
        lines.append([])
        for idx in range(len(data_loader.HALPE_EDGES)):
            lines[-1].append(
                ax.plot(
                    (x[data_loader.HALPE_EDGES[idx][0]],
                     x[data_loader.HALPE_EDGES[idx][1]]),
                    (y[data_loader.HALPE_EDGES[idx][0]],
                     y[data_loader.HALPE_EDGES[idx][1]]),
                    (z[data_loader.HALPE_EDGES[idx][0]],
                     z[data_loader.HALPE_EDGES[idx][1]])
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
                    (x[data_loader.HALPE_EDGES[idx][0]],
                     x[data_loader.HALPE_EDGES[idx][1]]),
                    (y[data_loader.HALPE_EDGES[idx][0]],
                     y[data_loader.HALPE_EDGES[idx][1]]))
                line.set_3d_properties(
                    (z[data_loader.HALPE_EDGES[idx][0]],
                     z[data_loader.HALPE_EDGES[idx][1]])
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

    mmpose = MMPoseInferencer('rtmpose-x_8xb256-700e_body8-halpe26-384x288')

    for expriment in data_loader.EXPERIMENTS.keys():
        for exp_cam in data_loader.EXPERIMENTS[expriment].keys():
            dir = data_loader.EXPERIMENTS[expriment][exp_cam]
            
            id_exp = f'{expriment}_{exp_cam}_skeleton_3D'
            
            print(f'Extracting skeletons for {expriment} and {exp_cam}')
            max_people = 1 if exp_cam == 'azure_kinect1_4_calib_snap' else 2
            poses = extract_poses(dir, exp_cam, mmpose, max_people)

            if VISUALIZE:
                visualize_poses(poses=poses)

            # TODO: Move this to a store function
            STORE_DIR = "./keypoints_3d"
            if not os.path.exists(STORE_DIR):
                os.mkdir(STORE_DIR)
            store_path = os.path.join(STORE_DIR,
                                      'keypoints3d_{}_{}.pkl'.format(expriment, exp_cam))
            with open(store_path, 'wb') as handle:
                pickle.dump(np.array(poses), handle, protocol=pickle.HIGHEST_PROTOCOL)
