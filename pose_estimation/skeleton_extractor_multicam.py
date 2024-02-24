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
    # mtx_dpt = intrinsics[camera + '_infrared']['mtx']
    # dist_dpt = intrinsics[camera + '_infrared']['dist']

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

    return np.array(poses)

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

    # min_value = np.min(people_keypoints)
    # max_value = np.max(people_keypoints)

    # ax.axes.set_xlim3d(min_value, max_value)
    # ax.axes.set_zlim3d(min_value, max_value)
    # ax.axes.set_ylim3d(min_value, max_value)

    def update_graph(num):            
        people_keypoints = np.array(poses[num])
        for person, keypoints in enumerate(people_keypoints[:num_people]):
            # Define the data for the scatter plot
            x = [point[0] for point in keypoints]
            y = [point[2] for point in keypoints]
            z = [point[1] for point in keypoints]

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


# Generated by Gemini
def transform_points(points, camera_extrinsics):
    """
    Transforms 3D points from one camera perspective to another camera's reference frame.

    Args:
        points: A numpy array of shape (N, 3) representing the 3D points in the first camera's reference frame.
        camera_extrinsics: A numpy array of shape (4, 4) representing the extrinsic parameters of the second camera.

    Returns:
        A numpy array of shape (N, 3) representing the transformed 3D points in the second camera's reference frame.
    """

    # Extract rotation and translation from extrinsic parameters
    rotation_matrix = camera_extrinsics[:3, :3]
    translation_vector = camera_extrinsics[:3, 3]

    # Transform points
    transformed_points = np.dot(rotation_matrix, points.T).T + translation_vector

    return transformed_points


def get_parameters(cam, cache):
    if cam == cam24:
        mtx = cache['extrinsics'][cam24 + 'infrared']['mtx_l']
    elif cam == cam15:
        mtx = cache['extrinsics'][cam24 + 'infrared']['mtx_r']
    elif cam == cam14:
        mtx = cache['extrinsics'][cam15 + 'infrared']['mtx_r']
    elif cam == cam34:
        mtx = cache['extrinsics'][cam14 + 'infrared']['mtx_r']
    elif cam == cam35:
        mtx = cache['extrinsics'][cam34 + 'infrared']['mtx_r']

    R = cache['extrinsics'][cam24 + 'infrared']['rotation']
    T = cache['extrinsics'][cam24 + 'infrared']['transition']
    R2 = cache['extrinsics'][cam15 + 'infrared']['rotation']
    T2 = cache['extrinsics'][cam15 + 'infrared']['transition']
    R3 = cache['extrinsics'][cam14 + 'infrared']['rotation']
    T3 = cache['extrinsics'][cam14 + 'infrared']['transition']
    R4 = cache['extrinsics'][cam34 + 'infrared']['rotation']
    T4 = cache['extrinsics'][cam34 + 'infrared']['transition']
    
    extrinsics = np.identity(4)
    if cam == cam24:
        r = np.array([[1, 0, 0],
                     [0, 1, 0],
                     [0, 0, 1]])
        t = np.array([0, 0, 0])
        extrinsics[:3, :3] = r
        extrinsics[:3, 3] = t.reshape(3)
    elif cam == cam15:
        extrinsics[:3, :3] = R
        extrinsics[:3, 3] = T.reshape(3)
    elif cam == cam14:
        R2_com = np.dot(R2, R)
        T2_com = (np.dot(R2, T).reshape(3, 1) + T2).reshape(3,)
        extrinsics[:3, :3] = R2_com
        extrinsics[:3, 3] = T2_com / 1000
    elif cam == cam34:
        R2_com = np.dot(R2, R)
        T2_com = (np.dot(R2, T).reshape(3, 1) + T2).reshape(3,)
        R3_com = np.dot(R3, R2_com)
        T3_com = (np.dot(R3, T2_com).reshape(3, 1) + T3).reshape(3,)
        extrinsics[:3, :3] = R3_com
        extrinsics[:3, 3] = T3_com
    elif cam == cam35:
        R2_com = np.dot(R2, R)
        T2_com = (np.dot(R2, T).reshape(3, 1) + T2).reshape(3,)
        R3_com = np.dot(R3, R2_com)
        T3_com = (np.dot(R3, T2_com).reshape(3, 1) + T3).reshape(3,)
        R4_com = np.dot(R4, R3_com)
        T4_com = (np.dot(R4, T3_com).reshape(3, 1) + T4).reshape(3,)
        extrinsics[:3, :3] = R4_com
        extrinsics[:3, 3] = T4_com / 1000

    return mtx, extrinsics


cam24 = 'azure_kinect2_4_calib_snap'
cam15 = 'azure_kinect1_5_calib_snap'
cam14 = 'azure_kinect1_4_calib_snap'
cam34 = 'azure_kinect3_4_calib_snap'
cam35 = 'azure_kinect3_5_calib_snap'
# Just for test
# Clean and shorten the test
if __name__ == "__main__":
    cache = diskcache.Cache('../calibration/cache')

    cache_process = cache.get('process', {})

    mmpose = MMPoseInferencer('rtmpose-x_8xb256-700e_body8-halpe26-384x288')

    cameras = [
        cam24,
        cam15,
        # cam14,
        cam34,
        # cam35
    ]

    poses_global = None
    for expriment in data_loader.EXPERIMENTS.keys():
        for camera in cameras:
            dir = data_loader.EXPERIMENTS[expriment][camera]
            
            id_exp = f'{expriment}_{camera}_skeleton_3D'
            
            print(f'Extracting skeletons for {expriment} and {camera}')
            max_people = 1 if camera == 'azure_kinect1_4_calib_snap' else 2
            poses = extract_poses(dir, camera, mmpose, max_people)

            mtx, extrinsics = get_parameters(camera, cache)

            # poses[:,:,:,1] /= 1000.
            # poses[:,:,:,0] = (poses[:,:,:,0] - mtx[0, 2]) * poses[:,:,:,1] / mtx[0, 0]
            # poses[:,:,:,2] = (poses[:,:,:,2] - mtx[1, 2]) * poses[:,:,:,1] / mtx[1, 1]

            print(poses.shape)
            print(extrinsics.shape)

            # TODO: Visualize with Open3D
            
            if poses_global is None:
                visualize_poses(poses=poses)
            else:
                visualize_poses(poses=np.concatenate((poses_global, poses), axis=1))

            points_homogenous = np.hstack([poses.reshape(-1, 3), np.ones((poses.reshape(-1, 3).shape[0], 1))])
            transformed_points_homogenous = np.dot(np.linalg.inv(extrinsics), points_homogenous.T)
            poses = transformed_points_homogenous[:3, :].T.reshape(poses.shape)

            # poses = transform_points(poses.reshape(-1, 3), extrinsics).reshape(poses.shape)

            if poses_global is None:
                poses_global = poses
            else:
                # poses_global = np.concatenate((poses_global, poses), axis=1)
                poses_global = (poses_global + poses) / 2
            
            print(poses_global.shape)
            visualize_poses(poses=poses_global)

        break