import sys
sys.path.append('../')

import os
import cv2
import pickle
import pycalib
import diskcache
import numpy as np

from tqdm import tqdm
from utils import data_loader
from calibration import rgb_depth_map
from mmpose.apis import MMPoseInferencer


VISUALIZE = True
EXP_LENGTH = 200
DIR_STORE = "./keypoints_3d"


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


def _get_skeleton(image, inferencer, max_people=2, invert=False):
    result_generator = inferencer(image)
    
    detected_keypoints = []
    for result in result_generator:
        poeple_keypoints = filter_sort(result['predictions'][0],
                                       num_select=max_people,
                                       invert=invert)
        for predictions in poeple_keypoints:
            keypoints = predictions['keypoints']
            detected_keypoints.append(keypoints)

    return np.array(detected_keypoints)


def extract_poses(dir, camera, model, max_people=2, invert=False):
    cache = diskcache.Cache('../calibration/cache')

    poses = []

    img_rgb_paths = data_loader.list_rgb_images(os.path.join(dir, "color"))
    img_dpt_paths = data_loader.list_depth_images(os.path.join(dir, "depth"))
    for idx in tqdm(range(len(img_dpt_paths[:EXP_LENGTH]))):
        img_rgb = cv2.imread(img_rgb_paths[idx])
        img_rgb = data_loader.downsample_keep_aspect_ratio(
            img_rgb,
            (data_loader.IMAGE_INFRARED_WIDTH, data_loader.IMAGE_INFRARED_HEIGHT))

        img_rgb = rgb_depth_map.align_image_rgb(img_rgb, camera, cache)

        people_keypoints = _get_skeleton(img_rgb, model, max_people, invert)

        poses.append(people_keypoints)

    return np.array(poses)


def get_intrinsics(cam, cache):
    if cam == cam24:
        mtx = cache['extrinsics'][cam24 + 'infrared']['mtx_l']
        dist = cache['extrinsics'][cam24 + 'infrared']['dist_l']
    elif cam == cam15:
        mtx = cache['extrinsics'][cam24 + 'infrared']['mtx_r']
        dist = cache['extrinsics'][cam24 + 'infrared']['dist_r']
    elif cam == cam14:
        mtx = cache['extrinsics'][cam15 + 'infrared']['mtx_r']
        dist = cache['extrinsics'][cam15 + 'infrared']['dist_r']
    elif cam == cam34:
        mtx = cache['extrinsics'][cam14 + 'infrared']['mtx_r']
        dist = cache['extrinsics'][cam14 + 'infrared']['dist_r']
    elif cam == cam35:
        mtx = cache['extrinsics'][cam34 + 'infrared']['mtx_r']
        dist = cache['extrinsics'][cam34 + 'infrared']['dist_r']

    return mtx, dist


def get_extrinsics(cam, cache):
    R = cache['extrinsics'][cam24 + 'infrared']['rotation']
    T = cache['extrinsics'][cam24 + 'infrared']['transition']
    R2 = cache['extrinsics'][cam15 + 'infrared']['rotation']
    T2 = cache['extrinsics'][cam15 + 'infrared']['transition']
    R3 = cache['extrinsics'][cam14 + 'infrared']['rotation']
    T3 = cache['extrinsics'][cam14 + 'infrared']['transition']
    R4 = cache['extrinsics'][cam34 + 'infrared']['rotation']
    T4 = cache['extrinsics'][cam34 + 'infrared']['transition']
    
    extrinsics = np.zeros((3, 4), dtype=float)
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
        extrinsics[:3, 3] = T2_com
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
        extrinsics[:3, 3] = T4_com

    return extrinsics


def get_parameters(cam, cache):
    mtx, dist = get_intrinsics(cam, cache)

    extrinsics = get_extrinsics(cam, cache)

    return mtx, dist, extrinsics


def store_poses(poses, name, store_dir):
    if not os.path.exists(store_dir):
        os.mkdir(store_dir)

    store_path = os.path.join(store_dir, name + '.pkl')
    with open(store_path, 'wb') as handle:
        pickle.dump(poses, handle, protocol=pickle.HIGHEST_PROTOCOL)


# TODO: Too long
def calc_3d_skeleton(cameras, model_2d, cache):
    points_2d = []
    camera_indices = []
    point_indices = []
    camera_params = []

    poses_multicam = []
    p_gt = []
    for idx_cam, camera in enumerate(cameras):
        dir = data_loader.EXPERIMENTS[experiment][camera]

        max_people = 1 if camera == cam14 else 2
        invert = True if camera == cam34 or camera == cam35 else False
        poses = extract_poses(dir, camera, model_2d, max_people, invert)

        poses_multicam.append(poses)

        mtx, dist, extrinsics = get_parameters(camera, cache)
        p_gt.append(mtx @ extrinsics)

        # Bundle adjustment parameters
        points_2d.extend(poses.reshape(-1, 2))
        camera_indices.extend([idx_cam] * len(poses.reshape(-1, 2)))
        point_indices.extend([i for i in range(len(poses.reshape(-1, 2)))])
        params = {
            'mtx': mtx,
            'dist': dist,
            'extrinsics': extrinsics
        }
        camera_params.append(params)

    poses_multicam = np.asarray(poses_multicam)
    p_gt = np.asarray(p_gt)

    poses_global = pycalib.triangulate_Npts(
        pt2d_CxPx2=poses_multicam.reshape(len(cameras), -1, 2), P_Cx3x4=p_gt)

    # TODO: Refactor the magic numbers
    poses_global = poses_global.reshape(EXP_LENGTH, 2, 26, 3)

    ba_parameters = {
        "poses_3d": poses_global.reshape(-1, 3),
        "points_2d": np.array(points_2d),
        "camera_indices": np.array(camera_indices),
        "point_indices": np.array(point_indices),
        "params": camera_params,
    }

    return poses_global, ba_parameters


# TODO: Move the cameras somewhere else
cam24 = 'azure_kinect2_4_calib_snap'
cam15 = 'azure_kinect1_5_calib_snap'
cam14 = 'azure_kinect1_4_calib_snap'
cam34 = 'azure_kinect3_4_calib_snap'
cam35 = 'azure_kinect3_5_calib_snap'
# TODO: Too long
if __name__ == "__main__":
    cache = diskcache.Cache('../calibration/cache')

    model_2d = MMPoseInferencer('rtmpose-x_8xb256-700e_body8-halpe26-384x288')
    
    # TODO: It is possible to add the 1_4 camera but it requires
    # some conditioning. make it happen
    cameras = [
        cam24,
        cam15,
        # cam14,
        cam34,
        cam35
    ]

    poses_global = None
    for experiment in data_loader.EXPERIMENTS.keys():
        poses_global, ba_parameters = calc_3d_skeleton(cameras, model_2d, cache)
        
        store_poses(
            poses_global,
            'keypoints3d_{}'.format(experiment),
            DIR_STORE)
        
        store_poses(
            ba_parameters,
            'keypoints3d_{}_ba'.format(experiment),
            DIR_STORE)
        
        
