import sys
sys.path.append('../')

import cv2
import yaml
import argparse
import numpy as np

from tqdm import tqdm
from utils import data_loader
from mmpose.apis import MMPoseInferencer


def _get_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-c', '--config',
        help='Path to the config file',
        type=str,
        default='configs/skeleton_detection.yml',
    )

    args = parser.parse_args()

    return args


def _load_configs(path):
    with open(path, 'r') as yaml_file:
        configs = yaml.safe_load(yaml_file)

    return configs


def filter_sort(people_keypoints, num_select=2, invert=False):
    heights = []
    for person in people_keypoints:
        person = np.array(person['keypoints'])
        heights.append(
            (np.max(person[:, 1]) - np.min(person[:, 1])) *
            (np.max(person[:, 0]) - np.min(person[:, 0])) *
            (700 -  np.linalg.norm(np.mean(person[:], axis=0) - (1920 / 2, 1080 / 2)))
        )

    indecies = np.argsort(heights)[::-1]
    people_keypoints = [people_keypoints[indecies[idx]]
                        for idx in range(num_select)]

    horizontal_position = []
    for person in people_keypoints:
        person = person['keypoints']
        horizontal_position.append(person[0][0])

    indecies = np.argsort(horizontal_position)[::-1]
    if invert:
        indecies = indecies[::-1]
    people_keypoints = [people_keypoints[indecies[idx]]
                        for idx in range(num_select)]

    return people_keypoints


def _get_skeleton(image, inferencer, max_people=2, invert=False):
    result_generator = inferencer(image)
    
    detected_keypoints = []
    detected_confidences = []
    for result in result_generator:
        poeple_keypoints = filter_sort(result['predictions'][0],
                                       num_select=max_people,
                                       invert=invert)
        for predictions in poeple_keypoints:
            keypoints = predictions['keypoints']
            confidences = predictions['keypoint_scores']
            detected_keypoints.append(keypoints)
            detected_confidences.append(confidences)

    return detected_keypoints, detected_confidences


def extract_poses(dir, camera, model, intrinsics, max_people, invert,
                  configs):
    mtx = np.array(intrinsics[camera]['mtx'], np.float32)
    dist = np.array(intrinsics[camera]['dist'], np.float32)

    poses = []
    poses_confidence = []

    img_rgb_paths = data_loader.list_rgb_images(dir)
    for idx in tqdm(range(len(img_rgb_paths[:configs['experiment_length']]))):
        img_rgb = cv2.imread(img_rgb_paths[idx])

        img_rgb = cv2.undistort(img_rgb, mtx, dist, None, None)

        people_keypoints, confidences = _get_skeleton(
            img_rgb, model, max_people, invert)
        if configs['visualize']:
            visualize_keypoints(img_rgb, people_keypoints)

        poses.append(people_keypoints)
        poses_confidence.append(confidences)

    return poses, poses_confidence


def visualize_keypoints(image, keypoints):
    for idx_person, person in enumerate(keypoints):
        for point in person:
            cv2.circle(image, (int(point[0]), int(point[1])),
                    5, (0, 255, 0), -1)
            cv2.putText(image, str(idx_person), 
                (int(point[0]), int(point[1])), 
                cv2.FONT_HERSHEY_SIMPLEX,
                1, (255, 255, 255), 1, 2)
        
    cv2.imshow("Detected", cv2.resize(image, (1280, 720)))
    cv2.waitKey(1)


def _store_artifacts(artifact, output):
    with open(output, 'w') as handle:
        yaml.dump(artifact, handle)


def calc_2d_skeleton(cameras, model_2d, configs):
    with open(configs['intrinsics']) as handler:
        intrinsics = yaml.safe_load(handler)

    keypoints = {}
    for idx_cam, camera in enumerate(cameras):
        dir = configs['calibration_folders'][idx_cam]['path']

        max_people = 1 if camera == "cam1_4" else 2
        invert = True if camera == "cam3_4" or camera == "cam3_5" else False
        pose, pose_confidence = extract_poses(
            dir, camera, model_2d, intrinsics, max_people, invert, configs)
        
        keypoints[camera] = {
            'pose': pose,
            'pose_confidence': pose_confidence
        }

    return keypoints


if __name__ == "__main__":
    args = _get_arguments()
    configs = _load_configs(args.config)

    print(f"Config loaded: {configs}")

    model_2d = MMPoseInferencer(configs["model"])
    cameras = [item['camera_name'] for item in configs["calibration_folders"]]
        
    keypoints = calc_2d_skeleton(cameras, model_2d, configs)
    
    _store_artifacts(
        keypoints,
        configs['output'])
