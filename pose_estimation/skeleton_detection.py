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
        required=True,
    )

    args = parser.parse_args()

    return args


def _load_configs(path):
    with open(path, 'r') as yaml_file:
        configs = yaml.safe_load(yaml_file)

    return configs


def filter_sort(people_keypoints, num_select=2):
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
    people_keypoints = [people_keypoints[indecies[idx]]
                        for idx in range(num_select)]

    return people_keypoints


def _get_skeleton(image, inferencer, max_people, configs):
    result_generator = inferencer(image)
    
    detected_keypoints = []
    detected_confidences = []
    for result in result_generator:
        poeple_keypoints = filter_sort(result['predictions'][0],
                                       num_select=max_people)
        for predictions in poeple_keypoints:
            keypoints = predictions['keypoints']
            # Divided by 10 to normalize between 0 and 1
            # TODO: Clear the mess here
            confidences = (np.array(predictions['keypoint_scores']) \
                           / configs['confidence_coeff']).tolist()
            
            detected_keypoints.append(keypoints)
            detected_confidences.append(confidences)

    return detected_keypoints, detected_confidences


def extract_poses(dir, camera, model, intrinsics, max_people,
                  configs):
    mtx = np.array(intrinsics[camera]['mtx'], np.float32)
    dist = np.array(intrinsics[camera]['dist'], np.float32)

    poses = []
    poses_confidence = []

    cap = cv2.VideoCapture(dir)
    for _ in range(configs['calibration_folders'][camera]['offset']):
        cap.grab()

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    search_depth = min(frame_count, configs['experiment_length'])

    bar = tqdm(range(search_depth))
    bar.set_description(camera)
    for idx in bar:
        _, img_rgb = cap.read()

        img_rgb = cv2.undistort(img_rgb, mtx, dist, None, None)

        people_keypoints, confidences = _get_skeleton(
            img_rgb, model, max_people, configs)
        if configs['visualize']:
            visualize_keypoints(img_rgb, people_keypoints, confidences)

        poses.append(people_keypoints)
        poses_confidence.append(confidences)

    return poses, poses_confidence


def visualize_keypoints(image, keypoints, confidences):
    for idx_person, person in enumerate(keypoints):
        for idx_point, point in enumerate(person):
            cv2.circle(image, (int(point[0]), int(point[1])),
                    5, (0, 255, 0), -1)
            cv2.putText(image, str(idx_person),
                (int(point[0]), int(point[1])),
                cv2.FONT_HERSHEY_SIMPLEX,
                1, (255, 255, 255), 1, 2)
            # cv2.putText(image, str(round(confidences[idx_person][idx_point], 2)),
            #     (int(point[0]), int(point[1])),
            #     cv2.FONT_HERSHEY_SIMPLEX,
            #     1, (255, 255, 255), 1, 2)
        
    cv2.imshow("Detected", cv2.resize(image, (1280, 720)))
    cv2.waitKey(1)


def _store_artifacts(artifact, output):
    with open(output, 'w') as handle:
        yaml.dump(artifact, handle)


def calc_2d_skeleton(cameras, model_2d, configs):
    with open(configs['intrinsics']) as handler:
        intrinsics = yaml.safe_load(handler)

    keypoints = {}
    for _, camera in enumerate(cameras):
        dir = configs['calibration_folders'][camera]['path']

        max_people=1
        pose, pose_confidence = extract_poses(
            dir, camera, model_2d, intrinsics, max_people, configs)
        
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
    cameras = configs["calibration_folders"].keys()
        
    keypoints = calc_2d_skeleton(cameras, model_2d, configs)
    
    _store_artifacts(
        keypoints,
        configs['output'])
