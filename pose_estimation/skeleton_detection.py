import sys
sys.path.append('../')

import cv2
import yaml
import torch
import argparse
import numpy as np

from tqdm import tqdm
from queue import Queue
from mmpose.apis import MMPoseInferencer
from transformers import ViTModel, ViTImageProcessor


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


def filter_sort(image, people_keypoints, feature_store, configs):
    people_keypoints = [person
                        for person in people_keypoints
                        if person['bbox_score'] > configs['detect_threshold'] and \
                            person['bbox'][0][3] - person['bbox'][0][1] > configs['detect_min_height'] and \
                            person['bbox'][0][0] > configs['detect_min_x'] and \
                            person['bbox'][0][2] < configs['detect_max_x']]
    people_keypoints = sorted(
        people_keypoints,
        key=lambda x: abs(((x['bbox'][0][0] + x['bbox'][0][2]) / 2) - 960) + (1080 - x['bbox'][0][3]),
        reverse=False)
    
    match_indices = []
    for idx_person, person in enumerate(people_keypoints):
        bbox = person['bbox'][0]
        x0, y0, x1, y1 = map(int, bbox)
        image_person = image[y0:y1, x0:x1]

        # TODO: Use the model from input arguments
        inputs = processor(images=image_person, return_tensors="pt").to(device)
        outputs = model(**inputs)
        last_hidden_states = outputs.last_hidden_state[0]
        feature_current = last_hidden_states[0].cpu().detach().numpy()

        similarities = []
        for features_queue in feature_store:
            features = list(features_queue.queue)
            
            scores = []
            for feature in features:
                dot_product = np.dot(feature_current, feature)
                cosine_similarity = dot_product / \
                    (np.linalg.norm(feature_current) * np.linalg.norm(feature))
                scores.append(cosine_similarity)

            scores = np.array(scores)
            similarities.append(np.max(scores) + (len(features) / 5 * .1))

        if len(similarities) > 0 and \
                similarities[np.argmax(similarities)] > configs['track_threshold'] and \
                np.argmax(similarities) not in match_indices:
            match_index = np.argmax(similarities).item()
            feature_store[match_index].put(feature_current)
            if feature_store[match_index].qsize() > 5:
                feature_store[match_index].get()
        else:
            feature_store.append(Queue())
            feature_store[-1].put(feature_current)
            match_index = len(feature_store) - 1

        match_indices.append(match_index)

    return people_keypoints, match_indices


def _get_skeleton(image, inferencer, feature_store, configs):
    result_generator = inferencer(image)
    
    detected_keypoints = []
    detected_confidences = []
    for result in result_generator:
        poeple_keypoints, match_indcies = filter_sort(
            image, result['predictions'][0], feature_store, configs)
        for predictions in poeple_keypoints:
            keypoints = predictions['keypoints']
            # Divided by 10 to normalize between 0 and 1
            # TODO: Clear the mess here
            confidences = (np.array(predictions['keypoint_scores']) \
                           / configs['confidence_coeff']).tolist()
            
            detected_keypoints.append(keypoints)
            detected_confidences.append(confidences)

    return detected_keypoints, detected_confidences, match_indcies


def extract_poses(dir, camera, model_2d, intrinsics,
                  configs):
    mtx = np.array(intrinsics[camera]['mtx'], np.float32)
    dist = np.array(intrinsics[camera]['dist'], np.float32)

    poses = []
    poses_confidence = []
    poses_ids = []

    feature_store = []

    cap = cv2.VideoCapture(dir)

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    bar = tqdm(range(frame_count))
    bar.set_description(camera)
    for idx in bar:
        ret, img_rgb = cap.read()

        if ret:
            img_rgb = cv2.undistort(img_rgb.copy(), mtx, dist, None, None)

            people_keypoints, confidences, detected_ids = _get_skeleton(
                img_rgb, model_2d, feature_store, configs)
            if configs['visualize']:
                visualize_keypoints(img_rgb, people_keypoints, confidences, detected_ids)

            poses.append(people_keypoints)
            poses_confidence.append(confidences)
            poses_ids.append(detected_ids)

    return poses, poses_confidence, poses_ids


def visualize_keypoints(image, keypoints, confidences, detected_ids):
    for idx_person, person in enumerate(keypoints):
        for idx_point, point in enumerate(person):
            cv2.circle(image, (int(point[0]), int(point[1])),
                    5, (0, 255, 0), -1)
            cv2.putText(image, str(detected_ids[idx_person]),
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
        dir = configs['videos'][camera]['path']

        pose, pose_confidence, ids = extract_poses(
            dir, camera, model_2d, intrinsics, configs)
        
        keypoints[camera] = {
            'pose': pose,
            'pose_confidence': pose_confidence,
            'ids': ids,
        }

    return keypoints


if __name__ == "__main__":
    args = _get_arguments()
    configs = _load_configs(args.config)
    print(f"Config loaded: {configs}")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    processor = ViTImageProcessor.from_pretrained('google/vit-large-patch32-384')
    model = ViTModel.from_pretrained('google/vit-large-patch32-384')
    model.to(device)

    model_2d = MMPoseInferencer(configs["model"], device=device)
    cameras = configs["videos"].keys()
        
    keypoints = calc_2d_skeleton(cameras, model_2d, configs)
    
    _store_artifacts(
        keypoints,
        configs['output'])
