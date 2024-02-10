import sys
sys.path.append('../')

import cv2
import json
import numpy as np

from tqdm import tqdm
from mmpose.apis import MMPoseInferencer


VISUALIZE = False
OUTPUT_PATH = "alphapose_keypoints.json"


path_video = "/home/hamid/Documents/phd/footefield/footefield/videos/a1_azure_kinect2_4_crop.mp4"


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


def extract_pose(frame):
    people = []

    result_generator = inferencer(frame)
    for result in result_generator:
        people_frame = []

        poeple_keypoints = filter_sort(result['predictions'][0], num_select=2)
        for idx_person, predictions in enumerate(poeple_keypoints):
            keypoints = predictions['keypoints']
            keypoint_scores = predictions['keypoint_scores']
            
            keypoints_coco = [keypoints[i] + [keypoint_scores[i]] for i in range(len(keypoints))]
            keypoints_coco = [item
                              for point in keypoints_coco
                              for item in point]
            people_frame.append(keypoints_coco)

        people.append(people_frame)

    return people


def visualize_points(people, frame):
    for idx_person, person in enumerate(people):
        for idx in range(len(person) // 3):
            x = int(person[idx * 3 + 0])
            y = int(person[idx * 3 + 1])

            cv2.circle(
                frame, (x, y), 10, (0, 0, 0),
                thickness=-1, lineType=8)

            cv2.putText(
                frame, str(idx_person) + '_' + str(idx), (x - 5, y + 5),
                cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), thickness=2)


if __name__ == "__main__":
    alphapose_data = []

    inferencer = MMPoseInferencer('rtmpose-x_8xb256-700e_body8-halpe26-384x288')

    cap = cv2.VideoCapture(path_video)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for idx_frame in tqdm(range(length)):
        ret, frame = cap.read()

        if not ret:
            break

        people = extract_pose(frame)

        for people_frame in people:
            for idx_person, person in enumerate(people_frame):
                alphapose_data.append(
                    {
                        "image_id": f"image_{idx_frame}",
                        "category_id": 1,
                        "idx": idx_person,
                        "keypoints": person,
                        "score": 1.0
                    }
                )

        if VISUALIZE:
            visualize_points(people, frame)
            
            cv2.imshow("Frame", frame)
            if cv2.waitKey(1) == ord('q'):
                break

    with open(OUTPUT_PATH, 'w') as handle:
	    json.dump(alphapose_data, handle, indent=4)
