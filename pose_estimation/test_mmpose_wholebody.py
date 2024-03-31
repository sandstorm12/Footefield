import sys
sys.path.append('../')

import cv2
import numpy as np

from utils import data_loader
from mmpose.apis import MMPoseInferencer

# INDEX_BODY = [1~17] 17
# INDEX_FEET = [18~23] 6
# INDEX_FACE = [24~91] 68
# INDEX_HANDS = [92~133] 42

dir = "/home/hamid/Documents/phd/footefield/data/AzureKinectRecord_0729/a1/azure_kinect1_4/color"
img_rgb_paths = data_loader.list_rgb_images(dir)

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


# inferencer = MMPoseInferencer('human')
inferencer = MMPoseInferencer('rtmpose-l_8xb32-270e_coco-wholebody-384x288')

for img_path in img_rgb_paths:
    image = cv2.imread(img_path)
    # image = data_loader.downsample_keep_aspect_ratio(
    #     image,
    #     (data_loader.IMAGE_INFRARED_WIDTH, data_loader.IMAGE_INFRARED_HEIGHT))

    result_generator = inferencer(image)
    for result in result_generator:
        poeple_keypoints = filter_sort(result['predictions'][0])
        for idx_person, predictions in enumerate(poeple_keypoints):
            print(predictions.keys())
            keypoints = predictions['keypoints']
            keypoints = keypoints[91:]
            keypoints_score = predictions['keypoint_scores'][91:]
            for idx, point in enumerate(keypoints):
                x = int(point[0])
                y = int(point[1])

                if keypoints_score[idx] > .7:
                    cv2.circle(
                        image, (x, y), 4, (255, 255, 255),
                        thickness=-1, lineType=8)

                # cv2.putText(
                #     image, str(idx_person) + '_' + str(idx), (x - 5, y + 5),
                #     cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), thickness=2)
                
    cv2.imshow('frame', image)
    if cv2.waitKey(0) == ord('q'):
        break
