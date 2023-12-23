import sys
sys.path.append('../')

import cv2
import diskcache
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from utils import data_loader


DISPARITY = -18


def get_skeleton(image, inferencer, visualize=False):
    result_generator = inferencer(image)
    
    detected_keypoints = []
    for result in result_generator:
        for predictions in result['predictions'][0]:
            keypoints = predictions['keypoints']
            detected_keypoints.append(keypoints)
            for idx, point in enumerate(keypoints):

                x = int(point[0])
                y = int(point[1])

                cv2.circle(
                    image, (x, y), 10, (0),
                    thickness=-1, lineType=8)

                cv2.putText(
                    image, str(idx), (x - 5, y + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, .5, (255), thickness=2)
    
    if visualize:
        cv2.imshow('frame', image)
        cv2.waitKey(0)

    return np.array(detected_keypoints)


def points_to_depth(mmpose, image_rgb, image_inf, cache):
    # points1 = keypoints.reshape((-1, 1, 2))

    # print("keypoints", points1.shape)

    cam_1 = 'azure_kinect3_4_calib_snap'

    if not cache.__contains__('depth_matching'):
        raise Exception('Depth matching not cached. '
                        'Run rgb_depth_calibration script.')
    
    map1x = cache['depth_matching'][cam_1]['map_rgb_x']
    map1y = cache['depth_matching'][cam_1]['map_rgb_y']
    map2x = cache['depth_matching'][cam_1]['map_infrared_x']
    map2y = cache['depth_matching'][cam_1]['map_infrared_y']
    
    image_rgb = cv2.remap(image_rgb, map1x, map1y, cv2.INTER_LANCZOS4)

    # Remove magic number .8
    image_inf = cv2.resize(
        image_inf,
        (data_loader.IMAGE_RGB_WIDTH, data_loader.IMAGE_RGB_HEIGHT))
    image_inf = cv2.remap(image_inf, map2x, map2y, cv2.INTER_LANCZOS4)

    # Add the dispartiy between RGB and INFRARED cameras
    image_inf = np.roll(image_inf, DISPARITY, axis=1)
    image_inf = np.clip(
            image_inf.astype(np.float32) * .1, 0, 255).astype('uint8')
    
    keypoints = get_skeleton(image_rgb, mmpose, visualize=False)
    
    for person_keypoints in keypoints:
        for idx, point in enumerate(person_keypoints):
            x = int(point[0])
            y = int(point[1])

            cv2.circle(
                image_inf, (x, y), 10, (0, 0, 0),
                thickness=-1, lineType=8)

            cv2.putText(
                image_inf, str(idx), (x - 5, y + 5),
                cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), thickness=2)
    
    image_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2GRAY)
    img_cmb = (image_rgb * .5 + image_inf * .5).astype(np.uint8)

    # cv2.imshow("CMB", image_inf)
    # cv2.waitKey(0)

    # cv2.circle(
    #     image_1, (x, y), 10, (0, 0, 0),
    #     thickness=-1, lineType=8)
    
    # # image_2 = np.clip(image_2.astype(np.float32) * 2., 0, 255).astype('uint8')
    # cv2.circle(
    #     image_2, (int(map2x[x, y]), int(map2y[x, y])), 10, (100, 100, 100),
    #     thickness=-1, lineType=8)
    
    # # print()

    f, axarr = plt.subplots(1, 2)
    axarr[0].imshow(image_rgb)
    axarr[1].imshow(image_inf)

    # plt.imshow(image_inf)
    # plt.imshow(image_rgb)
    plt.show()

    # # Print the result
    # print(f"The corresponding point in the right image is ({point_right})")


# Just for test
if __name__ == "__main__":
    cache = diskcache.Cache('cache')

    from mmpose.apis import MMPoseInferencer
    mmpose = MMPoseInferencer('human')

    img_rgb_path = '/home/hamid/Documents/footefield/data/AzureKinectRecord_0729/a1/azure_kinect3_4/color/color00000.jpg'
    img_dpt_path = '/home/hamid/Documents/footefield/data/AzureKinectRecord_0729/a1/azure_kinect3_4/depth/depth00000.png'
    img_rgb = cv2.imread(img_rgb_path)
    img_dpt = cv2.imread(img_dpt_path, -1)
    img_dpt = cv2.resize(img_dpt, (1920, 1080))

    # keypoints = get_skeleton(img_rgb, mmpose, visualize=False)

    points_to_depth(mmpose, img_rgb, img_dpt, cache)

    # img_dpt = np.clip(img_dpt.astype(np.float32) * 2, 0, 255).astype('uint8')
    # cv2.imshow("Depth", img_dpt)
    # cv2.waitKey(0)
