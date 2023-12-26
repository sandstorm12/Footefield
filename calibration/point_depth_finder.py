import sys
sys.path.append('../')

import cv2
import diskcache
import numpy as np
import matplotlib.pyplot as plt

from utils import data_loader


DISPARITY = -18


# Just for test
def _get_skeleton(image, inferencer):
    result_generator = inferencer(image)
    
    detected_keypoints = []
    for result in result_generator:
        for predictions in result['predictions'][0]:
            keypoints = predictions['keypoints']
            detected_keypoints.append(keypoints)

    return np.array(detected_keypoints)


def _map(x, y, mapx, mapy):
    i = x
    j = y

    # Calculate the distance of original point and out guessed point
    delta_old = abs(mapx[y, x] - x) + abs(mapy[y, x] - y)
    while True:
        next_i = i
        next_j = j
        # Searching the 8 neighbour points for the smallest distance
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                # Make sure we don't go out of the image using a max-min
                search_point_x = max(min(i+dx, mapx.shape[1] - 1), 0)
                search_point_y = max(min(j+dy, mapx.shape[0] - 1), 0)
                delta_x = abs(mapx[search_point_y, search_point_x] - x)
                delta_y = abs(mapy[search_point_y, search_point_x] - y)
                # If the distance of the new point was less than
                # the distance of the older point, replace the point
                if delta_old >= delta_x + delta_y:
                    delta_old = delta_x + delta_y
                    next_i = search_point_x
                    next_j = search_point_y

        # If the newly found point is no better than the old point we stop
        if next_i == i and next_j == j:
            break
        else:
            i = next_i
            j = next_j

    return next_i, next_j


def points_to_depth(keypoints, image_rgb, image_inf, camera, cache):
    if not cache.__contains__('depth_matching'):
        raise Exception('Depth matching not cached. '
                        'Run rgb_depth_calibration script.')
    
    map1x = cache['depth_matching'][camera]['map_rgb_x']
    map1y = cache['depth_matching'][camera]['map_rgb_y']
    map2x = cache['depth_matching'][camera]['map_infrared_x']
    map2y = cache['depth_matching'][camera]['map_infrared_y']
    
    image_rgb = cv2.remap(image_rgb, map1x, map1y, cv2.INTER_LINEAR)
    
    for i in range(keypoints.shape[0]):
        for j in range(keypoints.shape[1]):
            x, y = keypoints[i, j]
            keypoints[i, j] = _map(int(x), int(y), map1x, map1y)


    # Remove magic number .8
    image_inf = cv2.resize(
        image_inf,
        (data_loader.IMAGE_RGB_WIDTH, data_loader.IMAGE_RGB_HEIGHT))
    image_inf = cv2.remap(image_inf, map2x, map2y, cv2.INTER_LANCZOS4)

    # Add the dispartiy between RGB and INFRARED cameras
    image_inf = np.roll(image_inf, DISPARITY, axis=1)
    image_inf = np.clip(
            image_inf.astype(np.float32) * .1, 0, 255).astype('uint8')
    
    # keypoints = _get_skeleton(image_rgb, mmpose, visualize=False)
    
    for person_keypoints in keypoints:
        for idx, point in enumerate(person_keypoints):
            x = int(point[0])
            y = int(point[1])

            cv2.circle(
                image_rgb, (x, y), 10, (0, 0, 0),
                thickness=-1, lineType=8)
            cv2.circle(
                image_inf, (x, y), 10, (0, 0, 0),
                thickness=-1, lineType=8)

            cv2.putText(
                image_rgb, str(idx), (x - 5, y + 5),
                cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), thickness=2)
            cv2.putText(
                image_inf, str(idx), (x - 5, y + 5),
                cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), thickness=2)
    
    image_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2GRAY)

    f, axarr = plt.subplots(1, 2)
    axarr[0].imshow(image_rgb)
    axarr[1].imshow(image_inf)

    plt.show()


# Just for test
if __name__ == "__main__":
    cache = diskcache.Cache('cache')

    from mmpose.apis import MMPoseInferencer
    mmpose = MMPoseInferencer('human')

    camera = 'azure_kinect3_4_calib_snap'

    img_rgb_path = '/home/hamid/Documents/footefield/data/AzureKinectRecord_0729/a1/azure_kinect3_4/color/color00000.jpg'
    img_dpt_path = '/home/hamid/Documents/footefield/data/AzureKinectRecord_0729/a1/azure_kinect3_4/depth/depth00000.png'
    img_rgb = cv2.imread(img_rgb_path)
    img_dpt = cv2.imread(img_dpt_path, -1)
    img_dpt = cv2.resize(img_dpt, (1920, 1080))

    keypoints = _get_skeleton(img_rgb, mmpose)

    points_to_depth(keypoints, img_rgb, img_dpt, camera, cache)

    # img_dpt = np.clip(img_dpt.astype(np.float32) * 2, 0, 255).astype('uint8')
    # cv2.imshow("Depth", img_dpt)
    # cv2.waitKey(0)
