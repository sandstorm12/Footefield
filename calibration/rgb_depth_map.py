import sys
sys.path.append('../')

import cv2
import diskcache
import numpy as np
import matplotlib.pyplot as plt

from utils import data_loader


DISPARITY = 9
DEPTH_AREA_MIN = 10
DEPTH_AREA_MAX = 30
DEPTH_AREA_STEP = 2
MAX_DIST = 5000
MIN_DIST = 100


def invert_map(map2x: np.ndarray, map2y: np.ndarray):
    F = np.stack((map2x, map2y), axis=2)
    # shape is (h, w, 2), an "xymap"
    (h, w) = F.shape[:2]
    I = np.zeros_like(F)
    I[:,:,1], I[:,:,0] = np.indices((h, w)) # identity map
    P = np.copy(I)
    for i in range(10):
        correction = I - cv2.remap(F, P, None, interpolation=cv2.INTER_LINEAR)
        P += correction * 0.5
    
    return P[:,:,0], P[:,:,1]


def align_image_rgb_to_unified(image, camera, cache):
    if not cache.__contains__('depth_matching'):
        raise Exception('Depth matching not cached. '
                        'Run rgb_depth_calibration script.')
    
    map1x = cache['depth_matching'][camera]['map_rgb_x']
    map1y = cache['depth_matching'][camera]['map_rgb_y']

    image = cv2.remap(image, map1x, map1y, cv2.INTER_CUBIC)

    image = np.roll(image, DISPARITY, axis=1)

    return image


def align_image_unified_to_depth(image, camera, cache):
    if not cache.__contains__('depth_matching'):
        raise Exception('Depth matching not cached. '
                        'Run rgb_depth_calibration script.')

    map2x = cache['depth_matching'][camera]['map_infrared_x']
    map2y = cache['depth_matching'][camera]['map_infrared_y']
    map2x, map2y = invert_map(map2x, map2y)

    image = cv2.remap(image, map2x, map2y, cv2.INTER_CUBIC)

    image = np.roll(image, DISPARITY, axis=1)

    return image


def align_image_rgb_to_depth(image, camera, cache):
    if not cache.__contains__('depth_matching'):
        raise Exception('Depth matching not cached. '
                        'Run rgb_depth_calibration script.')

    map1x = cache['depth_matching'][camera]['map_rgb_x']
    map1y = cache['depth_matching'][camera]['map_rgb_y']
    map2x = cache['depth_matching'][camera]['map_infrared_x']
    map2y = cache['depth_matching'][camera]['map_infrared_y']
    map2x, map2y = invert_map(map2x, map2y)

    image = cv2.remap(image, map1x, map1y, cv2.INTER_CUBIC)
    image = cv2.remap(image, map2x, map2y, cv2.INTER_CUBIC)

    image = np.roll(image, DISPARITY, axis=1)

    return image


def align_image_depth_to_unified(image, camera, cache):
    if not cache.__contains__('depth_matching'):
        raise Exception('Depth matching not cached. '
                        'Run rgb_depth_calibration script.')

    map2x = cache['depth_matching'][camera]['map_infrared_x']
    map2y = cache['depth_matching'][camera]['map_infrared_y']

    image = cv2.remap(image, map2x, map2y, cv2.INTER_NEAREST)

    image = np.roll(image, -1 * DISPARITY, axis=1)

    return image


def align_image_unified_to_rgb(image, camera, cache):
    if not cache.__contains__('depth_matching'):
        raise Exception('Depth matching not cached. '
                        'Run rgb_depth_calibration script.')
    
    map1x = cache['depth_matching'][camera]['map_rgb_x']
    map1y = cache['depth_matching'][camera]['map_rgb_y']
    map1x, map1y = invert_map(map1x, map1y)

    image = cv2.remap(image, map1x, map1y, cv2.INTER_NEAREST)

    image = np.roll(image, -1 * DISPARITY, axis=1)

    return image


def align_image_depth_to_rgb(image, camera, cache):
    map1x = cache['depth_matching'][camera]['map_rgb_x']
    map1y = cache['depth_matching'][camera]['map_rgb_y']
    map1x, map1y = invert_map(map1x, map1y)
    map2x = cache['depth_matching'][camera]['map_infrared_x']
    map2y = cache['depth_matching'][camera]['map_infrared_y']

    image = cv2.remap(image, map2x, map2y, cv2.INTER_NEAREST)
    image = cv2.remap(image, map1x, map1y, cv2.INTER_NEAREST)

    image = np.roll(image, -1 * DISPARITY, axis=1)

    return image


def points_to_depth(people_keypoints, image_depth):
    keypoints_3d = []
    for keypoints in people_keypoints:
        keypoints_3d.append([])
        for i in range(len(keypoints)):
            x, y = keypoints[i]
            x = int(x)
            y = int(y)

            for area in range(DEPTH_AREA_MIN,
                              DEPTH_AREA_MAX,
                              DEPTH_AREA_STEP):
                roi = image_depth[y-area:y+area,
                                x-area:x+area]
                roi = roi[np.logical_and(roi > MIN_DIST, roi < MAX_DIST)]
                
                if len(roi) > 0:
                    # roi = reject_outliers(roi)
                    depth = np.median(roi)
                    break
            else:
                # print('No depth found')
                depth = 0

            # print('Selected:', depth)
            keypoints_3d[-1].append((x, y, depth))

    return keypoints_3d


def reject_outliers(data, quantile_lower=.4, quantile_upper=.6):
    lower = np.quantile(data, quantile_lower)
    upper = np.quantile(data, quantile_upper)
    filtered = [x for x in data if lower <= x <= upper]

    return filtered

# Just for test
# Clean and shorten the test
if __name__ == "__main__":
    cache = diskcache.Cache('cache')

    from mmpose.apis import MMPoseInferencer
    mmpose = MMPoseInferencer('human')

    def _get_skeleton(image, inferencer):
        result_generator = inferencer(image)
        
        detected_keypoints = []
        for result in result_generator:
            for predictions in result['predictions'][0]:
                keypoints = predictions['keypoints']
                detected_keypoints.append(keypoints)

        return np.array(detected_keypoints)

    camera = 'azure_kinect3_4_calib_snap'

    img_rgb_path = '/home/hamid/Documents/phd/footefield/data/AzureKinectRecord_0729/a1/azure_kinect3_4/color/color00000.jpg'
    img_dpt_path = '/home/hamid/Documents/phd/footefield/data/AzureKinectRecord_0729/a1/azure_kinect3_4/depth/depth00000.png'
    img_rgb = cv2.imread(img_rgb_path)
    img_rgb = data_loader.downsample_keep_aspect_ratio(
        img_rgb,
        (
            data_loader.IMAGE_INFRARED_WIDTH,
            data_loader.IMAGE_INFRARED_HEIGHT
        )
    )
    img_dpt = cv2.imread(img_dpt_path, -1)

    img_rgb = align_image_rgb_to_unified(img_rgb, camera, cache)
    img_depth = align_image_depth_to_unified(img_dpt, camera, cache)

    people_keypoints = _get_skeleton(img_rgb, mmpose)

    people_keypoints_3d = points_to_depth(
        people_keypoints, img_dpt)

    for keypoints_3d in people_keypoints_3d:
        for idx, point in enumerate(keypoints_3d):
            x = int(point[0])
            y = int(point[1])

            cv2.circle(
                img_dpt, (x, y), 10, (MAX_DIST, MAX_DIST, MAX_DIST),
                thickness=-1, lineType=8)

            cv2.putText(
                img_dpt, str(idx), (x - 5, y + 5),
                cv2.FONT_HERSHEY_SIMPLEX, .5,
                (MAX_DIST, MAX_DIST, MAX_DIST), thickness=2)
        
    plt.imshow(img_dpt)
    plt.show()

    # Create a figure and a 3D axis
    fig = plt.figure(figsize=(20, 20))
    ax = fig.add_subplot(111, projection='3d')

    # Create the scatter plot
    for keypoints_3d in people_keypoints_3d:
        # Define the data for the scatter plot
        x = [point[0] for point in keypoints_3d]
        y = [point[2] for point in keypoints_3d]
        z = [576 - point[1] for point in keypoints_3d]

        scatter = ax.scatter(x, y, z, c='r', marker='o')
    ax.view_init(elev=1, azim=-89)

    # Remove the grid background
    ax.grid(False)

    # Add the index of each point as a text on top of each point
    for i, txt in enumerate(range(len(x))):
        ax.text(x[i], y[i], z[i], str(txt), color='black')

    # Set the labels for the axes
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    ax.axes.set_xlim3d(0, 640)
    ax.axes.set_zlim3d(0, 576)
    ax.axes.set_ylim3d(0, 3000)

    # Show the plot
    plt.show()
