import sys
sys.path.append('../')

import cv2
import diskcache
import numpy as np
import matplotlib.pyplot as plt


DISPARITY = -18
DEPTH_AREA = 10


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


def align_image_rgb(image, camera, cache):
    if not cache.__contains__('depth_matching'):
        raise Exception('Depth matching not cached. '
                        'Run rgb_depth_calibration script.')
    
    map1x = cache['depth_matching'][camera]['map_rgb_x']
    map1y = cache['depth_matching'][camera]['map_rgb_y']

    image_rgb = cv2.remap(image, map1x, map1y, cv2.INTER_LANCZOS4)

    return image_rgb


def align_image_depth(image, camera, cache):
    if not cache.__contains__('depth_matching'):
        raise Exception('Depth matching not cached. '
                        'Run rgb_depth_calibration script.')
    
    map2x = cache['depth_matching'][camera]['map_infrared_x']
    map2y = cache['depth_matching'][camera]['map_infrared_y']

    image_depth = cv2.remap(image, map2x, map2y, cv2.INTER_LANCZOS4)
    image_depth = np.roll(image_depth, DISPARITY, axis=1)

    return image_depth


def points_to_depth(people_keypoints, image_depth, camera, cache):
    image_depth = align_image_depth(image_depth, camera, cache)

    map1x = cache['depth_matching'][camera]['map_rgb_x']
    map1y = cache['depth_matching'][camera]['map_rgb_y']
    
    keypoints_3d = []
    for keypoints in people_keypoints:
        keypoints_3d.append([])
        for i in range(len(keypoints)):
            x, y = keypoints[i]
            x = int(x)
            y = int(y)
            x_new, y_new = _map(x, y, map1x, map1y)
            roi = image_depth[y_new-DEPTH_AREA:y_new+DEPTH_AREA,
                            x_new-DEPTH_AREA:x_new+DEPTH_AREA]
            roi = roi[roi != 0]
            if len(roi) > 0:
                depth = np.max(roi)
                keypoints_3d[-1].append((x_new, y_new, depth))

    return keypoints_3d


# Just for test
# Clean and shorten the test
if __name__ == "__main__":
    cache = diskcache.Cache('cache')
    
    camera = 'azure_kinect3_4_calib_snap'
    img_rgb_path = '/home/hamid/Documents/footefield/data/AzureKinectRecord_0729/a1/azure_kinect3_4/color/color00000.jpg'
    img_dpt_path = '/home/hamid/Documents/footefield/data/AzureKinectRecord_0729/a1/azure_kinect3_4/depth/depth00000.png'

    img_rgb = cv2.imread(img_rgb_path)
    img_dpt = cv2.imread(img_dpt_path, -1)
    img_dpt = cv2.resize(img_dpt, (1920, 1080))

    img_rgb = align_image_rgb(img_rgb, camera, cache)
    img_dpt = align_image_depth(img_dpt, camera, cache)

    f, axarr = plt.subplots(1,2)
    implot = axarr[0].imshow(img_rgb)
    implot = axarr[1].imshow(img_dpt)

    def onclick(event):
        if event.xdata != None and event.ydata != None:
            print(event.xdata, event.ydata)
            x = int(event.xdata)
            y = int(event.ydata)
            
            circle = plt.Circle((x, y), 10, color="red", fill=True)
            axarr[0].add_patch(circle)

            roi = img_dpt[y-DEPTH_AREA:y+DEPTH_AREA,
                            x-DEPTH_AREA:x+DEPTH_AREA]
            roi = roi[roi != 0]
            if len(roi) > 0:
                depth = np.max(roi)
                axarr[1].annotate(str(depth), (x, y), color='w', weight='bold', fontsize=6, ha='center', va='center')
            else:
                circle = plt.Circle((x, y), 10, color=(1, 1, 1, 1), fill=True)
                axarr[1].add_patch(circle)
            plt.draw()

    cid = f.canvas.mpl_connect('button_press_event', onclick)

    plt.show()
