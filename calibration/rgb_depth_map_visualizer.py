import sys
sys.path.append('../')

import cv2
import diskcache
import numpy as np
import matplotlib.pyplot as plt

from utils import data_loader


DISPARITY = -9
DEPTH_AREA = 10
MAX_DIST = 5000
MIN_DIST = 100
MAX_STD = 30


def align_image_rgb(image, camera, cache):
    if not cache.__contains__('depth_matching'):
        raise Exception('Depth matching not cached. '
                        'Run rgb_depth_calibration script.')
    
    map1x = cache['depth_matching'][camera]['map_rgb_x']
    map1y = cache['depth_matching'][camera]['map_rgb_y']

    image_rgb = cv2.remap(image, map1x, map1y, cv2.INTER_NEAREST)

    return image_rgb


def align_image_depth(image, camera, cache):
    if not cache.__contains__('depth_matching'):
        raise Exception('Depth matching not cached. '
                        'Run rgb_depth_calibration script.')
    
    map2x = cache['depth_matching'][camera]['map_infrared_x']
    map2y = cache['depth_matching'][camera]['map_infrared_y']

    image_depth = cv2.remap(image, map2x, map2y, cv2.INTER_NEAREST)
    image_depth = np.roll(image_depth, DISPARITY, axis=1)

    return image_depth


def reject_outliers(data, quantile_lower=.4, quantile_upper=.6):
    lower = np.quantile(data, quantile_lower)
    upper = np.quantile(data, quantile_upper)
    filtered = [x for x in data if lower <= x <= upper]

    return filtered


# Just for test
# Clean and shorten the test
if __name__ == "__main__":
    cache = diskcache.Cache('cache')
    
    camera = 'azure_kinect3_4_calib_snap'
    img_rgb_path = '/home/hamid/Documents/footefield/data/AzureKinectRecord_0729/a1/azure_kinect3_4/color/color00000.jpg'
    img_dpt_path = '/home/hamid/Documents/footefield/data/AzureKinectRecord_0729/a1/azure_kinect3_4/depth/depth00000.png'

    img_rgb = cv2.imread(img_rgb_path)
    img_rgb = data_loader.downsample_keep_aspect_ratio(
        img_rgb,
        (data_loader.IMAGE_INFRARED_WIDTH,
         data_loader.IMAGE_INFRARED_HEIGHT))
    img_dpt = cv2.imread(img_dpt_path, -1)

    img_rgb = align_image_rgb(img_rgb, camera, cache)
    img_dpt = align_image_depth(img_dpt, camera, cache)

    f, axarr = plt.subplots(1,2)
    implot = axarr[0].imshow(img_rgb)
    implot = axarr[1].imshow(img_dpt)

    def onclick(event):
        if event.xdata != None and event.ydata != None:
            x = int(event.xdata)
            y = int(event.ydata)
            
            circle = plt.Circle((x, y), 5, color="red", fill=True)
            axarr[0].add_patch(circle)

            roi = img_dpt[y-DEPTH_AREA:y+DEPTH_AREA,
                            x-DEPTH_AREA:x+DEPTH_AREA]
            roi = roi[np.logical_and(roi > MIN_DIST, roi < MAX_DIST)]

            if len(roi) > 0:
                roi = reject_outliers(roi)

            if len(roi) > 0 and np.std(roi) < MAX_STD:
                depth = np.median(roi)
                axarr[1].annotate(str(depth), (x, y), color='w', weight='bold', fontsize=6, ha='center', va='center')
            else:
                circle = plt.Circle((x, y), 5, color=(1, 1, 1, 1), fill=True)
                axarr[1].add_patch(circle)
            plt.draw()

    cid = f.canvas.mpl_connect('button_press_event', onclick)

    plt.show()
