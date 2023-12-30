import sys
sys.path.append('../')

import os
import cv2
import diskcache
import numpy as np
import matplotlib.pyplot as plt

from utils import data_loader
from calibration import rgb_depth_map
from tqdm import tqdm

from mmpose.apis import MMPoseInferencer
import matplotlib.animation


DISPARITY = -18
DEPTH_AREA = 10

DIRS = [
    [
        "/home/hamid/Documents/footefield/data/AzureKinectRecord_0729/a1/azure_kinect1_4",
        "/home/hamid/Documents/footefield/data/AzureKinectRecord_0729/a1/azure_kinect1_5",
        "/home/hamid/Documents/footefield/data/AzureKinectRecord_0729/a1/azure_kinect2_4",
        "/home/hamid/Documents/footefield/data/AzureKinectRecord_0729/a1/azure_kinect3_4",
        "/home/hamid/Documents/footefield/data/AzureKinectRecord_0729/a1/azure_kinect3_5",
    ],[
        "/home/hamid/Documents/footefield/data/AzureKinectRecord_0729/a2/azure_kinect1_4",
        "/home/hamid/Documents/footefield/data/AzureKinectRecord_0729/a2/azure_kinect1_5",
        "/home/hamid/Documents/footefield/data/AzureKinectRecord_0729/a2/azure_kinect2_4",
        "/home/hamid/Documents/footefield/data/AzureKinectRecord_0729/a2/azure_kinect3_4",
        "/home/hamid/Documents/footefield/data/AzureKinectRecord_0729/a2/azure_kinect3_5",
    ]
]

MMPOSE_EDGES = [
    (0, 1),
    (0, 2),
    (1, 2),
    (1, 3),
    (2, 4),
    (3, 5),
    (4, 6),
    (5, 6),
    (5, 7),
    (5, 11),
    (6, 8),
    (6, 12),
    (7, 9),
    (8, 10),
    (11, 12),
    (11, 13),
    (12, 14),
    (13, 15),
    (14, 16),
]


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


def points_to_depth(people_keypoints, image_depth, camera, cache):
    keypoints_3d = []
    for keypoints in people_keypoints:
        keypoints_3d.append([])
        for i in range(len(keypoints)):
            x, y = keypoints[i]
            x = int(x)
            y = int(y)
            roi = image_depth[y-DEPTH_AREA:y+DEPTH_AREA,
                            x-DEPTH_AREA:x+DEPTH_AREA]
            roi = roi[np.logical_and(roi < 3000, roi > 100)]
            if len(roi) > 0:
                depth = np.max(roi)
                keypoints_3d[-1].append((x, y, depth))
            else:
                keypoints_3d[-1].append((x, y, 0))

    return keypoints_3d


def get_image_paths(dir):
    img_rgb_paths = []
    img_dpt_paths = []

    img_rgb_paths = data_loader.list_rgb_images(os.path.join(dir, "color"))
    img_dpt_paths = data_loader.list_depth_images(os.path.join(dir, "depth"))

    return img_rgb_paths, img_dpt_paths


# Just for test
# Clean and shorten the test
if __name__ == "__main__":
    cache = diskcache.Cache('../calibration/cache')

    cache_process = cache.get('process', {})

    mmpose = MMPoseInferencer('human')

    for case in DIRS[1:]:
        for dir in case:
            camera = dir.split("/")[-1] + "_calib_snap"
            img_rgb_paths, img_dpt_paths = get_image_paths(dir)

            people_keypoints_3d_time = []
            for img_rgb_path, img_dpt_path in tqdm(zip(img_rgb_paths, img_dpt_paths)):
                print(img_rgb_path)
                print(img_dpt_path)

                img_rgb = cv2.imread(img_rgb_path)
                img_dpt = cv2.imread(img_dpt_path, -1)
                img_dpt = cv2.resize(img_dpt, (1920, 1080))

                img_rgb = rgb_depth_map.align_image_rgb(img_rgb, camera, cache)
                img_dpt = rgb_depth_map.align_image_depth(img_dpt, camera, cache)

                people_keypoints = _get_skeleton(img_rgb, mmpose)

                people_keypoints_3d = points_to_depth(people_keypoints, img_dpt, camera, cache)[:2]
                people_keypoints_3d_time.append(people_keypoints_3d)

                if len(people_keypoints_3d_time) > 5:
                    break
            if len(people_keypoints_3d_time) > 5:
                    break

                # f, axarr = plt.subplots(1,2)
                # implot = axarr[0].imshow(img_rgb)
                # implot = axarr[1].imshow(img_dpt)

                # for keypoints_3d in people_keypoints_3d:
                #     for idx, point in enumerate(keypoints_3d):
                #         x = int(point[0])
                #         y = int(point[1])

                #         cv2.circle(
                #             img_rgb, (x, y), 10, (100, 100, 100),
                #             thickness=-1, lineType=8)
                #         cv2.circle(
                #             img_dpt, (x, y), 10, (3000, 3000, 3000),
                #             thickness=-1, lineType=8)

                #         cv2.putText(
                #             img_dpt, str(idx), (x - 5, y + 5),
                #             cv2.FONT_HERSHEY_SIMPLEX, .5,
                #             (3000, 3000, 3000), thickness=2)
                #         cv2.putText(
                #             img_rgb, str(idx), (x - 5, y + 5),
                #             cv2.FONT_HERSHEY_SIMPLEX, .5,
                #             (0, 0, 0), thickness=2)
                    
                # plt.imshow(img_dpt)
                # plt.show()


        # # Create a figure and a 3D axis
        # fig = plt.figure(figsize=(20, 20))
        # ax = fig.add_subplot(111, projection='3d')

        # # Create the scatter plot
        # for keypoints_3d in people_keypoints_3d:
        #     # Define the data for the scatter plot
        #     x = [point[0] for point in keypoints_3d]
        #     y = [point[2] for point in keypoints_3d]
        #     z = [1080 - point[1] for point in keypoints_3d]

        #     scatter = ax.scatter(x, y, z, c='r', marker='o')
        # ax.view_init(elev=1, azim=-89)

        # # Remove the grid background
        # ax.grid(False)

        # # Add the index of each point as a text on top of each point
        # for i, txt in enumerate(range(len(x))):
        #     ax.text(x[i], y[i], z[i], str(txt), color='black')

        # # Set the labels for the axes
        # ax.set_xlabel('X Label')
        # ax.set_ylabel('Y Label')
        # ax.set_zlabel('Z Label')

        # ax.axes.set_xlim3d(0, 1920)
        # ax.axes.set_zlim3d(0, 1080)
        # ax.axes.set_ylim3d(0, 3000)

        # # Show the plot
        # plt.show()

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        title = ax.set_title('3D Test')

        # Create the scatter plot
        keypoints = np.array(people_keypoints_3d_time[0])
        print('Keypoints', keypoints.shape)
        keypoints_3d = keypoints.reshape(-1, 3)
        
        # Define the data for the scatter plot
        x = [point[0] for point in keypoints_3d if point[2] != 0]
        y = [point[2] for point in keypoints_3d if point[2] != 0]
        z = [1080 - point[1] for point in keypoints_3d if point[2] != 0]

        graph = ax.scatter(x, y, z, c='r', marker='o')
        lines = []
        for idx in range(len(MMPOSE_EDGES)):
            lines.append(
                ax.plot(
                    (x[MMPOSE_EDGES[idx][0]], x[MMPOSE_EDGES[idx][1]]),
                    (y[MMPOSE_EDGES[idx][0]], y[MMPOSE_EDGES[idx][1]]),
                    (z[MMPOSE_EDGES[idx][0]], z[MMPOSE_EDGES[idx][1]])
                )[0]
            )

        ax.view_init(elev=1, azim=-89)

        # Remove the grid background
        ax.grid(False)

        # # Add the index of each point as a text on top of each point
        # for i, txt in enumerate(range(len(x))):
        #     ax.text(x[i], y[i], z[i], str(txt), color='black')

        # Set the labels for the axes
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')

        ax.axes.set_xlim3d(0, 1920)
        ax.axes.set_zlim3d(0, 1080)
        ax.axes.set_ylim3d(0, 3000)

        def update_graph(num):            
            keypoints = np.array(people_keypoints_3d_time[num])
            keypoints_3d = keypoints.reshape(-1, 3)

            # Define the data for the scatter plot
            x = [point[0] for point in keypoints_3d if point[2] != 0]
            y = [point[2] for point in keypoints_3d if point[2] != 0]
            z = [1080 - point[1] for point in keypoints_3d if point[2] != 0]

            graph._offsets3d = (x, y, z)
            
            title.set_text('3D Test, time={}'.format(num))

            for idx, line in enumerate(lines):
                line.set_data((x[MMPOSE_EDGES[idx][0]], x[MMPOSE_EDGES[idx][1]]),
                            (y[MMPOSE_EDGES[idx][0]], y[MMPOSE_EDGES[idx][1]]))
                line.set_3d_properties(
                    (z[MMPOSE_EDGES[idx][0]], z[MMPOSE_EDGES[idx][1]])
                )

        ani = matplotlib.animation.FuncAnimation(
            fig, update_graph, len(people_keypoints_3d_time),
            interval=1000, blit=False)

        plt.show()
