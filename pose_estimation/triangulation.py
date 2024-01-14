import sys
sys.path.append('../')

import cv2
import diskcache
import numpy as np
import matplotlib.pyplot as plt

from utils import data_loader
from mmpose.apis import MMPoseInferencer


cams = [
    "azure_kinect1_5_calib_snap",
    "azure_kinect1_4_calib_snap",
    "azure_kinect3_4_calib_snap",
    "azure_kinect3_5_calib_snap",
    "azure_kinect2_4_calib_snap",
]

images = [
    '/home/hamid/Documents/footefield/data/AzureKinectRecord_0729/a1/azure_kinect1_5/color/color00001.jpg',
    '/home/hamid/Documents/footefield/data/AzureKinectRecord_0729/a1/azure_kinect1_4/color/color00001.jpg',
    '/home/hamid/Documents/footefield/data/AzureKinectRecord_0729/a1/azure_kinect3_4/color/color00001.jpg',
    '/home/hamid/Documents/footefield/data/AzureKinectRecord_0729/a1/azure_kinect3_5/color/color00001.jpg',
    '/home/hamid/Documents/footefield/data/AzureKinectRecord_0729/a1/azure_kinect2_4/color/color00001.jpg',
]

do_reverse = [
    None,
    None,
    False,
    False,
    False,
]


def load_pairing_images(idx0, idx1):
    img0 = cv2.imread(images[idx0])
    img1 = cv2.imread(images[idx1])

    img0 = data_loader.downsample_keep_aspect_ratio(
        img0,
        (data_loader.IMAGE_INFRARED_WIDTH, data_loader.IMAGE_INFRARED_HEIGHT))

    img1 = data_loader.downsample_keep_aspect_ratio(
        img1,
        (data_loader.IMAGE_INFRARED_WIDTH, data_loader.IMAGE_INFRARED_HEIGHT))
    
    return img0, img1


def extract_skeletons(img0, img1):
    inferencer = MMPoseInferencer('human')

    img0_people = detect_keypoints(img0, inferencer)[0:2]
    img1_people = detect_keypoints(img1, inferencer)[0:2]

    if do_reverse[idx0]:
        img0_people = np.flip(img0_people, 0)

    if do_reverse[idx1]:
        img1_people = np.flip(img1_people, 0)

    print('img0_people', img0_people.shape, img0_people.dtype,
        'img1_people', img1_people.shape, img1_people.dtype,)
    
    # Visualize 2d points
    for idx_person, person in enumerate(img0_people):
        for idx, point in enumerate(person):
            x = int(point[0])
            y = int(point[1])

            cv2.circle(
                img0, (x, y), 10, (0, 0, 0),
                thickness=-1, lineType=8)

            cv2.putText(
                img0, f'{idx_person}_'+str(idx), (x - 5, y + 5),
                cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), thickness=2)
                
    cv2.imshow('frame', img0)

    for idx_person, person in enumerate(img1_people):
        for idx, point in enumerate(person):
            x = int(point[0])
            y = int(point[1])

            cv2.circle(
                img1, (x, y), 10, (0, 0, 0),
                thickness=-1, lineType=8)

            cv2.putText(
                img1, f'{idx_person}_'+str(idx), (x - 5, y + 5),
                cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), thickness=2)
                
    cv2.imshow('frame1', img1)
    cv2.waitKey(0)

    return img0_people, img1_people


def detect_keypoints(image, inferencer):
    people = []

    result_generator = inferencer(image)
    for result in result_generator:
        for predictions in result['predictions'][0]:
            keypoints = predictions['keypoints']
            people.append(keypoints)

    return np.array(people)


def visualize_undistorted_points(img_points0_undist, img_points1_undist):
    for idx, point in enumerate(img_points0_undist):
        x = int(point[0][0])
        y = int(point[0][1])

        idx_person = idx // 17

        cv2.circle(
            img0, (x, y), 10, (0, 0, 0),
            thickness=-1, lineType=8)

        cv2.putText(
            img0, f'{idx_person}_'+str(idx), (x - 5, y + 5),
            cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), thickness=2)
                
    cv2.imshow('frame', img0)

    for idx, point in enumerate(img_points1_undist):
        x = int(point[0][0])
        y = int(point[0][1])

        idx_person = idx // 17

        cv2.circle(
            img1, (x, y), 10, (0, 0, 0),
            thickness=-1, lineType=8)

        cv2.putText(
            img1, f'{idx_person}_'+str(idx), (x - 5, y + 5),
            cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), thickness=2)
                
    cv2.imshow('frame1', img1)
    cv2.waitKey(0)


def triangulate_points(idx0, img0, img1, cache):
    cam0 = cams[idx0]

    mtx0 = cache['extrinsics'][cam0]['mtx_l']
    dist0 = cache['extrinsics'][cam0]['dist_l']
    mtx1 = cache['extrinsics'][cam0]['mtx_r']
    dist1 = cache['extrinsics'][cam0]['dist_r']
    R = cache['extrinsics'][cam0]['rotation']
    T = cache['extrinsics'][cam0]['transition']

    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        mtx0, dist0, mtx1, dist1,
        (data_loader.IMAGE_INFRARED_WIDTH, data_loader.IMAGE_INFRARED_HEIGHT),
        R, T, flags=0, alpha=-1)

    map0x, map0y = cv2.initUndistortRectifyMap(
        mtx0, dist0, R1, P1,
        (data_loader.IMAGE_INFRARED_WIDTH, data_loader.IMAGE_INFRARED_HEIGHT),
        cv2.CV_32FC1)
    map1x, map1y = cv2.initUndistortRectifyMap(
        mtx1, dist1, R2, P2,
        (data_loader.IMAGE_INFRARED_WIDTH, data_loader.IMAGE_INFRARED_HEIGHT),
        cv2.CV_32FC1)

    img0_mapped = cv2.remap(img0, map0x, map0y, cv2.INTER_NEAREST)
    img1_mapped = cv2.remap(img1, map1x, map1y, cv2.INTER_NEAREST)

    cv2.imshow('frame', img0_mapped)
    cv2.imshow('frame1', img1_mapped)
    cv2.waitKey(0)

    # Undistort the image points
    img_points0_undist = cv2.undistortPoints(img0_people.reshape(-1, 1, 2), mtx0, dist0, P=mtx0)
    img_points1_undist = cv2.undistortPoints(img1_people.reshape(-1, 1, 2), mtx1, dist1, P=mtx1)

    visualize_undistorted_points(img_points0_undist, img_points1_undist)

    # Triangulate the 3D point
    points_3d = cv2.triangulatePoints(P1, P2, img_points0_undist, img_points1_undist)

    print("3D:", points_3d.shape)

    # Convert from homogeneous to Euclidean coordinates
    points_3d = cv2.convertPointsFromHomogeneous(points_3d.T)

    points_3d = points_3d.reshape((-1, 17, 3))

    return points_3d


def visualizer_3d_points(point_3d_01):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title('3D Test')

    for person in point_3d_01:
        # Define the data for the scatter plot
        x = [point[0] for point in person]
        y = [point[2] for point in person]
        z = [-1 * point[1] for point in person]
        x.append(0)
        y.append(0)
        z.append(0)

        ax.scatter(x, y, z, c='r', marker='o')
        for idx in range(len(data_loader.MMPOSE_EDGES)):
            ax.plot(
                (x[data_loader.MMPOSE_EDGES[idx][0]],
                    x[data_loader.MMPOSE_EDGES[idx][1]]),
                (y[data_loader.MMPOSE_EDGES[idx][0]],
                    y[data_loader.MMPOSE_EDGES[idx][1]]),
                (z[data_loader.MMPOSE_EDGES[idx][0]],
                    z[data_loader.MMPOSE_EDGES[idx][1]])
            )[0]

    ax.grid(False)

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    plt.show()


if __name__ == "__main__":
    # Good pairs
    # 3_4 & 3_5
    # 2_4 & 1_5
    idx0, idx1 = 4, 0

    cache = diskcache.Cache('../calibration/cache')

    img0, img1 = load_pairing_images(idx0, idx1)
    img0_people, img1_people = extract_skeletons(img0, img1)
    points_3d = triangulate_points(idx0, img0, img1, cache)
    visualizer_3d_points(points_3d)
