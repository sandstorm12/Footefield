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

idx0 = 2
idx1 = 3
cam0 = cams[idx0]
cam1 = cams[idx1]
img0 = cv2.imread(images[idx0])
img1 = cv2.imread(images[idx1])


img0 = data_loader.downsample_keep_aspect_ratio(
    img0,
    (data_loader.IMAGE_INFRARED_WIDTH, data_loader.IMAGE_INFRARED_HEIGHT))

img1 = data_loader.downsample_keep_aspect_ratio(
    img1,
    (data_loader.IMAGE_INFRARED_WIDTH, data_loader.IMAGE_INFRARED_HEIGHT))


# Detect keypoints
def detect_keypoints(image, inferencer):
    people = []

    result_generator = inferencer(image)
    for result in result_generator:
        for predictions in result['predictions'][0]:
            keypoints = predictions['keypoints']
            people.append(keypoints)

    return np.array(people)

inferencer = MMPoseInferencer('human')

img0_people = detect_keypoints(img0, inferencer)[0:2]
img1_people = detect_keypoints(img1, inferencer)[0:2]

print("2D:", img0_people)
print("2D:", img1_people)

print('img0_people', img0_people.shape, img0_people.dtype,
      'img1_people', img1_people.shape, img1_people.dtype,)


# Visualize 2d points
for person in img0_people:
    for idx, point in enumerate(person):
        x = int(point[0])
        y = int(point[1])

        cv2.circle(
            img0, (x, y), 10, (0, 0, 0),
            thickness=-1, lineType=8)

        cv2.putText(
            img0, '0_'+str(idx), (x - 5, y + 5),
            cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), thickness=2)
            
cv2.imshow('frame', img0)

for person in img1_people:
    for idx, point in enumerate(person):
        x = int(point[0])
        y = int(point[1])

        cv2.circle(
            img1, (x, y), 10, (0, 0, 0),
            thickness=-1, lineType=8)

        cv2.putText(
            img1, '1_'+str(idx), (x - 5, y + 5),
            cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), thickness=2)
            
cv2.imshow('frame1', img1)
cv2.waitKey(0)

# Loading camera parameters
cache = diskcache.Cache('../calibration/cache')

mtx0 = cache['extrinsics'][cam0]['mtx_l']
dist0 = cache['extrinsics'][cam0]['dist_l']
mtx1 = cache['extrinsics'][cam0]['mtx_r']
dist1 = cache['extrinsics'][cam0]['dist_r']
R = cache['extrinsics'][cam0]['rotation']
T = cache['extrinsics'][cam0]['transition']

R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
    mtx0, dist0, mtx1, dist1,
    (data_loader.IMAGE_INFRARED_WIDTH, data_loader.IMAGE_INFRARED_HEIGHT),
    R, T, flags=cv2.CALIB_FIX_INTRINSIC, alpha=-1)


# Undistort the image points
img_points0_undist = cv2.undistortPoints(img0_people.reshape(-1, 1, 2), mtx0, dist0, P=mtx0)
img_points1_undist = cv2.undistortPoints(img1_people.reshape(-1, 1, 2), mtx1, dist1, P=mtx1)

for idx, point in enumerate(img_points0_undist):
    x = int(point[0][0])
    y = int(point[0][1])

    cv2.circle(
        img0, (x, y), 10, (0, 0, 0),
        thickness=-1, lineType=8)

    cv2.putText(
        img0, '2_'+str(idx), (x - 5, y + 5),
        cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), thickness=2)
            
cv2.imshow('frame', img0)

for idx, point in enumerate(img_points1_undist):
    x = int(point[0][0])
    y = int(point[0][1])

    cv2.circle(
        img1, (x, y), 10, (0, 0, 0),
        thickness=-1, lineType=8)

    cv2.putText(
        img1, '2_'+str(idx), (x - 5, y + 5),
        cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), thickness=2)
            
cv2.imshow('frame1', img1)
cv2.waitKey(0)

# Triangulate the 3D point
point_3d_01 = cv2.triangulatePoints(P1, P2, img_points0_undist, img_points1_undist)

print("3D:", point_3d_01.shape)

# Convert from homogeneous to Euclidean coordinates
point_3d_01 = cv2.convertPointsFromHomogeneous(point_3d_01.T)

point_3d_01 = point_3d_01.reshape((-1, 17, 3))

print("3D:", point_3d_01.shape)

# Print the result
print('The 3D coordinates of the point 01 are:', point_3d_01.shape)

# Visualize triangulated points
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
title = ax.set_title('3D Test')

for person in point_3d_01:
    # Define the data for the scatter plot
    x = [point[0] for point in person]
    y = [point[2] for point in person]
    z = [-1 * point[1] for point in person]
    x.append(0)
    y.append(0)
    z.append(0)

    center = (int(np.mean(x)), int(np.mean(y)), int(np.mean(z)))

    graph = ax.scatter(x, y, z, c='r', marker='o')
    for idx in range(len(data_loader.MMPOSE_EDGES)):
        ax.plot(
            (x[data_loader.MMPOSE_EDGES[idx][0]],
                x[data_loader.MMPOSE_EDGES[idx][1]]),
            (y[data_loader.MMPOSE_EDGES[idx][0]],
                y[data_loader.MMPOSE_EDGES[idx][1]]),
            (z[data_loader.MMPOSE_EDGES[idx][0]],
                z[data_loader.MMPOSE_EDGES[idx][1]])
        )[0]

# ax.view_init(elev=1, azim=-89)

# Remove the grid background
ax.grid(False)

# Set the labels for the axes
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

# ax.axes.set_xlim3d(center[0] - 500, center[0] + 500)
# ax.axes.set_ylim3d(center[1] - 500, center[1] + 500)
# ax.axes.set_zlim3d(center[2] - 500, center[2] + 500)

plt.show()
