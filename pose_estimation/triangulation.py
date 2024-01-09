import sys
sys.path.append('../')

import cv2
import diskcache
import numpy as np
import matplotlib.pyplot as plt

from utils import data_loader
from mmpose.apis import MMPoseInferencer


# Good pairs
# 3_4 3_5
# 1_5 1_4
# Load the images from two cameras
cam0 = "azure_kinect1_5_calib_snap"
cam1 = "azure_kinect1_4_calib_snap"
cam2 = "azure_kinect3_4_calib_snap"
cam3 = "azure_kinect3_5_calib_snap"
img0 = cv2.imread('/home/hamid/Documents/footefield/data/AzureKinectRecord_0729/a1/azure_kinect1_5/color/color00001.jpg')
img1 = cv2.imread('/home/hamid/Documents/footefield/data/AzureKinectRecord_0729/a1/azure_kinect1_4/color/color00001.jpg')
img2 = cv2.imread('/home/hamid/Documents/footefield/data/AzureKinectRecord_0729/a1/azure_kinect3_4/color/color00001.jpg')
img3 = cv2.imread('/home/hamid/Documents/footefield/data/AzureKinectRecord_0729/a1/azure_kinect3_5/color/color00001.jpg')


# Detect keypoints
def detect_keypoints(image, inferencer):
    people = []

    result_generator = inferencer(image)
    for result in result_generator:
        for predictions in result['predictions'][0]:
            keypoints = predictions['keypoints']
            people.append(keypoints)
            # for idx, point in enumerate(keypoints):

            #     x = int(point[0])
            #     y = int(point[1])

            #     cv2.circle(
            #         image, (x, y), 10, (0, 0, 0),
            #         thickness=-1, lineType=8)

            #     cv2.putText(
            #         image, str(idx), (x - 5, y + 5),
            #         cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), thickness=2)
                
        # cv2.imshow('frame', image)
        # cv2.waitKey(0)

    return np.array(people)

inferencer = MMPoseInferencer('human')

img0_people = detect_keypoints(img0, inferencer)[1]
img1_people = detect_keypoints(img1, inferencer)[1]
img2_people = detect_keypoints(img2, inferencer)[1]
img3_people = detect_keypoints(img3, inferencer)[1]

print('img0_people', img0_people.shape, img0_people.dtype,
      'img1_people', img1_people.shape, img1_people.dtype,
      'img2_people', img2_people.shape, img2_people.dtype,
      'img3_people', img3_people.shape, img3_people.dtype,)


# Visualize 2d points
for idx, point in enumerate(img0_people):
    x = int(point[0])
    y = int(point[1])

    cv2.circle(
        img0, (x, y), 10, (0, 0, 0),
        thickness=-1, lineType=8)

    cv2.putText(
        img0, '0_'+str(idx), (x - 5, y + 5),
        cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), thickness=2)
            
cv2.imshow('frame', img0)
cv2.waitKey(0)

for idx, point in enumerate(img1_people):
    x = int(point[0])
    y = int(point[1])

    cv2.circle(
        img1, (x, y), 10, (0, 0, 0),
        thickness=-1, lineType=8)

    cv2.putText(
        img1, '1_'+str(idx), (x - 5, y + 5),
        cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), thickness=2)
            
cv2.imshow('frame', img1)
cv2.waitKey(0)

for idx, point in enumerate(img2_people):
    x = int(point[0])
    y = int(point[1])

    cv2.circle(
        img2, (x, y), 10, (0, 0, 0),
        thickness=-1, lineType=8)

    cv2.putText(
        img2, '2_'+str(idx), (x - 5, y + 5),
        cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), thickness=2)
            
cv2.imshow('frame', img2)
cv2.waitKey(0)

for idx, point in enumerate(img3_people):
    x = int(point[0])
    y = int(point[1])

    cv2.circle(
        img3, (x, y), 10, (0, 0, 0),
        thickness=-1, lineType=8)

    cv2.putText(
        img3, '2_'+str(idx), (x - 5, y + 5),
        cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), thickness=2)
            
cv2.imshow('frame', img3)
cv2.waitKey(0)


# Loading camera parameters
cache = diskcache.Cache('../calibration/cache')

cam0_intrinsics = cache['intrinsics'][cam0]
cam1_intrinsics = cache['intrinsics'][cam1]
cam2_intrinsics = cache['intrinsics'][cam2]
cam3_intrinsics = cache['intrinsics'][cam3]


print("ret", cam0_intrinsics['ret'])
mtx0 = cam0_intrinsics['mtx']
dist0 = cam0_intrinsics['dist']
rvec0 = cam0_intrinsics['rvecs'][0]
tvec0 = cam0_intrinsics['tvecs'][0]

print("ret", cam1_intrinsics['ret'])
mtx1 = cam1_intrinsics['mtx']
dist1 = cam1_intrinsics['dist']
rvec1 = cam1_intrinsics['rvecs'][0]
tvec1 = cam1_intrinsics['tvecs'][0]

print("ret", cam2_intrinsics['ret'])
mtx2 = cam2_intrinsics['mtx']
dist2 = cam2_intrinsics['dist']
rvec2 = cam2_intrinsics['rvecs'][0]
tvec2 = cam2_intrinsics['tvecs'][0]

print("ret", cam3_intrinsics['ret'])
mtx3 = cam3_intrinsics['mtx']
dist3 = cam3_intrinsics['dist']
rvec3 = cam3_intrinsics['rvecs'][0]
tvec3 = cam3_intrinsics['tvecs'][0]


# Trinagulate
# Convert the rotation vectors to rotation matrices
R0, _ = cv2.Rodrigues(rvec0)
R1, _ = cv2.Rodrigues(rvec1)
R2, _ = cv2.Rodrigues(rvec2)
R3, _ = cv2.Rodrigues(rvec3)

# Compute the projection matrices of each camera
P0 = np.dot(mtx0, np.hstack((R0, tvec0)))
P1 = np.dot(mtx1, np.hstack((R1, tvec1)))
P2 = np.dot(mtx2, np.hstack((R2, tvec2)))
P3 = np.dot(mtx3, np.hstack((R3, tvec3)))

# Undistort the image points
img_points0_undist = cv2.undistortPoints(img0_people.reshape(-1, 1, 2), mtx0, dist0, P=mtx0)
img_points1_undist = cv2.undistortPoints(img1_people.reshape(-1, 1, 2), mtx1, dist1, P=mtx1)
img_points2_undist = cv2.undistortPoints(img2_people.reshape(-1, 1, 2), mtx2, dist2, P=mtx2)
img_points3_undist = cv2.undistortPoints(img3_people.reshape(-1, 1, 2), mtx3, dist3, P=mtx3)

# Triangulate the 3D point
point_3d_01 = cv2.triangulatePoints(P0, P1, img_points0_undist, img_points1_undist)
point_3d_23 = cv2.triangulatePoints(P2, P3, img_points2_undist, img_points3_undist)

# Convert from homogeneous to Euclidean coordinates
point_3d_01 = cv2.convertPointsFromHomogeneous(point_3d_01.T)
point_3d_23 = cv2.convertPointsFromHomogeneous(point_3d_23.T)

point_3d_01 = point_3d_01.reshape((-1, 17, 3))
point_3d_23 = point_3d_23.reshape((-1, 17, 3))

# Print the result
print('The 3D coordinates of the point 01 are:', point_3d_01.shape)
print('The 3D coordinates of the point 12 are:', point_3d_23.shape)

# Visualize triangulated points
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
title = ax.set_title('3D Test')

# for person in point_3d_01:
#     # Define the data for the scatter plot
#     x = [point[0] for point in person]
#     y = [point[2] for point in person]
#     z = [1080 - point[1] for point in person]

#     graph = ax.scatter(x, y, z, c='r', marker='o')
#     for idx in range(len(data_loader.MMPOSE_EDGES)):
#         ax.plot(
#             (x[data_loader.MMPOSE_EDGES[idx][0]],
#                 x[data_loader.MMPOSE_EDGES[idx][1]]),
#             (y[data_loader.MMPOSE_EDGES[idx][0]],
#                 y[data_loader.MMPOSE_EDGES[idx][1]]),
#             (z[data_loader.MMPOSE_EDGES[idx][0]],
#                 z[data_loader.MMPOSE_EDGES[idx][1]])
#         )[0]

for person in point_3d_23:
    # Define the data for the scatter plot
    x = [point[0] for point in person]
    y = [point[2] for point in person]
    z = [1080 - point[1] for point in person]

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

ax.view_init(elev=1, azim=-89)

# Remove the grid background
ax.grid(False)

# Set the labels for the axes
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

ax.axes.set_xlim3d(0, 1920)
ax.axes.set_zlim3d(0, 1080)
ax.axes.set_ylim3d(0, 3000)

plt.show()
