import os
import pickle
import diskcache
import numpy as np

import math
from scipy.spatial.transform import Rotation
from numpy.linalg import norm


DIR_INPUT = './keypoints_3d_ba'
DIR_OUTPUT = './keypoints_3d_pose2smpl'


def magnitude (v):
    return math.sqrt (sum (v [i] ** 2 for i in range (len (v))))


def dot_product (v1, v2):
    return sum (v1 [i] * v2 [i] for i in range (len (v1)))


def angle (v1, v2):
    cos_theta = dot_product (v1, v2) / (magnitude (v1) * magnitude (v2))

    return math.degrees (math.acos (cos_theta))


def vector_difference (p1, p2):
    return tuple (p2 [i] - p1 [i] for i in range (len (p1)))


def line_angles (p1, p2):
    # Define the x, y, and z axis as unit vectors
    x_axis = (1, 0, 0)
    y_axis = (0, 1, 0)
    z_axis = (0, 0, 1)

    # Calculate the line vector as the difference of the two points
    line = vector_difference (p1, p2)

    # Calculate the angle between the line and the x, y, and z axis
    angle_x = angle (line, x_axis)
    angle_y = angle (line, y_axis)
    angle_z = angle (line, z_axis)

    # Return the angles as a tuple
    return (angle_x, angle_y, angle_z)


def get_plane(a, b, c):
    u = b - a
    v = c - a

    # find the normal vector to the plane by taking the cross product of u and v
    n = np.cross(u, v)

    # normalize the normal vector to have unit length
    n = n / np.linalg.norm(n)

    return n


def rotate_x(points, angle):
    rad = math.radians(angle)
    cos = math.cos(rad)
    sin = math.sin(rad)
    rotation_matrix = [[1, 0, 0], [0, cos, -sin], [0, sin, cos]]
    return [[sum([rotation_matrix[i][j] * point[j] for j in range(3)]) for i in range(3)] for point in points]


def rotate_y(points, angle):
    rad = math.radians(angle)
    cos = math.cos(rad)
    sin = math.sin(rad)
    rotation_matrix = [[cos, 0, sin], [0, 1, 0], [-sin, 0, cos]]
    return [[sum([rotation_matrix[i][j] * point[j] for j in range(3)]) for i in range(3)] for point in points]


def rotate_z(points, angle):
    rad = math.radians(angle)
    cos = math.cos(rad)
    sin = math.sin(rad)
    rotation_matrix = [[cos, -sin, 0], [sin, cos, 0], [0, 0, 1]]
    return [[sum([rotation_matrix[i][j] * point[j] for j in range(3)]) for i in range(3)] for point in points]


def skeleton_2_numpypkl(path_input, dir_output, name):
    with open(path_input, 'rb') as handle:
        output = pickle.load(handle)

    skeleton = output['points_3d']
    skeleton = skeleton.reshape(-1, 2, 26, 3)
    skeleton = skeleton[:, 0]

    print(skeleton.shape)

    x_range = np.max(skeleton[:, :, 0]) - np.min(skeleton[:, :, 0])
    middle = np.copy(skeleton[:, 19])
    y_range = np.max(skeleton[:, :, 1]) - np.min(skeleton[:, :, 1])
    z_range = np.max(skeleton[:, :, 2]) - np.min(skeleton[:, :, 2])
    g_range = max(x_range, y_range, z_range)

    x_angle, y_angle, z_angle = line_angles(skeleton[0][11], skeleton[0][12])
    for i in range(len(skeleton)):
        for j in range(len(skeleton[i])):
            skeleton[i][j] = (skeleton[i][j] - middle[i]) / g_range
            skeleton[i][j][1] *= -1

        skeleton[i] = np.array(rotate_y(skeleton[i], z_angle - 50))  # facing angle
        skeleton[i] = np.array(rotate_z(skeleton[i], x_angle - 90))  # facing angle

        # while True:
        #     # x_angle, y_angle, z_angle = line_angles(skeleton[i][11], skeleton[i][12])
        #     if z_angle - 90 < 5:
        #         break
        #     skeleton[i] = np.array(rotate_y(skeleton[i], z_angle - 90))  # facing angle

        # while True:
        #     # x_angle, y_angle, z_angle = line_angles(skeleton[i][11], skeleton[i][12])
        #     if abs(x_angle - 180) < 5:
        #         break
        #     skeleton[i] = np.array(rotate_z(skeleton[i], x_angle - 180))  # facing angle
            

    output_path = os.path.join(dir_output, f'{name}.npy')
    with open(output_path, 'wb') as handle:
        np.save(handle, skeleton)


if __name__ == "__main__":
    if not os.path.exists(DIR_OUTPUT):
        os.makedirs(DIR_OUTPUT)

    names = os.listdir(DIR_INPUT)
    for name in names:
        path = os.path.join(DIR_INPUT, name)

        print('Processing: {}'.format(path))
    
        skeleton_2_numpypkl(path, DIR_OUTPUT,
                            name.split('.')[0])
