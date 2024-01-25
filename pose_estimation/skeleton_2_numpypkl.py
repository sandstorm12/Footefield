import os
import diskcache
import numpy as np

import math
from scipy.spatial.transform import Rotation
from numpy.linalg import norm


OUTPUT_DIR = './skeleton_numpypkl'


# Define a function to calculate the magnitude of a vector
def magnitude (v):
    return math.sqrt (sum (v [i] ** 2 for i in range (len (v))))

# Define a function to calculate the dot product of two vectors
def dot_product (v1, v2):
    return sum (v1 [i] * v2 [i] for i in range (len (v1)))

# Define a function to calculate the angle between two vectors
def angle (v1, v2):
    cos_theta = dot_product (v1, v2) / (magnitude (v1) * magnitude (v2))
    return math.degrees (math.acos (cos_theta))

# Define a function to calculate the vector difference of two points
def vector_difference (p1, p2):
    return tuple (p2 [i] - p1 [i] for i in range (len (p1)))

# Define a function to calculate the angles between a line and the x, y, and z axis
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


def skeleton_2_numpypkl(output_dir, cache_process):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for key in cache_process.keys():
        if 'skeleton_3D_smooth' in key:
            skeleton = np.array([keypoints[0]
                                 for keypoints in cache_process[key]],
                                dtype=float)
            
            print(skeleton.shape)
            
            if skeleton.shape[0] != 200:
                continue

            x_range = np.max(skeleton[:, :, 0]) - np.min(skeleton[:, :, 0])
            middle = np.copy(skeleton[:, 19])
            y_range = np.max(skeleton[:, :, 1]) - np.min(skeleton[:, :, 1])
            z_range = np.max(skeleton[:, :, 2]) - np.min(skeleton[:, :, 2])
            g_range = max(x_range, y_range, z_range) / 2

            for i in range(len(skeleton)):
                for j in range(len(skeleton[i])):
                    skeleton[i][j] = (skeleton[i][j] - middle[i]) / g_range
                    skeleton[i][j][1] *= -1

                x_angle, y_angle, z_angle = line_angles(skeleton[i][5], skeleton[i][6])

                print("Before", x_angle, y_angle, z_angle)

                while True:
                    x_angle, y_angle, z_angle = line_angles(skeleton[i][5], skeleton[i][6])
                    if abs(z_angle - 90) < 1:
                        break
                    skeleton[i] = np.array(rotate_y(skeleton[i], 1))  # facing angle

                x_angle, y_angle, z_angle = line_angles(skeleton[i][5], skeleton[i][6])

                print("After", x_angle, y_angle, z_angle)

                scale = np.sqrt(np.sum((skeleton[i][8] - skeleton[i][10]) ** 2))
                print(scale)

                # skeleton[i][21] = (skeleton[i][21] + skeleton[i][23]) / 2
                # skeleton[i][20] = (skeleton[i][20] + skeleton[i][22]) / 2

                # skeleton[i][5][1] = skeleton[i][5][1] - .4 * scale
                # skeleton[i][6][1] = skeleton[i][6][1] - .4 * scale
                # skeleton[i][7][1] = skeleton[i][7][1] - .4 * scale
                # skeleton[i][8][1] = skeleton[i][8][1] - .4 * scale
                # skeleton[i][9][1] = skeleton[i][9][1] - .4 * scale
                # skeleton[i][10][1] = skeleton[i][10][1] - .4 * scale
                
                # skeleton[i][11][0] = skeleton[i][11][0] - .15 * scale
                # skeleton[i][11][1] = skeleton[i][11][1] + .4 * scale

                # skeleton[i][12][0] = skeleton[i][12][0] + .15 * scale
                # skeleton[i][12][1] = skeleton[i][12][1] + .4 * scale

                # skeleton[i][18][1] = skeleton[i][18][1] - .4 * scale
                # skeleton[i][19][1] = skeleton[i][19][1] + .4 * scale
                
            output_path = os.path.join(output_dir, f'{key}.npy')
            with open(output_path, 'wb') as handle:
                np.save(handle, skeleton)


if __name__ == "__main__":
    cache = diskcache.Cache('../calibration/cache')
    cache_process = cache.get('process', {})

    skeleton_2_numpypkl(OUTPUT_DIR, cache_process)
