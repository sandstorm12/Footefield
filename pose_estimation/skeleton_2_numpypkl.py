import os
import diskcache
import numpy as np

import math
from scipy.spatial.transform import Rotation
from numpy.linalg import norm


OUTPUT_DIR = './skeleton_numpypkl'


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

# TODO: Needs heavy refactor
def skeleton_2_numpypkl(output_dir, cache_process):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for key in cache_process.keys():
        if 'skeleton_3D_smooth' in key:
            skeleton = np.array([keypoints[0]
                                 for keypoints in cache_process[key]],
                                dtype=float)
            
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

                while True:
                    x_angle, y_angle, z_angle = line_angles(skeleton[i][5], skeleton[i][6])
                    if abs(z_angle - 80) < 1:
                        break
                    skeleton[i] = np.array(rotate_y(skeleton[i], 1))  # facing angle

                x_angle, y_angle, z_angle = line_angles(skeleton[i][5], skeleton[i][6])

                skeleton[i][20] = (skeleton[i][20] + skeleton[i][22]) / 2
                skeleton[i][21] = (skeleton[i][21] + skeleton[i][23]) / 2

                upper_scale = abs(skeleton[i][18][1] - skeleton[i][19][1])
                lower_scale = abs(skeleton[i][12][1] - skeleton[i][16][1])
                transform_scale = 3*lower_scale/(4*upper_scale)

                for j in range(len(skeleton[i])):
                    if skeleton[i][j][1] > 0:
                        skeleton[i][j][1] = skeleton[i][j][1] * transform_scale * .9
                
                COEFF_1 = 1.5
                skeleton[i][5][0] = skeleton[i][5][0] * COEFF_1
                skeleton[i][6][0] = skeleton[i][6][0] * COEFF_1
                skeleton[i][7][0] = skeleton[i][7][0] * COEFF_1
                skeleton[i][8][0] = skeleton[i][8][0] * COEFF_1
                # skeleton[i][9][0] = skeleton[i][9][0] * COEFF_1
                # skeleton[i][10][0] = skeleton[i][10][0] * COEFF_1

                COEFF_2 = 1.5
                skeleton[i][13][0] = skeleton[i][13][0] * COEFF_2
                skeleton[i][14][0] = skeleton[i][14][0] * COEFF_2
                skeleton[i][15][0] = skeleton[i][15][0] * COEFF_2
                skeleton[i][16][0] = skeleton[i][16][0] * COEFF_2

            output_path = os.path.join(output_dir, f'{key}.npy')
            with open(output_path, 'wb') as handle:
                np.save(handle, skeleton)


if __name__ == "__main__":
    cache = diskcache.Cache('../calibration/cache')
    cache_process = cache.get('process', {})

    skeleton_2_numpypkl(OUTPUT_DIR, cache_process)
