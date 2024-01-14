import os
import diskcache
import numpy as np

import math
from scipy.spatial.transform import Rotation
from numpy.linalg import norm


OUTPUT_DIR = './skeleton_numpypkl'


def calc_y_angle(p1, p2):
    a_x = p2[0] - p1[0]
    a_y = p2[1] - p1[1]
    a_z = p2[2] - p1[2]

    # Calculate the magnitude of a
    a_mag = math.sqrt (a_x ** 2 + a_y ** 2 + a_z ** 2)

    # Calculate the dot product of a and b
    b_y = 1 # The unit vector along the y axis
    dot = a_y * b_y

    # Calculate the cosine of the angle
    cos_theta = dot / a_mag

    # Calculate the angle in radians
    theta_rad = math.acos (cos_theta)

    # Convert the angle to degrees
    theta_deg = math.degrees (theta_rad)

    return theta_deg


def skeleton_2_numpypkl(output_dir, cache_process):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for key in cache_process.keys():
        if 'skeleton_3D_smooth' in key:
            skeleton = np.array([keypoints[0]
                                 for keypoints in cache_process[key]],
                                dtype=float)
            x_range = np.max(skeleton[:, :, 0]) - np.min(skeleton[:, :, 0])
            x_middle = int(np.max(skeleton[:, :, 0]) + np.min(skeleton[:, :, 0])) // 2
            y_range = np.max(skeleton[:, :, 1]) - np.min(skeleton[:, :, 1])
            y_middle = int(np.max(skeleton[:, :, 1]) + np.min(skeleton[:, :, 1])) // 2
            z_range = np.max(skeleton[:, :, 2]) - np.min(skeleton[:, :, 2])
            z_middle = int(np.max(skeleton[:, :, 2]) + np.min(skeleton[:, :, 2])) // 2
            g_range = max(x_range, y_range, z_range) / 2
            
            for i in range(len(skeleton)):
                for j in range(len(skeleton[i])):
                    #TODO: Remove the normalization from here
                    skeleton[i][j][0] = (skeleton[i][j][0] - x_middle) / g_range
                    skeleton[i][j][1] = ((skeleton[i][j][1] - y_middle) / g_range)
                    skeleton[i][j][2] = (skeleton[i][j][2] - z_middle) / g_range

                angle = calc_y_angle(skeleton[i][11], skeleton[i][12])
                print(angle)

                axis = [0, 1, 0]
                theta = math.radians(angle)
                axis = axis / norm([axis])
                rot = Rotation.from_rotvec(theta * axis)
                skeleton[i] = rot.apply(skeleton[i])

                skeleton[i][7][0] += .05
                skeleton[i][8][0] += -.05
                skeleton[i][13][0] += .03
                skeleton[i][14][0] += -.05

            output_path = os.path.join(output_dir, f'{key}.npy')
            with open(output_path, 'wb') as handle:
                np.save(handle, skeleton)


if __name__ == "__main__":
    cache = diskcache.Cache('../calibration/cache')
    cache_process = cache.get('process', {})

    skeleton_2_numpypkl(OUTPUT_DIR, cache_process)
