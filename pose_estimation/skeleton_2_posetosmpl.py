import os
import math
import pickle
import numpy as np

from tqdm import tqdm


DIR_INPUT = './keypoints_3d_ba'
DIR_OUTPUT = './keypoints_3d_pose2smpl'


def rotation_matrix_from_axis_angle(axis, angle):
    axis_normalized = axis / np.linalg.norm(axis)

    K = np.array([[0, -axis_normalized[2], axis_normalized[1]],
                  [axis_normalized[2], 0, -axis_normalized[0]],
                  [-axis_normalized[1], axis_normalized[0], 0]])

    R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * np.dot(K, K)

    return R


def get_normalize_rotation_matrix(skeleton):
    skeleton = np.copy(skeleton)

    # Rotation for torso plane
    p1 = skeleton[19]
    p2 = skeleton[5]
    p3 = skeleton[6]
    facing_normal = np.cross(p2 - p1, p3 - p1)
    facing_normal /= np.linalg.norm(facing_normal)
    desired_normal = np.array([math.radians(0),
                               math.radians(0),
                               math.radians(-90)])
    desired_normal = desired_normal / np.linalg.norm(desired_normal)
    axis = np.cross(facing_normal, desired_normal)
    cos_theta = np.dot(facing_normal, desired_normal)
    theta = np.arccos(cos_theta)
    R1 = rotation_matrix_from_axis_angle(axis, theta)
    skeleton = skeleton.dot(R1.T)

    # Rotation for standing axis
    p1 = skeleton[19]
    p2 = skeleton[18]
    facing_normal2 = p1 - p2
    facing_normal2 /= np.linalg.norm(facing_normal2)
    desired_normal2 = np.array([math.radians(0),
                                math.radians(90),
                                math.radians(0)])
    desired_normal2 = desired_normal2 / np.linalg.norm(desired_normal2)
    axis2 = np.cross(facing_normal2, desired_normal2)
    cos_theta2 = np.dot(facing_normal2, desired_normal2)
    theta2 = np.arccos(cos_theta2)
    R2 = rotation_matrix_from_axis_angle(axis2, theta2)

    R = R2 @ R1

    return R


def rotate_skeleton(skeleton, R):
    skeleton_rotated = skeleton.dot(R.T)

    return skeleton_rotated


def facing_angle(point1, point2, point3):
    facing_direction = np.cross(point2 - point1, point3 - point1)
    facing_direction = facing_direction / np.linalg.norm(facing_direction)

    angle_x = _calculate_angle(facing_direction, np.array([1, 0, 0]))
    angle_y = _calculate_angle(facing_direction, np.array([0, 1, 0]))
    angle_z = _calculate_angle(facing_direction, np.array([0, 0, 1]))

    return angle_x, angle_y, angle_z


def _calculate_angle(point, axis):
    point_dir = point / np.linalg.norm(point)
    axis_dir = axis / np.linalg.norm(axis)

    cosine_theta = np.dot(point_dir, axis_dir)

    angle = math.degrees(np.arccos(cosine_theta))

    return angle


def skeleton_2_numpypkl(path_input, dir_output, name):
    with open(path_input, 'rb') as handle:
        output = pickle.load(handle)

    skeleton_all = output['points_3d'].reshape(-1, 2, 26, 3)
    
    for idx_person in range(skeleton_all.shape[1]):
        skeleton = skeleton_all[:, idx_person]

        rotation = get_normalize_rotation_matrix(skeleton[0])
        translation = np.copy(skeleton[0, 19])
        scale = np.max(abs(skeleton - translation))

        for i in range(len(skeleton)):
            skeleton[i] = (skeleton[i] - translation) / scale

            # print("Facing_angle before",
            #       facing_angle(skeleton[i, 19],
            #                    skeleton[i, 5],
            #                    skeleton[i, 6]))

            skeleton[i] = rotate_skeleton(skeleton[i], rotation)

            # print("Facing_angle after",
            #       facing_angle(skeleton[i, 19],
            #                    skeleton[i, 5],
            #                    skeleton[i, 6]))
                

        output_path_skeleton = os.path.join(
            dir_output,
            f'{name}_{idx_person}_normalized.npy')
        with open(output_path_skeleton, 'wb') as handle:
            np.save(handle, skeleton)

        output_path_params = os.path.join(
            dir_output,
            f'{name}_{idx_person}_params.pkl')
        with open(output_path_params, 'wb') as handle:
            np.save(handle, {
                'rotation': rotation,
                'translation': translation,
                'scale': scale,
            })


if __name__ == "__main__":
    if not os.path.exists(DIR_OUTPUT):
        os.makedirs(DIR_OUTPUT)

    names = os.listdir(DIR_INPUT)
    for name in names:
        path = os.path.join(DIR_INPUT, name)

        print('Processing: {}'.format(path))
    
        skeleton_2_numpypkl(path, DIR_OUTPUT,
                            name.split('.')[0])
