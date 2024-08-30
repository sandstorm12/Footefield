import yaml
import math
import argparse
import numpy as np


def _get_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-c', '--config',
        help='Path to the config file',
        type=str,
        default='configs/skeleton_3d_normalizer_wb.yml',
    )

    args = parser.parse_args()

    return args


def _load_configs(path):
    with open(path, 'r') as yaml_file:
        configs = yaml.safe_load(yaml_file)

    return configs


def rotation_matrix_from_axis_angle(axis, angle):
    axis_normalized = axis / np.linalg.norm(axis)

    K = np.array([[0, -axis_normalized[2], axis_normalized[1]],
                  [axis_normalized[2], 0, -axis_normalized[0]],
                  [-axis_normalized[1], axis_normalized[0], 0]])

    R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * np.dot(K, K)

    return R


def get_normalize_rotation_matrix(skeleton):
    skeleton = np.copy(skeleton)

    # Rotation for standing axis
    p1 = (skeleton[11] + skeleton[12]) / 2
    p2 = (skeleton[5] + skeleton[6]) / 2
    facing_normal2 = p1 - p2
    facing_normal2 /= np.linalg.norm(facing_normal2)
    desired_normal2 = np.array([math.radians(0),
                                math.radians(-90),
                                math.radians(0)])
    desired_normal2 = desired_normal2 / np.linalg.norm(desired_normal2)
    axis2 = np.cross(facing_normal2, desired_normal2)
    cos_theta2 = np.dot(facing_normal2, desired_normal2)
    theta2 = np.arccos(cos_theta2)
    R2 = rotation_matrix_from_axis_angle(axis2, theta2)
    skeleton = skeleton.dot(R2.T)

    # Rotation for torso plane
    p1 = skeleton[11]
    p2 = skeleton[12]
    facing_normal = p1 - p2
    facing_normal[1] = 0
    facing_normal /= np.linalg.norm(facing_normal)
    desired_normal = np.array([math.radians(90),
                               math.radians(0),
                               math.radians(0)])
    desired_normal = desired_normal / np.linalg.norm(desired_normal)
    axis = np.cross(facing_normal, desired_normal)
    cos_theta = np.dot(facing_normal, desired_normal)
    theta = np.arccos(cos_theta)
    R1 = rotation_matrix_from_axis_angle(axis, theta)

    R = R1 @ R2

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


def normalize_skeleton(path_input, path_output):
    with open(path_input) as handler:
        poses = np.array(yaml.safe_load(handler))

    poses_normalized = []
    for idx_person in range(poses.shape[1]):
        skeleton = poses[:, idx_person]

        rotation = np.array([get_normalize_rotation_matrix(skeleton[i])
                             for i in range(len(skeleton))])
        translation = np.copy((skeleton[:, 11] + skeleton[:, 12]) / 2)
        for i in range(len(skeleton)):
            skeleton[i] = skeleton[i] - translation[i]

            skeleton[i] = rotate_skeleton(skeleton[i], rotation[i])

        scale = np.max(abs(skeleton))
        for i in range(len(skeleton)):
            skeleton[i] = skeleton[i] / scale

        bundle = {
            'pose_normalized': skeleton.tolist(),
            'rotation': rotation.tolist(),
            'translation': translation.tolist(),
            'scale': scale.item(),
        }
        poses_normalized.append(bundle)

    _store_artifacts(poses_normalized, path_output)


def _store_artifacts(artifact, output):
    with open(output, 'w') as handle:
        yaml.dump(artifact, handle)


if __name__ == "__main__":
    args = _get_arguments()
    configs = _load_configs(args.config)

    print(f"Config loaded: {configs}")

    normalize_skeleton(
        configs['skeletons_3d'],
        configs['output'])
