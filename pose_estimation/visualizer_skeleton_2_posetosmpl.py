import sys
sys.path.append('../')

import os
import math
import glob
import numpy as np

from scipy.spatial.transform import Rotation
from numpy.linalg import norm


import diskcache
import numpy as np
import matplotlib.animation
import matplotlib.pyplot as plt

from utils import data_loader


def init_graph(poses, ax):
    lines = []
    graphs = []

    # axis = [0, 1, 0]
    # theta = math.radians(45)
    # axis = axis / norm([axis])
    # rot = Rotation.from_rotvec(theta * axis)
    # keypoints = rot.apply(poses[0])
    keypoints = poses[0]
    
    x = [point[0] for point in keypoints]
    y = [point[2] for point in keypoints]
    z = [point[1] for point in keypoints]

    graph = ax.scatter(x, y, z, c='r', marker='o')
    graphs.append(graph)
    lines.append([])
    for idx in range(len(data_loader.MMPOSE_EDGES)):
        lines[-1].append(
            ax.plot(
                (x[data_loader.MMPOSE_EDGES[idx][0]],
                    x[data_loader.MMPOSE_EDGES[idx][1]]),
                (y[data_loader.MMPOSE_EDGES[idx][0]],
                    y[data_loader.MMPOSE_EDGES[idx][1]]),
                (z[data_loader.MMPOSE_EDGES[idx][0]],
                    z[data_loader.MMPOSE_EDGES[idx][1]])
            )[0]
        )
    
    return graphs, lines


def update_graph(idx, poses, graphs, lines, title):
    # axis = [0, 1, 0]
    # theta = math.radians(45)
    # axis = axis / norm([axis])
    # rot = Rotation.from_rotvec(theta * axis)
    # keypoints = rot.apply(poses[idx])
    keypoints = poses[idx]

    # Define the data for the scatter plot
    x = [point[0] for point in keypoints]
    y = [point[2] for point in keypoints]
    z = [point[1] for point in keypoints]

    graphs[0]._offsets3d = (x, y, z)

    for line_idx, line in enumerate(lines[0]):
        line.set_data(
            (x[data_loader.MMPOSE_EDGES[line_idx][0]],
                x[data_loader.MMPOSE_EDGES[line_idx][1]]),
            (y[data_loader.MMPOSE_EDGES[line_idx][0]],
                y[data_loader.MMPOSE_EDGES[line_idx][1]]))
        line.set_3d_properties(
            (z[data_loader.MMPOSE_EDGES[line_idx][0]],
                z[data_loader.MMPOSE_EDGES[line_idx][1]])
        )
    title.set_text('3D Test, time={}'.format(idx))

# Its too long
# Make it also more robust
def visualize_poses(poses):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    title = ax.set_title('3D Test')

    graphs, lines = init_graph(poses, ax)

    ax.view_init(elev=1, azim=-89)

    # Remove the grid background
    ax.grid(False)

    # Set the labels for the axes
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    RANGE = 1
    ax.axes.set_xlim3d(-RANGE, RANGE)
    ax.axes.set_zlim3d(-RANGE, RANGE)
    ax.axes.set_ylim3d(-RANGE, RANGE)

    ani = matplotlib.animation.FuncAnimation(
        fig, update_graph, len(poses), fargs=(poses, graphs, lines, title),
        interval=100, blit=False)

    plt.show()



if __name__ == "__main__":
    dir = "./keypoints_3d_pose2smpl"
    files = glob.glob(os.path.join(dir, "*"))

    for file in files:
        print(f"Visualizing: {file}")
        with open(file, 'rb') as handle:
            skeleton = np.load(handle)

        print(skeleton.shape)

        visualize_poses(skeleton)
