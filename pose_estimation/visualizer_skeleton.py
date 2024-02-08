import sys
sys.path.append('../')

import diskcache
import numpy as np
import matplotlib.animation
import matplotlib.pyplot as plt

from utils import data_loader


def init_graph(poses, ax):
    lines = []
    graphs = []
    for keypoints in np.array(poses[0]):
        x = [point[0] for point in keypoints]
        y = [point[2] for point in keypoints]
        z = [point[1] for point in keypoints]
        x.append(0)
        y.append(0)
        z.append(0)

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

        ax.view_init(-160, 56, 0)
    
    return graphs, lines


def update_graph(idx, poses, graphs, lines, title, ax):
    keypoints = np.array(poses[idx])
    for person_idx in range(min(len(keypoints), len(graphs))):
        # Define the data for the scatter plot
        x = [point[0] for point in keypoints[person_idx]]
        y = [point[2] for point in keypoints[person_idx]]
        z = [point[1] for point in keypoints[person_idx]]
        x.append(0)
        y.append(0)
        z.append(0)

        graphs[person_idx]._offsets3d = (x, y, z)

        for line_idx, line in enumerate(lines[person_idx]):
            line.set_data(
                (x[data_loader.MMPOSE_EDGES[line_idx][0]],
                    x[data_loader.MMPOSE_EDGES[line_idx][1]]),
                (y[data_loader.MMPOSE_EDGES[line_idx][0]],
                    y[data_loader.MMPOSE_EDGES[line_idx][1]]))
            line.set_3d_properties(
                (z[data_loader.MMPOSE_EDGES[line_idx][0]],
                    z[data_loader.MMPOSE_EDGES[line_idx][1]])
            )
    ax.view_init(-160, -180 + (360 / 200) * idx, 0)
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

    # ax.set_xlim([1200, 2100])
    # ax.set_ylim([700, 1100])

    ani = matplotlib.animation.FuncAnimation(
        fig, update_graph, len(poses), fargs=(poses, graphs, lines, title, ax),
        interval=100, blit=False)

    plt.show()


if __name__ == "__main__":
    cache = diskcache.Cache('../calibration/cache')

    cache_process = cache.get('process', {})

    print(cache_process.keys())

    for key in cache_process.keys():
        if "skeleton_3D" in key and "smooth" not in key:
            print(f"Visualizing {key}")

            poses = cache_process[key]

            visualize_poses(poses)
