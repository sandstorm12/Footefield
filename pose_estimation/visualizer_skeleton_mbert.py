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
    
    keypoints = poses[0]
    x = [-point[0] for point in keypoints]
    y = [-point[2] for point in keypoints]
    z = [-point[1] for point in keypoints]
    # x.append(0)
    # y.append(0)
    # z.append(0)

    graph = ax.scatter(x, y, z, c='r', marker='o')
    graphs.append(graph)
    lines.append([])
    for idx in range(len(data_loader.MBERT_EDGES)):
        lines[-1].append(
            ax.plot(
                (x[data_loader.MBERT_EDGES[idx][0]],
                    x[data_loader.MBERT_EDGES[idx][1]]),
                (y[data_loader.MBERT_EDGES[idx][0]],
                    y[data_loader.MBERT_EDGES[idx][1]]),
                (z[data_loader.MBERT_EDGES[idx][0]],
                    z[data_loader.MBERT_EDGES[idx][1]])
            )[0]
        )

    # ax.view_init(-160, 56, 0)
    ax.view_init(elev=12., azim=80)
    
    return graphs, lines


def update_graph(idx, poses, graphs, lines, title, ax):
    keypoints = np.array(poses[idx])
    
    # Define the data for the scatter plot
    x = [-point[0] for point in keypoints]
    y = [-point[2] for point in keypoints]
    z = [-point[1] for point in keypoints]
    # x.append(0)
    # y.append(0)
    # z.append(0)

    graphs[0]._offsets3d = (x, y, z)

    for line_idx, line in enumerate(lines[0]):
        line.set_data(
            (x[data_loader.MBERT_EDGES[line_idx][0]],
                x[data_loader.MBERT_EDGES[line_idx][1]]),
            (y[data_loader.MBERT_EDGES[line_idx][0]],
                y[data_loader.MBERT_EDGES[line_idx][1]]))
        line.set_3d_properties(
            (z[data_loader.MBERT_EDGES[line_idx][0]],
                z[data_loader.MBERT_EDGES[line_idx][1]])
        )
    
    # ax.view_init(-160, -180 + (360 / 200) * idx, 0)
    title.set_text('3D Test, time={}'.format(idx))

# Its too long
# Make it also more robust
def visualize_poses(poses):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    title = ax.set_title('3D Test')

    graphs, lines = init_graph(poses, ax)

    # ax.view_init(elev=1, azim=-89)
    ax.view_init(elev=12., azim=80)

    # Remove the grid background
    ax.grid(False)

    # Set the labels for the axes
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    # ax.set_xlim(-1e6, 1e6)
    # ax.set_ylim(-1e6, 1e6)
    # ax.set_zlim(-1e6, 1e6)

    ax.set_xlim(-512, 0)
    ax.set_ylim(-256, 256)
    ax.set_zlim(-512, 0)

    ani = matplotlib.animation.FuncAnimation(
        fig, update_graph, len(poses), fargs=(poses, graphs, lines, title, ax),
        interval=100, blit=False)

    plt.show()


if __name__ == "__main__":
    path = "/home/hamid/Documents/phd/footefield/MotionBERT/output/X3D.npy"

    poses = np.load(path)

    # Got it directly from the MotionBERT code so I don't understand it
    poses = np.transpose(poses, (1,2,0))
    offset = np.ones([3, poses.shape[2]]).astype(np.float32)
    offset[2,:] = 0
    poses = (poses + offset) * 512 / 2
    poses = np.transpose(poses, (2, 0, 1))

    visualize_poses(poses)  
