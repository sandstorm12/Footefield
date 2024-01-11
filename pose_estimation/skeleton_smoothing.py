import sys
sys.path.append('../')

import copy
import diskcache
import numpy as np
import matplotlib.pyplot as plt

from utils import data_loader
from scipy.signal import savgol_filter


def get_person_spatiotemporal_points(idx, poses):
    person_points = np.array([item[idx] for item in poses])

    return person_points


def smooth_depth(poses):
    for idx_point in range(poses.shape[1]):
        last_non_zero = 0
        for i in range(poses.shape[0]):
            if poses[i, idx_point, 2] == 0:
                poses[i, idx_point, 2] = last_non_zero
            else:
                last_non_zero = poses[i, idx_point, 2]

        last_non_zero = 0
        for i in reversed(range(poses.shape[0])):
            if poses[i, idx_point, 2] == 0:
                poses[i, idx_point, 2] = last_non_zero
            else:
                last_non_zero = poses[i, idx_point, 2]

        
        poses[:, idx_point, 2] = savgol_filter(poses[:, idx_point, 2], 30, 1)
    

def visualize_depth(poses):
    rows = 4
    cols = 5

    fig, ax = plt.subplots(nrows=rows, ncols=cols)
    
    for i, row in enumerate(ax):
        for j, col in enumerate(row):
            idx = i*cols+j
            if idx < poses.shape[1]:
                col.plot(np.arange(0, 50), poses[:, i*cols+j, 2])
                col.set_xlim([0, 50])
                col.set_ylim([0, 3000])
            else:
                break

    plt.xlim([0, 50])
    plt.ylim([0, 3000])

    plt.show()


if __name__ == "__main__":
    cache = diskcache.Cache('../calibration/cache')

    cache_process = cache.get('process', {})

    for expriment in data_loader.EXPERIMENTS.keys():
        for dir in data_loader.EXPERIMENTS[expriment]:
            camera = dir.split("/")[-1] + "_calib_snap"
            
            id_exp = f'{expriment}_{camera}_skeleton_3D'
            id_exp_smooth = f'{expriment}_{camera}_skeleton_3D_smooth'

            print(f"Visualizing {id_exp}")

            poses = copy.deepcopy(cache_process[id_exp])

            print(len(poses),
                  len(poses[0]),
                  len(poses[0][0]),
                  len(poses[0][0][0]))

            for i in range(2):
                person_points = get_person_spatiotemporal_points(i, poses)

                smooth_depth(person_points)
                visualize_depth(person_points)

                for j in range(len(person_points)):
                    poses[j][i] = person_points[j]

            cache_process[id_exp_smooth] = poses     

    cache['process'] = cache_process
