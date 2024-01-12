import os
import diskcache
import numpy as np


OUTPUT_DIR = './skeleton_numpypkl'


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

            output_path = os.path.join(output_dir, f'{key}.npy')
            with open(output_path, 'wb') as handle:
                np.save(handle, skeleton)


if __name__ == "__main__":
    cache = diskcache.Cache('../calibration/cache')
    cache_process = cache.get('process', {})

    skeleton_2_numpypkl(OUTPUT_DIR, cache_process)
