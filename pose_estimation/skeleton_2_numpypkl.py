import os
import diskcache
import numpy as np


OUTPUT_DIR = './skeleton_numpypkl'


def skeleton_2_numpypkl(output_dir, cache_process):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for key in cache_process.keys():
        if 'skeleton_3D' in key:
            skeleton = np.array([keypoints[0]
                                 for keypoints in cache_process[key]],
                                dtype=float)
            for i in range(len(skeleton)):
                for j in range(len(skeleton[i])):
                    #TODO: Remove the normalization from here
                    skeleton[i][j][0] = (skeleton[i][j][0] / (1920. / 2.) - 1)
                    skeleton[i][j][1] = (skeleton[i][j][1] / (1080. / 2.) - 1)
                    skeleton[i][j][2] = 1 - (skeleton[i][j][2] / (4000. / 2.) - 1)

            output_path = os.path.join(output_dir, f'{key}.npy')
            with open(output_path, 'wb') as handle:
                np.save(handle, skeleton)


if __name__ == "__main__":
    cache = diskcache.Cache('../calibration/cache')
    cache_process = cache.get('process', {})

    skeleton_2_numpypkl(OUTPUT_DIR, cache_process)
