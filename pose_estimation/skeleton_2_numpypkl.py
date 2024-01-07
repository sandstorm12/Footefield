import os
import diskcache
import numpy as np

import matplotlib.pyplot as plt


cache = diskcache.Cache('../calibration/cache')

cache_process = cache.get('process', {})

OUTPUT_DIR = './skeleton_numpypkl'

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

for key in cache_process.keys():
    if 'skeleton_3D' in key:
        print(key)
        print(type(cache_process[key]),
              len(cache_process[key]),
              len(cache_process[key][0]),
              len(cache_process[key][0][0]),
              len(cache_process[key][0][0][0]))
        
        skeleton = np.array([keypoints[0] for keypoints in cache_process[key]], dtype=float)
        for i in range(len(skeleton)):
            for j in range(len(skeleton[i])):
                skeleton[i][j][0] = (skeleton[i][j][0] / (1920. / 2.) - 1)
                skeleton[i][j][1] = (skeleton[i][j][1] / (1080. / 2.) - 1)
                skeleton[i][j][2] = 1 - (skeleton[i][j][2] / (4000. / 2.) - 1)

                print(skeleton[i][j][0], skeleton[i][j][1])


        print(skeleton.shape)

        # import numpy as np
        # from mmhuman3d.core.conventions.keypoints_mapping import convert_kps

        # keypoints_coco, mask = convert_kps(skeleton, approximate=True, src='coco', dst='h36m')
        # print(keypoints_coco.shape)
        # print(np.min(keypoints_coco), np.mean(keypoints_coco), np.max(keypoints_coco))
        # print(sum(mask))
        # # assert mask.all()==1

        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # title = ax.set_title('3D Test')

        # # Define the data for the scatter plot
        # x = [point[0] for point in keypoints_coco[0]]
        # y = [point[2] for point in keypoints_coco[0]]
        # z = [1080 - point[1] for point in keypoints_coco[0]]

        # graph = ax.scatter(x, y, z, c='r', marker='o')

        # ax.view_init(elev=1, azim=-89)

        # # Remove the grid background
        # ax.grid(False)

        # # Set the labels for the axes
        # ax.set_xlabel('X Label')
        # ax.set_ylabel('Y Label')
        # ax.set_zlabel('Z Label')

        # ax.axes.set_xlim3d(0, 1920)
        # ax.axes.set_zlim3d(0, 1080)
        # ax.axes.set_ylim3d(0, 3000)

        # plt.show()

        with open(os.path.join(OUTPUT_DIR, f'{key}.npy'), 'wb') as handle:
            np.save(handle, skeleton)