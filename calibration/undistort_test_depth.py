import sys
sys.path.append('../')

import os
import cv2
import glob
import diskcache
import numpy as np

from utils import data_loader


if __name__ == "__main__":
    cache = diskcache.Cache('cache')
    intrinsics = cache.get("intrinsics", None)

    for image_dir in data_loader.PATH_CALIBS:
        key = image_dir.split("/")[-1]
        print(key)

        ret = intrinsics[key + '_infrared']['ret']
        mtx = intrinsics[key + '_infrared']['mtx']
        dist = intrinsics[key + '_infrared']['dist']

        print(f"Ret: {ret}")
        print(f"mtx: {mtx.shape}")
        print(f"dist: {dist.shape}")

        file_paths = glob.glob(os.path.join(image_dir, "infrared*"))
        for file_path in file_paths:
            img = cv2.imread(file_path, -1)
            img = np.clip(
                img.astype(np.float32) * .8, 0, 255).astype('uint8')

            undistorted_img = cv2.undistort(img, mtx, dist, None, mtx)

            cv2.imshow("Undistorted Image", undistorted_img)
            cv2.imshow("Original Image", img)
            
            if cv2.waitKey(0) == ord('q'):
                break
        
