import sys
sys.path.append('../')

import os
import cv2
import glob
import diskcache

from utils import data_loader


if __name__ == "__main__":
    cache = diskcache.Cache('cache')
    intrinsics = cache.get("intrinsics", None)

    for image_dir in data_loader.PATH_CALIBS:
        key = image_dir.split("/")[-1]
        print(key)

        ret = intrinsics[key]['ret']
        mtx = intrinsics[key]['mtx']
        dist = intrinsics[key]['dist']

        print(f"Ret: {ret}")
        print(f"mtx: {mtx.shape}")
        print(f"dist: {dist.shape}")

        file_paths = glob.glob(os.path.join(image_dir, "color*"))
        for file_path in file_paths:
            img = cv2.imread(file_path)

            undistorted_img = cv2.undistort(img, mtx, dist, None, mtx)

            cv2.imshow("Undistorted Image", undistorted_img)
            cv2.imshow("Original Image", img)
            
            if cv2.waitKey(0) == ord('q'):
                break
        
