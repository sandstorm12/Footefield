import os
import cv2
import glob
import diskcache
import numpy as np


IMAGE_DIRS = [
    "/home/hamid/Documents/footefield/data/AzureKinectRecord_0729/azure_kinect1_4_calib_snap",
    "/home/hamid/Documents/footefield/data/AzureKinectRecord_0729/azure_kinect1_5_calib_snap",
    "/home/hamid/Documents/footefield/data/AzureKinectRecord_0729/azure_kinect2_4_calib_snap",
    "/home/hamid/Documents/footefield/data/AzureKinectRecord_0729/azure_kinect3_4_calib_snap",
    "/home/hamid/Documents/footefield/data/AzureKinectRecord_0729/azure_kinect3_5_calib_snap",
]

if __name__ == "__main__":
    cache = diskcache.Cache('storage')
    intrinsics = cache.get("intrinsics", None)

    for image_dir in IMAGE_DIRS:
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

            # Undistort the image using cv2.undistort function
            # The new camera matrix can be the same as the original one
            undistorted_img = cv2.undistort(img, mtx, dist, None, mtx)

            # Display the undistorted image
            cv2.imshow("Undistorted Image", cv2.resize(undistorted_img, (720, 480)))
            cv2.imshow("Original Image", cv2.resize(img, (720, 480)))
            
            if cv2.waitKey(0) == ord('q'):
                break
        
