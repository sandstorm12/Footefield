import os
import cv2
import diskcache
import data_loader

from tqdm import tqdm
from multiprocessing import Process


cache = diskcache.Cache('storage')

CHESSBOARD_COLS = 8
CHESSBOARD_ROWS = 11
CHESSBOARD_SQRS = 60.


def image_name_from_fullpath(fullpath):
    image_name = "/".join(fullpath.split("/")[-2:])
    
    return image_name


def extract_chessboardcorners(image_paths, cache, display=False):
    print(
        f"Processing --> {os.path.dirname(image_paths[0])}")
    print(
        f"Images_info available: {cache.__contains__('images_info')}")
    
    images_info = cache.get("images_info", {})

    bar = tqdm(image_paths)
    bar.set_description(image_paths[0].split("/")[-2])
    for image_path in bar:
        image_name = image_name_from_fullpath(image_path)
        image = cv2.imread(image_path)
        image_resized = cv2.resize(
            image, (image.shape[1], image.shape[0])
        )
        gray = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)

        chessboard = cv2.findChessboardCorners(
            gray, (CHESSBOARD_COLS, CHESSBOARD_ROWS), None)

        images_info[image_name] = {
            "fullpath": image_path,
            "findchessboardcorners_rgb": chessboard,
            "width": image.shape[1],
            "height": image.shape[0],
        }

        if display and chessboard[0]:
            for point in chessboard[1]:
                x = int(point[0][0])
                y = int(point[0][1])

                cv2.circle(
                    image, (x, y), 10, (123, 105, 34),
                    thickness=-1, lineType=8) 

            cv2.imshow("image", image)
            key = cv2.waitKey(0)
            if key == ord('q'):
                break

        cache["images_info"] = images_info

    return 


if __name__ == "__main__":
    processes = []
    for path_calib in data_loader.PATH_CALIBS:
        calibration_images = data_loader.list_calibration_images(path_calib)
        process = Process(
            target=extract_chessboardcorners,
            args=(calibration_images['data'][data_loader.TYPE_RGB], cache,))
        process.start()
        processes.append(process)

    for process in processes:
        process.join()
    