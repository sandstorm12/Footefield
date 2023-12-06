import os
import cv2
import diskcache
import data_loader

from tqdm import tqdm
from threading import Thread


cache = diskcache.Cache('storage')

CHESSBOARD_COLS = 8
CHESSBOARD_ROWS = 11
CHESSBOARD_SQRS = 60.

_PARALLEL = True
_DISPLAY = False


def image_name_from_fullpath(fullpath):
    image_name = "/".join(fullpath.split("/")[-2:])
    
    return image_name


def extract_chessboardcorners(image_paths, images_info, display=False):
    # print(f"Processing --> {os.path.dirname(image_paths[0])}")
    
    camera_name = image_paths[0].split("/")[-2]

    success_count = 0

    bar = tqdm(image_paths)
    bar.set_description(camera_name)
    for image_path in bar:
        image_name = image_name_from_fullpath(image_path)
        image = cv2.imread(image_path)
        image_resized = cv2.resize(
            image, (image.shape[1], image.shape[0])
        )
        gray = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)

        chessboard = cv2.findChessboardCorners(
            gray, (CHESSBOARD_COLS, CHESSBOARD_ROWS),
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)

        images_info[image_name] = {
            "fullpath": image_path,
            "findchessboardcorners_rgb": chessboard,
            "width": image.shape[1],
            "height": image.shape[0],
        }

        # print(f"{chessboard[0]} --> {image_path}")

        if display:
            if chessboard[0]:
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

        if chessboard[0]:
            success_count += 1

    print(f"Found {success_count} chessboards from " +
        f"{len(image_paths)} image for {camera_name}")


if __name__ == "__main__":
    cache_available = cache.__contains__('images_info')
    print(
        f"Images_info available: {cache_available}")
    
    if not cache_available:
        cache['images_info'] = {}

    processes = []
    for path_calib in data_loader.PATH_CALIBS:
        calibration_images = data_loader.list_calibration_images(path_calib)
        process = Thread(
            target=extract_chessboardcorners,
            args=(calibration_images['data'][data_loader.TYPE_RGB], cache['images_info'], _DISPLAY))
        process.start()
        processes.append(process)

        if not _PARALLEL:
            process.join()

    for process in processes:
        process.join()

    print(f"Grand num of found chessboards: {len(cache['images_info'])}")
    