# Foote Field 3D Human Pose Estimation

Foote field camera calibration and 3D pose estimation project.

## Components

1. **calibration/detect_chessboard.py**: Loads calibration images and detects the chessboard in the image if present. Depends on `calibration/detect_chessboard_rgb.py` and `calibration/detect_chessboard_infrared.py`.

Image RGB chessboard points | Image infrared chessboard points
:-------------------------:|:-------------------------:
![](../README_data/chessboard_rgb.png)  |  ![](../README_data/chessboard_infrared.png)

1. **calibration/calc_intrinsic.py**: Using the cached chessboard points detected by `calibration/detect_chessboard.py` calculates the intrinsic parameters of each camera.

Distorted image | Undistorted image
:-------------------------:|:-------------------------:
![](../README_data/distorted.png)  |  ![](../README_data/undistorted.png)

1. **calibration/calc_extrinsic.py**: Finds matching images from all images with a valid chessboard detected. Finds extrinsic parameters between each pair of matching cameras.

Left stereo matched image | Right stereo matched image
:-------------------------:|:-------------------------:
![](../README_data/left.png)  |  ![](../README_data/right.png)

1. **calibration/rgb_depth_calibration.py**: Aligns the RGB and infrared images from each camera to enable accurate depth estimation of each point on the image.

RGB/infrared match 1 | RGB/infrared match 2
:-------------------------:|:-------------------------:
![](../README_data/align_1.png)  |  ![](../README_data/align_2.png)

1. **calibration/rgb_depth_map.py**: Finds the corresponding points on the RGB and depth image by aligning the two images.

RGB and corresponding depth match 1 |
:-------------------------:|
![](../README_data/depth_map_1.png)  |



## Usage

Run the scripts in this order:

`calibration/detect_chessboard.py` -> `calibration/calc_intrinsic.py` -> `calibration/calc_extrinsic.py` -> `calibration/rgb_depth_calibration.py`


## TODO

1. Refactor: Move or remove the `calibration/rgb_resize_test.py`
1. Refactor: Update the readme


## Known issues

1. ~~Depth estimation using single camera depth information is noisy~~


## Contributors

- Shihao Zou: szou2@ualberta.ca
- Hamid Mohammadi: hamid4@ualberta.ca