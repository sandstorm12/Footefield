# Foote Field 3D Human Pose Estimation

Foote field camera calibration and 3D pose estimation project.

## Components

[TODO]


## Usage

Run the scripts in this order:

1. `detect_chessboard.py`
2. `calc_intrinsic.py`
3. `calc_extrinsic.py`
4. `rgb_depth_calibration.py`


## TODO

1. Refactor: Move or remove the `calibration/rgb_resize_test.py`
1. ~~Refactor: Update the readme~~
1. ~~Feature: Calculate intrinsics and exterinsics for depth camera as well~~
1. Refactor: Instead of in-file options use arguments
1. Feature: Calculate the extrinsic parameters between all possible camera pairs (even RGB and Depth cameras of different devices) and select the minimum best extrinsic parameters for image alignment
1. Refactor: Use pickles instead of diskcache


## Known issues

1. ~~Depth estimation using single camera depth information is noisy~~
1. ~~`data_loader.downsample_keep_aspect_ratio` doesn't work correctly if the image size and the requested size are the same~~
1. ~~Cannot find matching pairs for some camera pairs for extrinsic parameter calculation~~
1. In display mode, the detect chessboard depth and rgb scipts do not go to next camera
1. Feature: Make the calibration scripts more generalizable to work with custom images
1. Feature: Move to video format
1. Refactor: Add descriptions to each of the scripts
1. Refactor: Intrinsics calculator for rgb abd depth images can be merged


## Contributors

- Hamid Mohammadi: hamid4@ualberta.ca
- Shihao Zou: szou2@ualberta.ca
