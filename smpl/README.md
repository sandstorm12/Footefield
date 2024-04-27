# Skinned Multi-person Linear Model

Converting the 3D skeleton keypoints and RGBD point cloud to the skinned model of human

## Components

[complete here]


## Usage

After obtaining the SMPL parameters using the scripts in the `pose_estimation folder`, run the scripts in the following order:

1. You might want to visualize the SMPL + 3D keypoints fitting results: run [visualizer_smpl.py](visualizer_smpl.py) for an Open3D visualization and [visualizer_smpl_video.py](visualizer_smpl_video.py) for a video reprojection visualization.


## TODO

1. ~~Feature: Automate point cloud registeration~~
1. ~~Feature: Combine the RGBD point cloud from all cameras~~
1. Feature: Automate or make it easier to register pointclouds
1. Feature: Extract each person's point cloud separately
1. Feature: Align pointclouds and 3D skeletons
1. Feature: Get intrinsic parameters from the more accurate single camera chessboard calibration results
1. Feature: Combine individually registered pointclouds in a full scene view


## Known issues

1. ~~Point clouds seems not matching perfectly with each other~~
1. Giving the output of the manual registeration as its input doesn't create the same rotation and transition.
