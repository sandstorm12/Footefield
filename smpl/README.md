# Skinned Multi-person Linear Model

Fitting a SMPL model to the 3D skeletons and multi-view RGBD footage.


## Components

[TODO]


## Usage

For SMPL fitting:

1. Run [smpl_opt_alpha.py](smpl_opt_alpha.py) to get the initial SMPL parameters by fitting to the 3D skeletons. Visualize using [visualizer_smpl_nn.py](visualizer_smpl_nn.py) and [visualizer_smpl_video.py](visualizer_smpl_video.py).
1. Optimize using both 3D skeletons and RGB masks by running [smpl_opt_global.py](smpl_opt_global.py) and visualizer using [visualizer_smpl_video_opt_mask.py](visualizer_smpl_video_opt_mask.py).


## TODO

1. ~~Feature: Automate point cloud registeration~~
1. ~~Feature: Combine the RGBD point cloud from all cameras~~
1. Feature: Automate or make it easier to register pointclouds
1. ~~Feature: Extract each person's point cloud separately~~
1. ~~Feature: Align pointclouds and 3D skeletons~~
1. ~~Feature: Get intrinsic parameters from the more accurate single camera chessboard calibration results~~
1. ~~Feature: Combine individually registered pointclouds in a full scene view~~
1. Refactor: Add the point cloud scripts into the pipeline.
1. Refactor: Add the mask extractor to this repo.
1. Refactor: Add rotation optimization to the fitter.


## Known issues

1. ~~Point clouds seems not matching perfectly with each other~~
1. ~~Giving the output of the manual registeration as its input doesn't create the same rotation and transition.~~
