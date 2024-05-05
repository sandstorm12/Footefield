# SMPL-X

Fitting a SMPL-X model to the 3D skeletons and multi-view RGBD footage.


## Components

[TODO]


## Usage

For SMPL-X fitting:

1. Run [smplx_opt_alpha.py](smplx_opt_alpha.py) to get the initial SMPL-X parameters by fitting to the 3D skeletons. Visualize using [visualizer_smplx_nn.py](visualizer_smplx_nn.py) and [visualizer_smplx_video.py](visualizer_smplx_video.py).
1. Optimize using both 3D skeletons and RGB masks by running [smplx_opt_global.py](smplx_opt_global.py) and visualizer using [visualizer_smplx_video_mask.py](visualizer_smplx_video_mask.py).


## TODO

1. Feature: Implement 3D open3D full-scene SMPL-X visualizer.
1. Feature: Find a way to record open3d visualizations.


## Known issues

1. 
