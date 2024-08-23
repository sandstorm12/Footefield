# Skeleton extraction 2D and 3D

2D and 3D pose estimation using the multi-view RGB images. Also normalize the 3D skeletons for SMPL and SMPL-X fitting.


## Components

[TODO]


## Usage

You need to first execute the scripts in the `calibration folder`.

For SMPL fitting run the scripts in the following order:
1. [skeleton_extractor_triangulation.py](skeleton_extractor_triangulation.py) and to visualize [visualizer_skeleton_triangulation.py](visualizer_skeleton_triangulation.py)
1. [skeleton_bundle_adjustment.py](skeleton_bundle_adjustment.py) and to visualize [visualizer_skeleton_ba.py](visualizer_skeleton_ba.py) and [visualizer_skeleton_video.py](visualizer_skeleton_video.py)
1. [skeleton_2_posetosmpl.py](skeleton_2_posetosmpl.py) and to visualize [visualizer_skeleton_2_posetosmpl_open3d.py](visualizer_skeleton_2_posetosmpl_open3d.py)

Now you can fit the SMPL parameters using the scripts in the `smpl` folder.

For SMPL-X fitting run the scripts in the following order:
1. [skeleton_extractor_triangulation_x.py](skeleton_extractor_triangulation_x.py) and to visualize [visualizer_skeleton_triangulation_x.py](visualizer_skeleton_triangulation_x.py)
1. [skeleton_bundle_adjustment_x.py](skeleton_bundle_adjustment_x.py) and to visualize [visualizer_skeleton_ba_x.py](visualizer_skeleton_ba_x.py) and [visualizer_skeleton_video_x.py](visualizer_skeleton_video_x.py)
1. [skeleton_2_posetosmpl_x.py](skeleton_2_posetosmpl_x.py) and to visualize [visualizer_skeleton_2_posetosmpl_open3d_x.py](visualizer_skeleton_2_posetosmpl_open3d_x.py)

Now you can fit the SMPL-X parameters using the scripts in the `smplx` folder.


## Troubleshooting

- If you get this error:

    ```bash
    libGL error: MESA-LOADER: failed to open radeonsi: /usr/lib/dri/radeonsi_dri.so: cannot open shared object file: No such file or directory (search paths /usr/lib/x86_64-linux-gnu/dri:\$${ORIGIN}/dri:/usr/lib/dri, suffix _dri)
    libGL error: failed to load driver: radeonsi
    libGL error: MESA-LOADER: failed to open radeonsi: /usr/lib/dri/radeonsi_dri.so: cannot open shared object file: No such file or directory (search paths /usr/lib/x86_64-linux-gnu/dri:\$${ORIGIN}/dri:/usr/lib/dri, suffix _dri)
    libGL error: failed to load driver: radeonsi
    libGL error: MESA-LOADER: failed to open swrast: /usr/lib/dri/swrast_dri.so: cannot open shared object file: No such file or directory (search paths /usr/lib/x86_64-linux-gnu/dri:\$${ORIGIN}/dri:/usr/lib/dri, suffix _dri)
    libGL error: failed to load driver: swrast
    [Open3D WARNING] GLFW Error: GLX: Failed to create context: GLXBadFBConfig
    [Open3D WARNING] Failed to create window
    ```

    Install this package:

    ```bash
    conda install -c conda-forge libstdcxx-ng
    ```


## TODO

1. ~~Bug: Fix tracking from all camera pairs for triangulation base 3d skeleton detection~~
1. ~~Feature: Automate the skeleton normalization (+rotation+smplfix)~~
1. ~~Feature: Visualize a full scene using SMPL~~
1. ~~Feature: Use Halpe keypoints to get more accurate SMPL estimations~~
1. ~~Bug: Fix the triangulation issue between `3_4` and `3_5` cameras~~
1. ~~Feature: Optimize SMPL for each frame separately~~
1. ~~Feature: Use `local search` to find acceptable depth for the keypoints with no depth~~
1. ~~Feature: Introduce triangulation to improve the accuracy of keypoints.~~
1. ~~Feature: Create 3D point cloud using the multi-camera depth images.~~
1. ~~Feature: Evaluate the accuracy of openpose skeleton detection as well.~~
1. Feature: We need explicit tracking of each person.
1. ~~Feature: Normalize the 3D coordinates~~
1. ~~Feature: Separate the the skeleton extraction from skeleton animation~~
1. ~~Feature: Use all the cameras for 3D skeleton extraction~~
1. ~~Feature: Visualize 3D keypoints using open3d~~
1. ~~Refactor: Move away from visualizing keypoints in matplotlib. Open3D visualizationq results in better understanding and easier development.~~
1. ~~Refactor: Remove redundant scripts~~
1. Feature: Add tracking id to the recorded poses
1. Refactor: Merge wholebody scripts with the normal ones
1. Refactor: Separate global param calculation from the triangulation script
1. Bug: Make the config path mandatory when the default path is not given
1. Feautre: Make the skeleton detection multi-thread
1. Feature: Make extrinsic finder independent of the number of cameras
1. Bug: We need separate translation and rotation for each timestep of skeleton normalizer and then propagate the change forward


## Known issues

1.
