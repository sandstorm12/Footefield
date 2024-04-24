# Skeleton extraction 2D and 3D

Scripts to extract 2D skeletons and project them to 3D world coordinates using depth data and triangulation.


## Components

1. **test_mmpose.py**: Visualizes the skeleton keypoints found by the `mmpose` library on an arbitrary image.

mmpose sample 1 | mmpose sample 2
:-------------------------:|:-------------------------:
![](../README_data/mmpose_1.png)  |  ![](../README_data/mmpose_2.png)

1. **skeleton_visualizer.py**: Visualizes a 3D animation of the skeletons detected from subsequent frames of each experiment.

Skeleton animation (angle 1) | Skeleton animation (angle 2)
:-------------------------:|:-------------------------:
![](../README_data/pose_animation_1.gif)  |  ![](../README_data/pose_animation_2.gif)


## Usage

You need to first execute the scripts in the `calibration folder`.

Run the scripts in the following order:
1. [skeleton_extractor_triangulation.py](skeleton_extractor_triangulation.py) and to visualize [visualizer_skeleton_triangulation.py](visualizer_skeleton_triangulation.py)
1. [skeleton_bundle_adjustment.py](skeleton_bundle_adjustment.py) and to visualize [visualizer_skeleton_ba.py](visualizer_skeleton_ba.py) and [visualizer_skeleton_video.py](visualizer_skeleton_video.py)
1. [skeleton_2_posetosmpl.py](skeleton_2_posetosmpl.py) and to visualize [visualizer_skeleton_2_posetosmpl_open3d.py](visualizer_skeleton_2_posetosmpl_open3d.py)

Now you can fit the SMPL parameters to the 3D skeletons using the `pose2smpl` model:

```bash
python3 fit/tools/main.py --dataset_name HALPE --dataset_path path/to/keypoints_3d_pose2smpl
```

Then you can go ahead and use the scripts in the `smpl folder` for beta optimization and fine-tuning the SMPL parameters


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
1. Feature: Visualize a full scene using SMPL
1. ~~Feature: Use Halpe keypoints to get more accurate SMPL estimations~~
1. ~~Bug: Fix the triangulation issue between `3_4` and `3_5` cameras~~
1. Feature: Optimize SMPL for each frame separately
1. ~~Feature: Use `local search` to find acceptable depth for the keypoints with no depth~~
1. ~~Feature: Introduce triangulation to improve the accuracy of keypoints.~~
1. ~~Feature: Create 3D point cloud using the multi-camera depth images.~~
1. ~~Feature: Evaluate the accuracy of openpose skeleton detection as well.~~
1. Feature: We need explicit tracking of each person.
1. ~~Feature: Normalize the 3D coordinates~~
1. ~~Feature: Separate the the skeleton extraction from skeleton animation~~
1. ~~Feature: Use all the cameras for 3D skeleton extraction~~
1. ~~Feature: Visualize 3D keypoints using open3d~~
1. Refactor: Move away from visualizing keypoints in matplotlib. Open3D visualizationq results in better understanding and easier development.
1. Refactor: Remove redundant scripts


## Known issues

1.
