# Foote Field 3D Human Pose Estimation

Foote field camera calibration and 3D pose estimation project.


## Setup

```bash
pip install -r requirements
```

For mmpose installation please refer to [https://mmcv.readthedocs.io/en/latest/get_started/installation.html](https://mmcv.readthedocs.io/en/latest/get_started/installation.html)

## Usage

- Calculate intrinsic and extrinsic parameters using the scripts in the calibration dir
- Align RGB and Depth cameras using the scripts in the calibration dir
- Extract and visualize the 2D and 3D skeletons using the scripts in the pose_estimation dir


## Milestones

1. Camera calibration
    * ~~Intrinsic and extrinsic parameters~~
    * ~~RGB/Depth alignment~~
1. Initial skeleton estimation
    * ~~2D skeleton estimation~~
    * ~~3D skeleton estimation~~
1. SMPL estimation
    * ~~SMPL skeleton (joint) regression~~
    * SMPL mesh regression
1. SMPLX estimation
    * 2D face and hand keypoints estimation
    * SMPLX skeleton (joint, face, hands) regression
    * SMPLX mesh regression
1. Full-scene motion capture
    * Render full 3D scene


## TODO

1. Feature: Update readmes
1. Feature: Construct 3D point cloud of the scene using multiple cameras
1. Feature: Detect face and hand keypoints
1. Feature: Regress SMPLX joint to the body, face, and hand keypoints
1. Feature: Render SMPL the multi-person skeletons in the orignal coordinate system
1. Feature: Move to a more end-user friendly result storage from diskcache
1. Feature: Move cache to the root folder of the project
1. Feature: Add a schematic of camera placements

## Known issues

1. 3D skeleton estimation does not use all the cameras
1. 3D skeleton extraction is too hard coded


## Contributors

- Shihao Zou: szou2@ualberta.ca
- Hamid Mohammadi: hamid4@ualberta.ca
