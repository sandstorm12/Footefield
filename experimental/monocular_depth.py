import numpy as np
import open3d as o3d


depth = np.load("color00000_depth_fp32.npy")
print(np.min(depth), np.mean(depth), np.max(depth))

points = np.empty(shape=(depth.shape[0] * depth.shape[1], 3))
idx = 0
for y in range(depth.shape[0]):
    for x in range(depth.shape[1]):
        depth_value = depth[y, x] * 1000
        x_value = x
        y_value = y
        points[idx] = (x_value, y_value, depth_value)
        idx += 1
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)

# depth_o3d = o3d.geometry.Image(depth)
# intrinsics = o3d.camera.PinholeCameraIntrinsic(1920, 1080, 924.45, 926.66, 959.71, 594.05)
# pcd = o3d.geometry.PointCloud.create_from_depth_image(depth_o3d, intrinsics)

o3d.visualization.draw_geometries([pcd])