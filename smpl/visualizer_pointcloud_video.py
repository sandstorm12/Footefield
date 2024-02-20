import cv2
import time
import diskcache
import numpy as np
import open3d as o3d


def get_cam(cam_name):
    return f'azure_kinect{cam_name}_calib_snap'

def get_images(cam_name, idx):
    img_depth = '/home/hamid/Documents/phd/footefield/data/AzureKinectRecord_0729/a1/azure_kinect{}/depth/depth{:05d}.png'.format(cam_name, idx)
    img_color = '/home/hamid/Documents/phd/footefield/data/AzureKinectRecord_0729/a1/azure_kinect{}/color/color{:05d}.jpg'.format(cam_name, idx)

    return img_color, img_depth


def rotation_matrix_from_euler_angles(roll, pitch, yaw):
    """
        Creates a 3x3 rotation matrix from euler angles (roll, pitch, yaw).

        Args:
        roll (float): Roll angle in radians.
        pitch (float): Pitch angle in radians.
        yaw (float): Yaw angle in radians.

        Returns:
        rotation_matrix (np.ndarray): 3x3 rotation matrix.
    """

    rotation_matrix = np.identity(3)

    # Roll
    c_roll = np.cos(roll)
    s_roll = np.sin(roll)

    rotation_matrix = np.dot(rotation_matrix, np.array(
        [[1, 0, 0],
        [0, c_roll, -s_roll],
        [0, s_roll, c_roll]]))

    # Pitch
    c_pitch = np.cos(pitch)
    s_pitch = np.sin(pitch)

    rotation_matrix = np.dot(rotation_matrix, np.array(
        [[c_pitch, 0, s_pitch],
        [0, 1, 0],
        [-s_pitch, 0, c_pitch]]))

    # Yaw
    c_yaw = np.cos(yaw)
    s_yaw = np.sin(yaw)

    rotation_matrix = np.dot(rotation_matrix, np.array(
        [[c_yaw, s_yaw, 0],
        [-s_yaw, c_yaw, 0],
        [0, 0, 1]]))

    return rotation_matrix


def get_pcd(cam, idx, cache):
    img_color_0, img_depth_0 = get_images(cam, idx)

    if cam == cam_name_0:
        cam0 = get_cam(cam_name_0)
        mtx0 = cache['extrinsics'][cam0 + 'infrared']['mtx_l']
    elif cam == cam_name_1:
        cam0 = get_cam(cam_name_0)
        mtx0 = cache['extrinsics'][cam0 + 'infrared']['mtx_r']
        R = cache['extrinsics'][cam0 + 'infrared']['rotation']
        T = cache['extrinsics'][cam0 + 'infrared']['transition']
    elif cam == cam_name_2:
        cam0 = get_cam(cam_name_1)
        mtx0 = cache['extrinsics'][cam0 + 'infrared']['mtx_r']
        R = cache['extrinsics'][get_cam(cam_name_0) + 'infrared']['rotation']
        T = cache['extrinsics'][get_cam(cam_name_0) + 'infrared']['transition']
        R2 = cache['extrinsics'][cam0 + 'infrared']['rotation']
        T2 = cache['extrinsics'][cam0 + 'infrared']['transition']
    elif cam == cam_name_3:
        cam0 = get_cam(cam_name_2)
        mtx0 = cache['extrinsics'][cam0 + 'infrared']['mtx_r']
        R = cache['extrinsics'][get_cam(cam_name_0) + 'infrared']['rotation']
        T = cache['extrinsics'][get_cam(cam_name_0) + 'infrared']['transition']
        R2 = cache['extrinsics'][get_cam(cam_name_1) + 'infrared']['rotation']
        T2 = cache['extrinsics'][get_cam(cam_name_1) + 'infrared']['transition']
        R3 = cache['extrinsics'][cam0 + 'infrared']['rotation']
        T3 = cache['extrinsics'][cam0 + 'infrared']['transition']

    color0 = cv2.imread(img_color_0)
    color0 = cv2.resize(color0, (640, 576))
    color0 = o3d.geometry.Image((color0).astype(np.uint8))
    depth0 = o3d.io.read_image(img_depth_0)

    rgbd0 = o3d.geometry.RGBDImage.create_from_color_and_depth(color0, depth0)

    intrinsic0 = o3d.camera.PinholeCameraIntrinsic(640, 576, mtx0[0, 0], mtx0[1, 1], mtx0[0, 2], mtx0[1, 2])

    if cam == cam_name_0:
        extrinsic0 = np.identity(4) # Extrinsic matrix
        r = np.array([[1, 0, 0], # Rotation matrix
                    [0, 1, 0],
                    [0, 0, 1]])
        t = np.array([0, 0, 0]) # Translation vector
        extrinsic0[:3, :3] = r
        extrinsic0[:3, 3] = t.reshape(3)
    elif cam == cam_name_1:
        extrinsic0 = np.identity(4) # Extrinsic matrix
        extrinsic0[:3, :3] = R
        extrinsic0[:3, 3] = T.reshape(3) / 1000
    elif cam == cam_name_2:
        extrinsic0 = np.identity(4) # Extrinsic matrix
        R2_com = np.dot(R2, R)
        T2_com = (np.dot(R2, T).reshape(3, 1) + T2).reshape(3,)
        extrinsic0[:3, :3] = R2_com
        extrinsic0[:3, 3] = (T2_com / 1000) + np.array([0, 0, 0])
    elif cam == cam_name_3:
        extrinsic0 = np.identity(4) # Extrinsic matrix
        R2_com = np.dot(R2, R)
        T2_com = (np.dot(R2, T).reshape(3, 1) + T2).reshape(3,)
        R3_com = np.dot(R3, R2_com)
        T3_com = (np.dot(R3, T2_com).reshape(3, 1) + T3).reshape(3,)
        extrinsic0[:3, :3] = R3_com
        extrinsic0[:3, 3] = (T3_com / 1000) + np.array([0, 0, 0])

    pcd0 = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd0, intrinsic0, extrinsic0)

    if cam == cam_name_0:
        pass
    elif cam == cam_name_1:
        transition = [0, 0, 0]
        angle = [0, 0, 0]
        rm = rotation_matrix_from_euler_angles(np.deg2rad(angle[0]),
                                            np.deg2rad(angle[1]),
                                            np.deg2rad(angle[2]))
        pcd0.rotate(rm)
        pcd0.translate(transition)
    elif cam == cam_name_2:
        transition = [0, 0, 0]
        angle = [0, 0, 0]
        rm = rotation_matrix_from_euler_angles(np.deg2rad(angle[0]),
                                            np.deg2rad(angle[1]),
                                            np.deg2rad(angle[2]))
        pcd0.rotate(rm)
        pcd0.translate(transition)

    return pcd0


cam_name_0 = '1_5'
cam_name_1 = '1_4'
cam_name_2 = '3_4'
cam_name_3 = '3_5'
if __name__ == "__main__":
    cache = diskcache.Cache('../calibration/cache')
    
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    # vis.get_render_option().mesh_show_back_face = True
    # material = o3d.visualization.rendering.Material()
    # material.shader = "defaultLit"

    geometry = o3d.geometry.PointCloud()
    geometry_mesh = o3d.geometry.TriangleMesh()

    for i in range(1000):
        pcd0 = get_pcd(cam_name_0, i, cache)
        pcd1 = get_pcd(cam_name_1, i, cache)
        pcd2 = get_pcd(cam_name_2, i, cache)
        pcd3 = get_pcd(cam_name_3, i, cache)

        # pcd = pcd0
        pcd = pcd0 + pcd1 + pcd2 + pcd3
        # pcd = pcd0 + pcd3
        # pcd.estimate_normals()

        # alpha = .02
        # voxel_down_pcd = pcd.voxel_down_sample(voxel_size=0.02)
        # mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(voxel_down_pcd, alpha)
        # mesh = mesh.filter_smooth_simple(number_of_iterations=1)
        # mesh.compute_vertex_normals()
        # mesh.compute_triangle_normals()

        geometry.points = pcd.points
        # geometry_mesh.vertices = mesh.vertices
        # geometry_mesh.triangles = mesh.triangles
        # geometry_mesh.vertex_normals = mesh.vertex_normals
        # geometry_mesh.triangle_normals = mesh.triangle_normals
        if i == 0:
            vis.add_geometry(geometry)
            # vis.add_geometry(geometry_mesh)
            # vis.get_render_option().mesh_show_back_face = True
        else:
            vis.update_geometry(geometry)
            # vis.update_geometry(geometry_mesh)
        vis.poll_events()
        vis.update_renderer()

        print(f"Update {i}: {time.time()}")
        # time.sleep(.05)
