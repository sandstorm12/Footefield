import numpy as np
import open3d as o3d

import os
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

# Util function for loading point clouds|
import numpy as np

# Data structures and functions for rendering
from pytorch3d.structures import Pointclouds
from pytorch3d.vis.plotly_vis import AxisArgs, plot_batch_individually, plot_scene
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVOrthographicCameras, 
    PointsRasterizationSettings,
    PointsRenderer,
    PulsarPointsRenderer,
    PointsRasterizer,
    AlphaCompositor,
    NormWeightedCompositor
)


def show(pointcloud):
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=True)
    vis.get_render_option().background_color = [0.203921569, 0.239215686, 0.274509804]
    vis.get_render_option().show_coordinate_frame = True
    vis.add_geometry(pointcloud)
    vis.run()


if __name__ == '__main__':
    path = './pointclouds/a1_1_0.pcd'

    pcd = o3d.io.read_point_cloud(path)
    # show(pcd)

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    print(f"Setting device: {device}")

    # Load point cloud
    verts = torch.from_numpy(np.asarray(pcd.points)).float().to(device)
    colors = (torch.ones((verts.shape[0], 3)) * 0.5).to(device)

    print(verts.shape, colors.shape)

    point_cloud = Pointclouds(points=[verts], features=[colors])

    # Initialize a camera.
    R, T = look_at_view_transform(40, 0, -10) # dist, elev, azim, degrees
    cameras = FoVOrthographicCameras(device=device, R=R, T=T, znear=0.01)

    # Define the settings for rasterization and shading. Here we set the output image to be of size
    # 512x512. As we are rendering images for visualization purposes only we will set faces_per_pixel=1
    # and blur_radius=0.0. Refer to raster_points.py for explanations of these parameters. 
    raster_settings = PointsRasterizationSettings(
        image_size=512, 
        radius = 0.003,
        points_per_pixel = 10
    )


    # Create a points renderer by compositing points using an alpha compositor (nearer points
    # are weighted more heavily). See [1] for an explanation.
    rasterizer = PointsRasterizer(cameras=cameras, raster_settings=raster_settings)
    renderer = PointsRenderer(
        rasterizer=rasterizer,
        compositor=AlphaCompositor()
    )

    images = renderer(point_cloud)
    plt.figure(figsize=(10, 10))
    plt.imshow(images[0, ..., :3].cpu().numpy())
    plt.axis("off")
    plt.show()