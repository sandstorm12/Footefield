import sys
sys.path.append('../')

import cv2
from utils import data_loader


img_path = '/home/hamid/Documents/footefield/data/AzureKinectRecord_0729/azure_kinect1_4_calib_snap/color00000.jpg'


img_rgb = cv2.imread(img_path)


def downsample_keep_aspect_ratio(img, size):
    aspect_ratio_org = img.shape[0] / img.shape[1]
    aspect_ratio_new = size[1] / size[0]

    if aspect_ratio_new > aspect_ratio_org:
        width_resize = int(img.shape[1] * (size[1] / img.shape[0]))
        height_resize = size[1]
        center = width_resize // 2

        img_resize = cv2.resize(img, (width_resize, height_resize))
        img_resize = img_resize[
            :, center - size[0] // 2:center + size[0] // 2]
    else:
        width_resize = size[0]
        height_resize = int(img.shape[0] * (size[0] / img.shape[1]))
        center = height_resize // 2

        img_resize = cv2.resize(img, (width_resize, height_resize))
        img_resize = img_resize[
            center - size[0] // 2:center + size[0] // 2, :]

    return img_resize


img_resize = downsample_keep_aspect_ratio(
    img_rgb,
    (
        data_loader.IMAGE_INFRARED_WIDTH,
        data_loader.IMAGE_INFRARED_HEIGHT
    )
)

print(img_resize.shape)

cv2.imshow('frame', img_resize)
cv2.waitKey(0)
