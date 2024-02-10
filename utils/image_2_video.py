import os
import cv2
import glob

import data_loader

from tqdm import tqdm


OUTPUT_DIR = './videos'


def load_image_paths(dir):
    file_paths = sorted(glob.glob(os.path.join(dir, '*')))

    return file_paths


def images_2_video(image_paths, video_name, fps, width, height):
    if not os.path.exists(OUTPUT_DIR):
        print(f'Output dir does not exist, creating: {OUTPUT_DIR}')
        os.makedirs(OUTPUT_DIR)

    video_path = os.path.join(OUTPUT_DIR, video_name + '.mp4')
    print(f'Writing {len(image_paths)} images to {video_path}')

    writer = cv2.VideoWriter(
        video_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (data_loader.IMAGE_RGB_WIDTH, data_loader.IMAGE_RGB_HEIGHT))
    
    for image_path in tqdm(image_paths):
        frame = cv2.imread(image_path)
        writer.write(frame)

    writer.release()


if __name__ == "__main__":
    for experiment in data_loader.EXPERIMENTS.keys():
        exp_dirs = data_loader.EXPERIMENTS[experiment].values()
        for exp_dir in exp_dirs:
            print(exp_dir)

            exp_dir_rgb = os.path.join(exp_dir, 'color')
            image_paths = load_image_paths(exp_dir_rgb)

            video_name = '_'.join(exp_dir.split('/')[-2:])
            
            images_2_video(
                image_paths, video_name, 8,
                data_loader.IMAGE_RGB_WIDTH, data_loader.IMAGE_RGB_HEIGHT)


