import os
import cv2
import glob
import data_loader

from tqdm import tqdm


OUTPUT_DIR = './videos'
DEPTH = 25


def load_image_paths(dir):
    file_paths = sorted(glob.glob(os.path.join(dir, '*')))
    file_paths = file_paths[:DEPTH]

    return file_paths


def images_2_video(mask_paths, image_paths, video_name, fps, width, height):
    if not os.path.exists(OUTPUT_DIR):
        print(f'Output dir does not exist, creating: {OUTPUT_DIR}')
        os.makedirs(OUTPUT_DIR)

    video_path = os.path.join(OUTPUT_DIR, video_name + '.mp4')
    print(f'Writing {len(mask_paths)} images to {video_path}')

    writer = cv2.VideoWriter(
        video_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (data_loader.IMAGE_RGB_WIDTH, data_loader.IMAGE_RGB_HEIGHT))
    
    for mask_path, image_path in zip(mask_paths, image_paths):
        image = cv2.imread(image_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        image[mask < 230] = (0, 0, 0)
        writer.write(image)

    writer.release()


if __name__ == "__main__":
    for experiment in data_loader.EXPERIMENTS:
        exp_dirs = data_loader.EXPERIMENTS[experiment]
        for exp_dir in exp_dirs:
            exp_dir_mask = os.path.join(exp_dirs[exp_dir], 'mask')
            mask_paths = load_image_paths(exp_dir_mask)

            exp_dir_mask = os.path.join(exp_dirs[exp_dir], 'color')
            image_paths = load_image_paths(exp_dir_mask)

            video_name = '_'.join(exp_dir.split('/')[-2:]) + f'_{experiment}'
            
            images_2_video(
                mask_paths, image_paths, video_name, 8,
                data_loader.IMAGE_RGB_WIDTH, data_loader.IMAGE_RGB_HEIGHT)