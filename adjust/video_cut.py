import os
import cv2
import yaml
import argparse

from tqdm import tqdm


def _get_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-c', '--config',
        help='Path to the config file',
        type=str,
        required=True,
    )

    args = parser.parse_args()

    return args


def _load_configs(path):
    with open(path, 'r') as yaml_file:
        configs = yaml.safe_load(yaml_file)

    return configs


def _process_folder(configs):
    for video in tqdm(configs['videos']):
        cap = cv2.VideoCapture(video['path'])
        
        segments_dir = os.path.join(configs['output_dir'], video['name'])
        if not os.path.exists(segments_dir):
            os.makedirs(segments_dir)
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, video['start'])
        length = video['length']
        for segment in tqdm(range(length // configs['segment_size'])):
            segment_path = os.path.join(segments_dir, f"{segment}.mp4")
            writer = cv2.VideoWriter(
                segment_path,
                cv2.VideoWriter_fourcc(*'mp4v'),
                5,
                (1920, 1080))

            for _ in range(configs['segment_size']):
                ret, frame = cap.read()

                if not ret:
                    raise Exception(
                        "Failed to read frame in cam {} at segment {}".format(
                            video['name'], segment
                        ))

                writer.write(frame)


def _create_output_dir(configs):
    if not os.path.exists(configs['output_dir']):
        os.makedirs(configs['output_dir'])


if __name__ == "__main__":
    args = _get_arguments()
    configs = _load_configs(args.config)

    print(f"Config loaded: {configs}")

    _create_output_dir(configs)

    _process_folder(configs)
