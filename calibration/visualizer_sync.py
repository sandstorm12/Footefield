import os
import cv2
import yaml
import argparse
import numpy as np


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


def _play_sync(configs):
    if not os.path.exists(configs['output_dir']):
        os.makedirs(configs['output_dir'])

    writer = cv2.VideoWriter(
        os.path.join(
            configs['output_dir'],
            'visualizer_sync.mp4'
        ),
        cv2.VideoWriter_fourcc(*'mp4v'),
        configs['fps'],
        (1280*2, 720*3),
    )

    caps = []
    for video in configs['videos']:
        path = configs['videos'][video]['path']
        offset = configs['videos'][video]['offset']

        cap = cv2.VideoCapture(path)
        for _ in range(offset):
            cap.grab()

        caps.append(cap)

    frame_agg = np.zeros((720*3, 1280*2, 3), dtype=np.uint8)
    width = 1280
    height = 720
    while True:
        readouts = [cap.read() for cap in caps]
        if not np.alltrue([readouts[i][0] for i in range(len(readouts))]):
            break

        frame_agg[0:height, 0:width] = cv2.resize(readouts[0][1], (width, height))
        frame_agg[0:height, width:width*2] = cv2.resize(readouts[1][1], (width, height))
        frame_agg[height:height*2, 0:width] = cv2.resize(readouts[2][1], (width, height))
        frame_agg[height:height*2, width:width*2] = cv2.resize(readouts[3][1], (width, height))
        frame_agg[height*2:height*3, 0:width] = cv2.resize(readouts[4][1], (width, height))
        frame_agg[height*2:height*3, width:width*2] = cv2.resize(readouts[5][1], (width, height))

        cv2.imshow("sync", frame_agg)
        writer.write(frame_agg)
        if cv2.waitKey(1) == ord('q'):
            break


if __name__ == "__main__":
    args = _get_arguments()
    configs = _load_configs(args.config)

    print(f"Config loaded: {configs}")

    _play_sync(configs)
