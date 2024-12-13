import os
import cv2
import time
import yaml
import torch
import argparse
import numpy as np

from tqdm import tqdm

from torchvision import transforms


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


def _load_model(device):
    model = torch.hub.load(
        'pytorch/vision:v0.20.0',
        # 'deeplabv3_resnet50',
        'deeplabv3_resnet101',
        # 'deeplabv3_mobilenet_v3_large',
        pretrained=True).to(device).eval()

    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    return model, preprocess


def _segment_video(model, preprocess, path_in, path_out, offset, device, configs):
    cap = cv2.VideoCapture(path_in)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = cv2.VideoWriter(
        path_out,
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (width, height),
    )

    length = length - offset
    length = min(length, configs['exp_length'])
    for i in range(offset):
        cap.grab()

    frames = [cap.read()[1] for _ in range(length)]

    batch_size = 1
    # TODO: Add batching. Update: didn't work as expected
    for idx_frame in tqdm(range(length // batch_size)):
        batch = torch.stack([
            preprocess(cv2.resize(cv2.cvtColor(
                frame, cv2.COLOR_BGR2RGB), (width, height)))
            for frame in frames[idx_frame * batch_size: (idx_frame + 1) * batch_size]])
        batch = batch.to(device)

        with torch.no_grad():
            output = model(batch)['out']

        for idx in range(batch_size):
            output_predictions = output[idx].argmax(0)

            output_predictions[output_predictions == 15] = 255
            output_predictions[output_predictions != 255] = 0

            mask = output_predictions.detach().cpu().numpy().astype(np.uint8)
            mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            mask = cv2.resize(mask, (width, height))

            # cv2.imshow("mask", mask)
            # if cv2.waitKey(0) == ord('q'):
            #     break

            if configs['see_through']:
                mask[mask == 255] = frames[idx_frame * batch_size + idx][mask == 255]

            writer.write(mask)


if __name__ == "__main__":
    args = _get_arguments()
    configs = _load_configs(args.config)

    print(f"Config loaded: {configs}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = 'cpu'

    print("Device", device)

    model, preprocess = _load_model(device)

    for video in configs['views']:
        path_in = configs['views'][video]['path_in']
        offset = configs['views'][video]['offset']
        path_out = os.path.join(configs['output_dir'],
                                configs['views'][video]['name_out'])
        _segment_video(model, preprocess, path_in, path_out, offset,
                       device, configs)
