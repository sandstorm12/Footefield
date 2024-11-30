import cv2
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
        'pytorch/vision:v0.10.0',
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


def _segment_video(model, preprocess, configs):
    path_in = configs['path_in']
    path_out = configs['path_out']

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

    # TODO: Add batching. Update: didn't work as expected
    for idx_frame in tqdm(range(length)):
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        input_tensor = preprocess(frame_rgb)
        input_batch = input_tensor.unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(input_batch)['out'][0]
        output_predictions = output.argmax(0)

        output_predictions[output_predictions == 15] = 255
        output_predictions[output_predictions != 255] = 0

        mask = output_predictions.detach().cpu().numpy().astype(np.uint8)
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

        if configs['see_through']:
            mask[mask == 255] = frame[mask == 255]

        writer.write(mask)


if __name__ == "__main__":
    args = _get_arguments()
    configs = _load_configs(args.config)

    print(f"Config loaded: {configs}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model, preprocess = _load_model('cuda')

    _segment_video(model, preprocess, configs)
