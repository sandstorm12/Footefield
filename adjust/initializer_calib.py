import os
import cv2
import yaml
import shutil
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


def _make_check_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)


def _make_dirs(configs):
    dir_videos_org = os.path.join(configs['output_dir'], "videos_org")
    _make_check_exists(dir_videos_org)

    dir_segments = os.path.join(configs['output_dir'], "segments")
    _make_check_exists(dir_segments)

    dir_artifacts = os.path.join(configs['output_dir'], "artifacts")
    _make_check_exists(dir_artifacts)

    dir_configs = os.path.join(configs['output_dir'], "configs")
    _make_check_exists(dir_configs)

    dir_sh = os.path.join(configs['output_dir'], "sh")
    _make_check_exists(dir_sh)

    return dir_videos_org, dir_segments, dir_artifacts, dir_configs, dir_sh


def _copy_original_videos(dir_videos_org, configs):
    for video in configs['videos']:
        file_name = os.path.split(video['path'])[1]
        file_name = video['name'] + '_' + file_name
        
        src = video['path']
        dst = os.path.join(dir_videos_org, file_name)
        
        shutil.copyfile(src, dst)


def _segment_videos(dir_segments, configs):
    for video in tqdm(configs['videos']):
        cap = cv2.VideoCapture(video['path'])
        
        path_segments_cam = os.path.join(dir_segments, video['name'])
        _make_check_exists(path_segments_cam)
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, video['start'])
        length = video['length']
        for segment in tqdm(range(length // configs['segment_size'])):
            segment_path = os.path.join(path_segments_cam, f"{segment}.mp4")
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


def _store_config(config, path):
    with open(path, 'w') as yaml_file:
        yaml.dump(config, yaml_file)


def _store_sh(sh, path):
    with open(path, 'w') as sh_file:
        sh_file.write(sh)


def _generate_detect_chessboard_rgb(
        dir_segments, dir_artifacts, dir_configs, configs):
    configs_default = _load_configs(
        configs['defaults']['detect_chessboard_rgb'])
    
    # Output dir
    configs_default['output_dir'] = os.path.join(
        dir_artifacts, configs_default['output_dir'])

    # Calibration videos
    calibration_videos = {}
    
    cameras = os.listdir(dir_segments)
    for camera in cameras:
        calibration_videos[camera] = {
            "path": os.path.join(dir_segments, camera, "0.mp4") 
        }
    configs_default['calibration_videos'] = calibration_videos

    _store_config(configs_default,
                  os.path.join(
                      dir_configs,
                      "detect_chessboard_rgb.yml"))
    

def _generate_calc_intrinsic_rgb(
        dir_artifacts, dir_configs, configs):
    configs_default = _load_configs(configs['defaults']['calc_intrinsic_rgb'])
    
    # Chessboards
    configs_default['chessboards'] = os.path.join(
        dir_artifacts, configs_default['chessboards'])
    
    # Output dir
    configs_default['output_dir'] = os.path.join(
        dir_artifacts, configs_default['output_dir'])

    _store_config(configs_default,
                  os.path.join(
                      dir_configs,
                      "calc_intrinsic_rgb.yml"))
    

def _generate_calc_extrinsic_rgb(
        dir_segments, dir_artifacts, dir_configs, configs):
    configs_default = _load_configs(configs['defaults']['calc_extrinsic_rgb'])
    
    # Intrinsic
    configs_default['intrinsics'] = os.path.join(
        dir_artifacts, configs_default['intrinsics'])
    
    # Chessboards
    configs_default['chessboards'] = os.path.join(
        dir_artifacts, configs_default['chessboards'])
    
    # Output dir
    configs_default['output_dir'] = os.path.join(
        dir_artifacts, configs_default['output_dir'])
    
    cameras = os.listdir(dir_segments)
    configs_default['cameras'] = cameras

    _store_config(configs_default,
                  os.path.join(
                      dir_configs,
                      "calc_extrinsic_rgb.yml"))


def _generate_configs(dir_segments, dir_artifacts, dir_configs, configs):
    _generate_detect_chessboard_rgb(
        dir_segments, dir_artifacts, dir_configs, configs)
    
    _generate_calc_intrinsic_rgb(
        dir_artifacts, dir_configs, configs)
    
    _generate_calc_extrinsic_rgb(
        dir_segments, dir_artifacts, dir_configs, configs)
    

def _generate_sh(dir_configs, dir_sh, configs):
    sh_default = _load_configs(configs['defaults']['sh'])

    pattern = sh_default['pattern']
    pattern = pattern.format(
        sh_default['detect_chessboard_rgb'],
        os.path.join(dir_configs, "detect_chessboard_rgb.yml"),
        sh_default['calc_intrinsic_rgb'],
        os.path.join(dir_configs, "calc_intrinsic_rgb.yml"),
        sh_default['calc_extrinsic_rgb'],
        os.path.join(dir_configs, "calc_extrinsic_rgb.yml"),
    )

    _store_sh(pattern, os.path.join(dir_sh, "sh.sh"))


if __name__ == "__main__":
    args = _get_arguments()
    configs = _load_configs(args.config)

    print(f"Config loaded: {configs}")

    dir_videos_org, dir_segments, dir_artifacts, dir_configs, dir_sh = \
        _make_dirs(configs)

    _copy_original_videos(dir_videos_org, configs)

    _segment_videos(dir_segments, configs)

    _generate_configs(dir_segments, dir_artifacts, dir_configs, configs)

    _generate_sh(dir_configs, dir_sh, configs)
