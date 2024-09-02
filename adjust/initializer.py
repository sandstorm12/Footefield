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

    dir_videos_test = os.path.join(configs['output_dir'], "videos_test")
    _make_check_exists(dir_videos_test)

    dir_artifacts = os.path.join(configs['output_dir'], "artifacts")
    _make_check_exists(dir_artifacts)

    dir_configs = os.path.join(configs['output_dir'], "configs")
    _make_check_exists(dir_configs)

    dir_sh = os.path.join(configs['output_dir'], "sh")
    _make_check_exists(dir_sh)

    return dir_videos_org, dir_segments, dir_videos_test, \
        dir_artifacts, dir_configs, dir_sh


def _copy_original_videos(dir_videos_org, configs):
    for video in configs['videos']:
        file_name = os.path.split(video['path'])[1]
        
        src = video['path']
        dst = os.path.join(dir_videos_org, file_name)
        
        shutil.copyfile(src, dst)


def _segment_videos(dir_segments, configs):
    num_segments = []
    for video in tqdm(configs['videos']):
        cap = cv2.VideoCapture(video['path'])
        
        path_segments_cam = os.path.join(dir_segments, video['name'])
        _make_check_exists(path_segments_cam)
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, video['start'])
        length = video['length']
        num_segments.append(length // configs['segment_size'])
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

    return sum(num_segments) // len(num_segments)


def _store_config(config, path):
    with open(path, 'w') as yaml_file:
        yaml.dump(config, yaml_file)


def _store_sh(sh, path):
    with open(path, 'w') as sh_file:
        sh_file.write(sh)


def _generate_skeleton_detection(dir_segments, dir_artifacts, dir_configs,
                                 segment, configs):
    configs_default = _load_configs(
        configs['defaults']['skeleton_detection'])
    
    # Output dir
    configs_default['output'] = os.path.join(
        dir_artifacts, configs_default['output'])
    
    # Intrinsics
    configs_default['intrinsics'] = os.path.join(
        configs['calib_data_dir'], configs_default['intrinsics'])

    # Videos
    videos = {}
    cameras = os.listdir(dir_segments)
    for camera in cameras:
        videos[camera] = {
            "path": os.path.join(dir_segments, camera, f"{segment}.mp4") 
        }
    configs_default['videos'] = videos

    _store_config(configs_default,
                  os.path.join(
                    dir_configs,
                    "skeleton_detection.yml"))
    

def _generate_skeleton_triangulation(dir_artifacts, dir_configs, configs):
    configs_default = _load_configs(
        configs['defaults']['skeleton_triangulation'])
    
    # Output dir
    configs_default['skeletons'] = os.path.join(
        dir_artifacts, configs_default['skeletons'])
    
    # Intrinsics
    configs_default['intrinsics'] = os.path.join(
        configs['calib_data_dir'], configs_default['intrinsics'])
    
    # Extrinsics
    configs_default['extrinsics'] = os.path.join(
        configs['calib_data_dir'], configs_default['extrinsics'])
    
    # Output dir
    configs_default['output'] = os.path.join(
        dir_artifacts, configs_default['output'])
    
    # Output dir
    configs_default['output_params'] = os.path.join(
        dir_artifacts, configs_default['output_params'])

    _store_config(configs_default,
                  os.path.join(
                      dir_configs,
                      "skeleton_triangulation.yml"))
    

def _generate_skeleton_bundle_adjustment(dir_artifacts, dir_configs, configs):
    configs_default = _load_configs(
        configs['defaults']['skeleton_bundle_adjustment'])
    
    # Skeleton 2D
    configs_default['skeletons_2d'] = os.path.join(
        dir_artifacts, configs_default['skeletons_2d'])
    
    # Skeleton 3D
    configs_default['skeletons_3d'] = os.path.join(
        dir_artifacts, configs_default['skeletons_3d'])
    
    # Parameters
    configs_default['params'] = os.path.join(
        dir_artifacts, configs_default['params'])
    
    # Output skeleton 3D
    configs_default['output_skeleton_3d'] = os.path.join(
        dir_artifacts, configs_default['output_skeleton_3d'])
    
    # Output params
    configs_default['output_params'] = os.path.join(
        dir_artifacts, configs_default['output_params'])

    _store_config(configs_default,
                  os.path.join(
                      dir_configs,
                      "skeleton_bundle_adjustment.yml"))
    

def _generate_skeleton_3d_normalizer(dir_artifacts, dir_configs, configs):
    configs_default = _load_configs(
        configs['defaults']['skeleton_3d_normalizer'])
    
    # Skeleton 3D
    configs_default['skeletons_3d'] = os.path.join(
        dir_artifacts, configs_default['skeletons_3d'])
    
    # Output
    configs_default['output'] = os.path.join(
        dir_artifacts, configs_default['output'])

    _store_config(configs_default,
                  os.path.join(
                      dir_configs,
                      "skeleton_3d_normalizer.yml"))
    

def _generate_visualizer_skeleton_3d_video_ba(dir_videos_test, dir_artifacts,
                                              dir_configs, segment, configs):
    configs_default = _load_configs(
        configs['defaults']['visualizer_skeleton_3d_video_ba'])
    
    # Output dir
    configs_default['skeletons'] = os.path.join(
        dir_artifacts, configs_default['skeletons'])
    
    # Params
    configs_default['params'] = os.path.join(
        dir_artifacts, configs_default['params'])
    
    # Output dir
    configs_default['output_dir'] = dir_videos_test

    # Videos
    videos = {}
    cameras = os.listdir(dir_segments)
    for camera in cameras:
        videos[camera] = {
            "path": os.path.join(dir_segments, camera, f"{segment}.mp4") 
        }
    configs_default['videos'] = videos

    _store_config(configs_default,
                  os.path.join(
                    dir_configs,
                    "visualizer_skeleton_3d_video_ba.yml"))


def _generate_configs(dir_videos_test, dir_segments, dir_artifacts,
                      dir_configs, segment, configs):
    _generate_skeleton_detection(
        dir_segments, dir_artifacts, dir_configs, segment, configs)
    
    _generate_skeleton_triangulation(
        dir_artifacts, dir_configs, configs
    )

    _generate_skeleton_bundle_adjustment(
        dir_artifacts, dir_configs, configs
    )

    _generate_skeleton_3d_normalizer(
        dir_artifacts, dir_configs, configs
    )

    _generate_visualizer_skeleton_3d_video_ba(
        dir_videos_test, dir_artifacts, dir_configs, segment, configs
    )
    

def _generate_sh(dir_configs, dir_sh, configs):
    sh_default = _load_configs(configs['defaults']['sh'])

    pattern = sh_default['pattern']
    pattern = pattern.format(
        sh_default['skeleton_detection'],
        os.path.join(dir_configs, "skeleton_detection.yml"),
        sh_default['skeleton_triangulation'],
        os.path.join(dir_configs, "skeleton_triangulation.yml"),
        sh_default['skeleton_bundle_adjustment'],
        os.path.join(dir_configs, "skeleton_bundle_adjustment.yml"),
        sh_default['skeleton_3d_normalizer'],
        os.path.join(dir_configs, "skeleton_3d_normalizer.yml"),
        sh_default['visualizer_skeleton_3d_video_ba'],
        os.path.join(dir_configs, "visualizer_skeleton_3d_video_ba.yml"),
    )

    _store_sh(pattern, os.path.join(dir_sh, "sh.sh"))


def _get_dirs_segment(dir_videos_test, dir_artifacts, dir_configs,
                      dir_sh, segment):
    dir_videos_test = os.path.join(dir_videos_test, f"{segment}")
    dir_artifacts_segment = os.path.join(dir_artifacts, f"{segment}")
    dir_configs_segment = os.path.join(dir_configs, f"{segment}")
    dir_sh_segment = os.path.join(dir_sh, f"{segment}")
    
    _make_check_exists(dir_videos_test)
    _make_check_exists(dir_artifacts_segment)
    _make_check_exists(dir_configs_segment)
    _make_check_exists(dir_sh_segment)

    return dir_videos_test, dir_artifacts_segment, \
        dir_configs_segment, dir_sh_segment


if __name__ == "__main__":
    args = _get_arguments()
    configs = _load_configs(args.config)

    print(f"Config loaded: {configs}")

    dir_videos_org, dir_segments, dir_videos_test, \
        dir_artifacts, dir_configs, dir_sh = \
        _make_dirs(configs)

    # print("Copying original videos...")
    # _copy_original_videos(dir_videos_org, configs)

    # print("Segmenting videos...")
    # num_segments = _segment_videos(dir_segments, configs)

    num_segments = 10

    for segment in range(num_segments):
        dir_videos_test_segment, dir_artifacts_segment, \
        dir_configs_segment, dir_sh_segment = \
            _get_dirs_segment(dir_videos_test, dir_artifacts,
                              dir_configs, dir_sh, segment)
        
        print(f"Generating configs... segment {segment}")
        _generate_configs(dir_videos_test_segment, dir_segments,
                          dir_artifacts_segment, dir_configs_segment,
                          segment, configs)

        print(f"Generating sh... segment {segment}")
        _generate_sh(dir_configs_segment, dir_sh_segment, configs)
