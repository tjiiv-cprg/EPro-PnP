"""
Copyright (C) 2010-2022 Alibaba Group Holding Limited.
This file is modified from
https://github.com/tjiiv-cprg/MonoRUn
"""

import os
if 'OMP_NUM_THREADS' not in os.environ:
    os.environ['OMP_NUM_THREADS'] = '8'

import argparse
from tqdm import tqdm
import numpy as np

from nuscenes.nuscenes import NuScenes
from nuscenes.utils import splits

split_to_version = dict(
    train='v1.0-trainval',
    val='v1.0-trainval',
    test='v1.0-test',
    mini_train='v1.0-mini',
    mini_val='v1.0-mini'
)


def parse_args():
    parser = argparse.ArgumentParser(description='Infer from a sequence of images in the nuScenes dataset')
    parser.add_argument('path', help='nuScenes directory path')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--split',
                        choices=['train', 'val', 'test', 'mini_train', 'mini_val'],
                        default='val',
                        help='data split')
    parser.add_argument(
        '--show-dir', 
        help='directory where visualizations will be saved (default: viz/<scene name>)')
    parser.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='ids of gpus to use')
    parser.add_argument(
        '--show-score-thr', type=float, default=0.3, help='bbox score threshold for visialization')
    parser.add_argument(
        '--show-views',
        type=str,
        nargs='+',
        help='views to show, e.g., "--show-views 2d 3d bev mc score pts orient" '
             'to fully visulize EProPnPDet')
    parser.add_argument(
        '--sweep',
        action='store_true',
        help='whether to use sweeps instead of keyframes')
    parser.add_argument(
        '--cameras',
        type=str,
        nargs='+',
        help='camera types (e.g., CAM_FRONT CAM_BACK)',
        default=[
            'CAM_FRONT',
            'CAM_FRONT_RIGHT',
            'CAM_FRONT_LEFT',
            'CAM_BACK',
            'CAM_BACK_LEFT',
            'CAM_BACK_RIGHT'])
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    if args.gpu_ids is not None:
        gpu_ids = args.gpu_ids
    elif 'CUDA_VISIBLE_DEVICES' in os.environ:
        gpu_ids = [int(i) for i in os.environ['CUDA_VISIBLE_DEVICES'].split(',')]
    else:
        gpu_ids = [0]
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(i) for i in gpu_ids])
    if len(gpu_ids) != 1:
        raise NotImplementedError('multi-gpu inference is not yet supported')

    from mmcv.cnn import fuse_conv_bn
    from epropnp_det.apis import init_detector, inference_detector, show_result

    version = split_to_version[args.split]
    nusc = NuScenes(version=version, dataroot=args.path, verbose=True)
    split_scenes = list(
        filter(lambda x: x['name'] in getattr(splits, args.split), nusc.scene))

    scene_names = []
    for scene in split_scenes:
        name = scene['name']
        description = scene['description']
        scene_names.append(name)
        print(name + '\t' + description)
    print('\nNumber of scenes: {}'.format(len(split_scenes)))
    input_scene = input('\nEnter the name of scene (e.g., "scene-XXXX") to infer, '
                        'or "X" to explore all the scenes: ')
    if input_scene in ['x', 'X']:
        os.makedirs('viz/nus_explore', exist_ok=True)
        for scene in split_scenes:
            sample = nusc.get('sample', scene['first_sample_token'])
            cam_token = sample['data']['CAM_FRONT']
            cam_data = nusc.get('sample_data', cam_token)
            image = os.path.join(args.path, cam_data['filename'])
            dest = 'viz/nus_explore/' + scene['name'] + '.jpg'
            if not os.path.lexists(dest):
                os.symlink(image, dest)
        print('\nThe first images of all scenes have been symlinked into "viz/nus_explore".\n')
        return

    scene = split_scenes[scene_names.index(input_scene)]
    sample = nusc.get('sample', scene['first_sample_token'])
    camera_types = args.cameras

    show_dir = args.show_dir
    if show_dir is None:
        show_dir = 'viz/' + input_scene
    os.makedirs(show_dir, exist_ok=True)

    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device='cuda:0')
    model = fuse_conv_bn(model)
    model.test_cfg['debug'] = args.show_views if args.show_views is not None else []

    kwargs = dict(views=args.show_views) if args.show_views is not None else dict()
    progress_bar = tqdm(total=scene['nbr_samples'])

    cam_data_list = [nusc.get('sample_data', sample['data'][cam]) for cam in camera_types]
    while True:
        images = []
        cam_mats = []
        for cam_data in cam_data_list:
            calibrated_sensor = nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])
            images.append(os.path.join(args.path, cam_data['filename']))
            cam_mats.append(np.array(calibrated_sensor['camera_intrinsic'], dtype=np.float32))
        result, data = inference_detector(
            model, images, cam_mats)
        show_result(
            model, result, data,
            show=False, out_dir=show_dir, show_score_thr=args.show_score_thr,
            out_dir_level=1, **kwargs)
        if cam_data_list[0]['is_key_frame']:
            progress_bar.update()
        if args.sweep:
            if cam_data_list[0]['next']:
                cam_data_list = [nusc.get('sample_data', cam_data['next']) for cam_data in cam_data_list]
                continue
        else:
            if sample['next']:
                sample = nusc.get('sample', sample['next'])
                cam_data_list = [nusc.get('sample_data', sample['data'][cam]) for cam in camera_types]
                continue
        break

    return


if __name__ == '__main__':
    main()
