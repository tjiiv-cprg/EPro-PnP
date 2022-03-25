"""
Copyright (C) 2010-2022 Alibaba Group Holding Limited.
This file is modified from
https://github.com/tjiiv-cprg/MonoRUn
"""

import os
if 'OMP_NUM_THREADS' not in os.environ:
    os.environ['OMP_NUM_THREADS'] = '8'

import argparse
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description='Infer from images in a directory')
    parser.add_argument('image_dir', help='directory of input images')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--intrinsic', help='camera intrinsic matrix in .csv format',
                        default='demo/nus_cam_front.csv')
    parser.add_argument(
        '--show-dir', 
        help='directory where visualizations will be saved (default: $IMAGE_DIR/viz)')
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

    from mmcv.utils import track_iter_progress
    from mmcv.cnn import fuse_conv_bn
    from epropnp_det.apis import init_detector, inference_detector, show_result

    image_dir = args.image_dir
    assert os.path.isdir(image_dir)
    show_dir = args.show_dir
    if show_dir is None:
        show_dir = os.path.join(image_dir, 'viz')
    os.makedirs(show_dir, exist_ok=True)
    cam_mat = np.loadtxt(args.intrinsic, delimiter=',').astype(np.float32)

    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device='cuda:0')
    model = fuse_conv_bn(model)
    model.test_cfg['debug'] = args.show_views if args.show_views is not None else []

    img_list = []
    for filename in os.listdir(image_dir):
        if os.path.splitext(filename)[1] in ['.jpg', '.jpeg', '.png']:
            img_list.append(filename)
    img_list.sort()
    kwargs = dict(views=args.show_views) if args.show_views is not None else dict()
    for img_filename in track_iter_progress(img_list):
        result, data = inference_detector(
            model, [os.path.join(image_dir, img_filename)], cam_mat)
        show_result(
            model, result, data,
            show=False, out_dir=show_dir, show_score_thr=args.show_score_thr,
            **kwargs)
    return


if __name__ == '__main__':
    main()
