"""
Copyright (C) 2010-2022 Alibaba Group Holding Limited.
This file is modified from
https://github.com/tjiiv-cprg/MonoRUn
"""

import os
if 'OMP_NUM_THREADS' not in os.environ:
    os.environ['OMP_NUM_THREADS'] = '8'

import sys
import argparse
import socket
from contextlib import closing


def parse_args():
    parser = argparse.ArgumentParser(description='Test (and evaluate)')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        help='evaluation metrics, use "--eval nds" for nuScenes evaluation')
    parser.add_argument(
        '--format-only',
        action='store_true',
        help='format the output results without perform evaluation. It is '
        'useful when you want to format the result to a specific format and '
        'submit it to the test server')
    parser.add_argument(
        '--eval-options',
        type=str,
        nargs='+',
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function')
    parser.add_argument(
        '--val-set',
        action='store_true',
        help='whether to test validation set instead of test set')
    parser.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='ids of gpus to use')
    parser.add_argument(
        '--show-dir', help='directory where visualizations will be saved')
    parser.add_argument(
        '--show-score-thr', type=float, default=0.3,
        help='bbox score threshold for visualization')
    parser.add_argument(
        '--show-views',
        type=str,
        nargs='+',
        help='views to show, e.g., "--show-views 2d 3d bev mc score pts orient" '
             'to fully visulize EProPnPDet')
    parser.add_argument(
        '--step',
        type=int,
        default=1,
        help='sample step, useful when you want to quickly visualize some samples'
             ' of the test set')
    parser.add_argument(
        '--timer',
        action='store_true',
        help='whether to print detailed inference time (this may affect inference speed '
             'due to synchronization).')
    args = parser.parse_args()

    if args.eval:
        assert args.step is None or args.step == 1, '"--step" argument incampatible with "--eval"'
    return args


def args_to_str(args):
    argv = [args.config, args.checkpoint, '--fuse-conv-bn']
    if args.eval is not None:
        argv += ['--eval'] + args.eval
    if args.format_only:
        argv.append('--format-only')
    if args.eval_options is not None:
        argv += ['--eval-options'] + args.eval_options
    if args.val_set:
        argv.append('--val-set')
    if args.show_dir is not None:
        argv += ['--show-dir', args.show_dir]
    if args.show_score_thr is not None:
        argv += ['--show-score-thr', str(args.show_score_thr)]
    if args.show_views is not None:
        argv += ['--show-views'] + args.show_views
    if args.step is not None:
        argv += ['--step', str(args.step)]
    if args.timer:
        argv.append('--timer')
    return argv


def main():
    args = parse_args()
    if args.gpu_ids is not None:
        gpu_ids = args.gpu_ids
    elif 'CUDA_VISIBLE_DEVICES' in os.environ:
        gpu_ids = [int(i) for i in os.environ['CUDA_VISIBLE_DEVICES'].split(',')]
    else:
        gpu_ids = [0]
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(i) for i in gpu_ids])
    if len(gpu_ids) == 1:
        import tools.test
        sys.argv = [''] + args_to_str(args)
        tools.test.main()
    else:
        assert not args.timer, '"--timer" unsupported for multi-gpu test'
        assert not args.show_dir, '"--show-dir" unsupported for multi-gpu test'
        from torch.distributed import launch
        for port in range(29500, 65536):
            with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
                res = sock.connect_ex(('localhost', port))
                if res != 0:
                    break
        sys.argv = ['',
                    '--nproc_per_node={}'.format(len(gpu_ids)),
                    '--master_port={}'.format(port),
                    './tools/test.py'
                    ] + args_to_str(args) + ['--launcher', 'pytorch']
        launch.main()


if __name__ == '__main__':
    main()
