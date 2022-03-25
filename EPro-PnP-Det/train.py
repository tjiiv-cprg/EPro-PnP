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
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--resume-from', help='the checkpoint file to resume from')
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='whether not to evaluate the checkpoint during training')
    parser.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='ids of gpus to use')
    parser.add_argument('--seed', type=int, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    args = parser.parse_args()
    return args


def args_to_str(args):
    argv = [args.config]
    if args.work_dir is not None:
        argv += ['--work-dir', args.work_dir]
    if args.resume_from is not None:
        argv += ['--resume-from', args.resume_from]
    if args.no_validate:
        argv.append('--no-validate')
    if args.seed is not None:
        argv += ['--seed', str(args.seed)]
    if args.deterministic:
        argv.append('--deterministic')
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
        import tools.train
        sys.argv = [''] + args_to_str(args)
        tools.train.main()
    else:
        from torch.distributed import launch
        for port in range(29500, 65536):
            with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
                res = sock.connect_ex(('localhost', port))
                if res != 0:
                    break
        os.environ['training_script'] = './tools/train.py'
        sys.argv = ['',
                    '--nproc_per_node={}'.format(len(gpu_ids)),
                    '--master_port={}'.format(port),
                    './tools/train.py'
                    ] + args_to_str(args) + ['--launcher', 'pytorch']
        launch.main()


if __name__ == '__main__':
    main()
