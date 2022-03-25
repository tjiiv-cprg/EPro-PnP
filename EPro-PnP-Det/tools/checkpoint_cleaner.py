"""
Copyright (C) 2010-2022 Alibaba Group Holding Limited.
"""

import argparse
import os
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description='Remove all checkpoints except the latests')
    parser.add_argument('workdir', help='directory of checkpoints')
    return parser.parse_args()


def main():
    args = parse_args()
    workdir = args.workdir
    print('This will remove all the non-latest checkpoints in {}'.format(Path(workdir).resolve()))
    answer = None
    while answer not in ('y', 'n'):
        answer = input('continue? [y/n]')
        if answer == 'n':
            exit()
    for dirpath, dirnames, filenames in os.walk(workdir):
        pth_filenames = [f for f in filenames if f.endswith('.pth')]
        if 'latest.pth' in pth_filenames:
            latest_path = os.path.join(dirpath, 'latest.pth')
            if os.path.islink(latest_path):
                latest_path_tgt = Path(latest_path).resolve()
                os.remove(latest_path)
                os.rename(latest_path_tgt, latest_path)
                pth_filenames.remove(latest_path_tgt.name)
            pth_filenames.remove('latest.pth')
        for f in pth_filenames:
            path = os.path.join(dirpath, f)
            os.remove(path)
            print('Removed {}'.format(path))


if __name__ == '__main__':
    main()
