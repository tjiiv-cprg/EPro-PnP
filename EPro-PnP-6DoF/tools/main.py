"""
This file is modified from
https://github.com/LZGMatrix/CDPN_ICCV2019_ZhigangLi

@author: Zhigang Li
@license: (C) Copyright.
@contact: aaalizhigang@163.com
@software: Pose6D
@file: main.py
@time: 18-10-24 下午10:24
@desc:
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import random
import time
import datetime
import pprint
import torch
import torch.utils.data
import matplotlib
matplotlib.use('Agg')
import cv2
# cv2.setNumThreads(0)
# pytorch issue 1355: possible deadlock in dataloader. OpenCL may be enabled by default in OpenCV3;
# disable it because it's not thread safe and causes unwanted GPU memory allocations.
cv2.ocl.setUseOpenCL(False)
import _init_paths
import ref
import utils.fancy_logger as logger
from utils.io import load_ply_vtx
from model import build_model, save_model
from datasets.lm import LM
from train import train
from test import test
from config import config
from tqdm import tqdm 

def main():
    cfg = config().parse()
    network, optimizer = build_model(cfg)
    criterions = {'L1': torch.nn.L1Loss(),
                  'L2': torch.nn.MSELoss()}

    if cfg.pytorch.gpu > -1:
        logger.info('Using GPU{}'.format(cfg.pytorch.gpu))
        network = network.cuda(cfg.pytorch.gpu)
        for k in criterions.keys():
            criterions[k] = criterions[k].cuda(cfg.pytorch.gpu)

    def _worker_init_fn():
        torch_seed = torch.initial_seed()
        np_seed = torch_seed // 2**32 - 1
        random.seed(torch_seed)
        np.random.seed(np_seed)

    test_loader = torch.utils.data.DataLoader(
        LM(cfg, 'test'),
        batch_size=cfg.train.train_batch_size if 'fast' in cfg.test.test_mode else 1,
        shuffle=False,
        num_workers=int(cfg.pytorch.threads_num),
        worker_init_fn=_worker_init_fn()
    )

    obj_vtx = {}
    logger.info('load 3d object models...')
    for obj in tqdm(cfg.dataset.classes):
        obj_vtx[obj] = load_ply_vtx(os.path.join(ref.lm_model_dir, '{}/{}.ply'.format(obj, obj)))
    obj_info = LM.load_lm_model_info(ref.lm_model_info_pth)

    if cfg.pytorch.test:
        _, preds = test(0, cfg, test_loader, network, obj_vtx, obj_info, criterions)
        if preds is not None:
            torch.save({'cfg': pprint.pformat(cfg), 'preds': preds}, os.path.join(cfg.pytorch.save_path, 'preds.pth'))
        return

    train_loader = torch.utils.data.DataLoader(
        LM(cfg, 'train'),
        batch_size=cfg.train.train_batch_size,
        shuffle=True,
        num_workers=int(cfg.pytorch.threads_num),
        worker_init_fn=_worker_init_fn()
    )

    for epoch in range(cfg.train.begin_epoch, cfg.train.end_epoch + 1):
        mark = epoch if (cfg.pytorch.save_mode == 'all') else 'last'
        log_dict_train, _ = train(epoch, cfg, train_loader, network, obj_info, criterions, optimizer)
        for k, v in log_dict_train.items():
            logger.info('{} {:8f} | '.format(k, v))
        if epoch % cfg.train.test_interval == 0:
            save_model(os.path.join(cfg.pytorch.save_path, 'model_{}.checkpoint'.format(mark)), network)  # optimizer
            log_dict_val, preds = test(epoch, cfg, test_loader, network, obj_vtx, obj_info, criterions)
        logger.info('\n')
        if epoch in cfg.train.lr_epoch_step:
            if optimizer is not None:
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= cfg.train.lr_factor
                    logger.info("drop lr to {}".format(param_group['lr']))

    torch.save(network.cpu(), os.path.join(cfg.pytorch.save_path, 'model_cpu.pth'))

if __name__ == '__main__':
    main()
