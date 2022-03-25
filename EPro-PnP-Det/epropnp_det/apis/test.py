"""
Copyright (C) 2010-2022 Alibaba Group Holding Limited.
This file is modified from
https://github.com/open-mmlab/mmdetection
"""

import mmcv
import torch

from mmdet.core import encode_mask_results
from .inference import show_result
from ..utils.timer import default_timers

default_timers.add_timer('total time (incl. data transfer)')


def single_gpu_test(model,
                    data_loader,
                    show=False,
                    out_dir=None,
                    show_score_thr=0.3,
                    enable_timer=False,
                    **kwargs):
    if enable_timer:
        default_timers.enable_all()
    model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        with torch.no_grad(), default_timers['total time (incl. data transfer)']:
            result = model(return_loss=False, rescale=True, **data)

        if show or out_dir:
            show_result(model.module, result, data,
                        show=show, out_dir=out_dir,
                        show_score_thr=show_score_thr,
                        out_dir_level=1, **kwargs)

        # encode mask results
        if isinstance(result[0], tuple):
            result = [(bbox_results, encode_mask_results(mask_results))
                      for bbox_results, mask_results in result]
        results.extend(result)

        batch_size = len(result)
        for _ in range(batch_size):
            prog_bar.update()

    print('')
    for timer in default_timers.values():
        timer.print_time()
    return results
