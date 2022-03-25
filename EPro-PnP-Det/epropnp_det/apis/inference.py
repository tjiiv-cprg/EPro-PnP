"""
Copyright (C) 2010-2022 Alibaba Group Holding Limited.
This file is modified from
https://github.com/open-mmlab/mmdetection
"""

# Copyright (c) OpenMMLab. All rights reserved.
import warnings
import os.path as osp

import mmcv
import numpy as np
import torch
from mmcv.ops import RoIPool
from mmcv.parallel import collate, scatter
from mmcv.runner import load_checkpoint
from mmcv.image import tensor2imgs

from mmdet.core import get_classes
from mmdet.datasets import replace_ImageToTensor
from mmdet.datasets.pipelines import Compose
from mmdet.models import build_detector


def show_result(model, result, data,
                show=False, out_dir=None, show_score_thr=0.3, ori_img=None,
                out_dir_level=0, **kwargs):
    if isinstance(data['img'][0], torch.Tensor):
        img_tensor = data['img'][0]
        img_metas = data['img_metas'][0]
        cam_intrinsic = data['cam_intrinsic'][0]
    else:
        img_tensor = data['img'][0].data[0]
        img_metas = data['img_metas'][0].data[0]
        cam_intrinsic = data['cam_intrinsic'][0].data[0]
    imgs = tensor2imgs(img_tensor, **img_metas[0]['img_norm_cfg'])
    assert len(imgs) == len(img_metas)

    gt_bboxes_3d = data.get('gt_bboxes_3d', None)
    gt_labels = data.get('gt_labels', None)
    if gt_bboxes_3d is not None and gt_labels is not None:
        gt_bboxes_3d = [gt_bboxes_3d_single.cpu().numpy()
                        for gt_bboxes_3d_single in gt_bboxes_3d[0].data[0]]
        gt_labels = [gt_labels_single.cpu().numpy()
                     for gt_labels_single in gt_labels[0].data[0]]
    else:
        gt_bboxes_3d = [None] * len(imgs)
        gt_labels = [None] * len(imgs)

    for j, (img, img_meta) in enumerate(zip(imgs, img_metas)):
        h, w, _ = img_meta['img_shape']
        img_show = img[:h, :w, :]

        if ori_img is not None:
            ori_img_ = ori_img
        elif img_meta['img_shape'] == img_meta['ori_shape']:
            ori_img_ = img_show
        else:
            ori_img_ = img_meta['ori_filename']

        if out_dir:
            components = osp.normpath(img_meta['ori_filename']).split(osp.sep)
            out_file = osp.join(out_dir,
                                osp.sep.join(components[-1 - out_dir_level:]))

        else:
            out_file = None

        model.show_result(
            img_show,
            result[j],
            show=show,
            ori_img=ori_img_,
            cam_intrinsic=cam_intrinsic[j].cpu().numpy(),
            gt_bboxes_3d=gt_bboxes_3d[j],
            gt_labels=gt_labels[j],
            out_file=out_file,
            score_thr=show_score_thr,
            **kwargs)


def init_detector(config, checkpoint=None, device='cuda:0', cfg_options=None):
    if isinstance(config, str):
        config = mmcv.Config.fromfile(config)
    elif not isinstance(config, mmcv.Config):
        raise TypeError('config must be a filename or Config object, '
                        f'but got {type(config)}')
    if cfg_options is not None:
        config.merge_from_dict(cfg_options)
    config.model.pretrained = None
    config.model.train_cfg = None
    model = build_detector(config.model, test_cfg=config.get('test_cfg'))
    if checkpoint is not None:
        checkpoint = load_checkpoint(model, checkpoint, map_location='cpu')
        if 'CLASSES' in checkpoint.get('meta', {}):
            model.CLASSES = checkpoint['meta']['CLASSES']
        else:
            warnings.simplefilter('once')
            warnings.warn('Class names are not saved in the checkpoint\'s '
                          'meta data, use COCO classes by default.')
            model.CLASSES = get_classes('coco')
        if hasattr(model, 'CLS_ORIENTATION'):
            if 'CLS_ORIENTATION' in checkpoint['meta']:
                model.CLS_ORIENTATION = checkpoint['meta']['CLS_ORIENTATION']
            else:
                warnings.simplefilter('once')
                warnings.warn('Class orientation types are not saved in the checkpoint\'s '
                              'meta data, use nuScenes types by default.')
                from ..datasets import NuScenes3DDataset
                model.CLS_ORIENTATION = NuScenes3DDataset.CLS_ORIENTATION

    model.cfg = config  # save the config in the model for convenience
    model.to(device)
    model.eval()
    return model


def inference_detector(model: object, imgs: object, cam_mat: object = None) -> object:
    cfg = model.cfg
    device = next(model.parameters()).device  # model device

    cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)
    test_pipeline = Compose(cfg.data.test.pipeline)

    datas = []
    for i, img in enumerate(imgs):
        # prepare data
        if isinstance(img, np.ndarray):
            # directly add img
            data = dict(img=img)
        else:
            # add information into dict
            data = dict(img_info=dict(filename=img), img_prefix=None)
        if cam_mat is not None:
            if isinstance(cam_mat, list):
                cam_mat_single = cam_mat[i]
            elif isinstance(cam_mat, np.ndarray):
                cam_mat_single = cam_mat
            else:
                raise TypeError('unknown cam_mat type')
            if 'img_info' in data:
                data['img_info'].update(cam_intrinsic=cam_mat_single)
            else:
                data['img_info'] = dict(cam_intrinsic=cam_mat_single)
        # build the data pipeline
        data = test_pipeline(data)
        datas.append(data)

    data = collate(datas, samples_per_gpu=len(imgs))
    # just get the actual data from DataContainer
    for key, value in data.items():
        data[key] = [container.data[0] for container in data[key]]
    if next(model.parameters()).is_cuda:
        # scatter to specified GPU
        data = scatter(data, [device])[0]
    else:
        for m in model.modules():
            assert not isinstance(
                m, RoIPool
            ), 'CPU inference with RoIPool is not supported currently.'

    # forward the model
    with torch.no_grad():
        results = model(return_loss=False, rescale=True, **data)

    return results, data
