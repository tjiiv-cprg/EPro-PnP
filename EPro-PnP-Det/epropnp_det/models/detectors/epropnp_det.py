"""
Copyright (C) 2010-2022 Alibaba Group Holding Limited.
"""

import os
import numpy as np
import mmcv
import matplotlib.pyplot as plt

from mmdet.core.visualization import imshow_det_bboxes
from mmdet.models import DETECTORS
from mmdet.models import SingleStageDetector
from ...core import draw_box_3d_pred, show_bev, deformable_point_vis
from ...utils.timer import default_timers

default_timers.add_timer('backbone time')
default_timers.add_timer('full model time')


@DETECTORS.register_module()
class EProPnPDet(SingleStageDetector):

    def __init__(self, *args, cls_orientation=None, **kwargs):
        super(EProPnPDet, self).__init__(*args, **kwargs)
        self.CLS_ORIENTATION = cls_orientation

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      depth=None,
                      **kwargs):
        super(SingleStageDetector, self).forward_train(img, img_metas)
        x = self.extract_feat(img)
        losses = self.bbox_head.forward_train(x, img_metas, gt_bboxes, **kwargs)
        return losses

    def simple_test(self, img, img_metas, rescale=False, **kwargs):
        with default_timers['backbone time']:
            x = self.extract_feat(img)
        return self.bbox_head.simple_test(x, img_metas, **kwargs)

    def aug_test(self, imgs, img_metas, rescale=False, **kwargs):
        with default_timers['backbone time']:
            x = self.extract_feats(imgs)
        return self.bbox_head.aug_test(x, img_metas, **kwargs)

    def forward_test(self, imgs, img_metas, **kwargs):
        """
        Args:
            imgs (List[Tensor]): the outer list indicates test-time
                augmentations and inner Tensor should have a shape NxCxHxW,
                which contains all images in the batch.
            img_metas (List[List[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch.
        """
        with default_timers['full model time']:
            for var, name in [(imgs, 'imgs'), (img_metas, 'img_metas')]:
                if not isinstance(var, list):
                    raise TypeError(f'{name} must be a list, but got {type(var)}')

            num_augs = len(imgs)
            if num_augs != len(img_metas):
                raise ValueError(f'num of augmentations ({len(imgs)}) '
                                 f'!= num of image meta ({len(img_metas)})')

            # NOTE the batched image size information may be useful, e.g.
            # in DETR, this is needed for the construction of masks, which is
            # then used for the transformer_head.
            for img, img_meta in zip(imgs, img_metas):
                batch_size = len(img_meta)
                for img_id in range(batch_size):
                    img_meta[img_id]['batch_input_shape'] = tuple(img.size()[-2:])

            if num_augs == 1:
                return self.simple_test(imgs[0], img_metas[0], **kwargs)
            else:
                return self.aug_test(imgs, img_metas, **kwargs)

    def show_result(self,
                    img,
                    result,
                    ori_img=None,
                    cam_intrinsic=None,
                    gt_bboxes_3d=None,
                    gt_labels=None,
                    score_thr=0.3,
                    cov_scale=5.0,
                    bev_scale=25,
                    thickness=2,
                    win_name='',
                    show=False,
                    wait_time=0,
                    out_file=None,
                    views=['2d', '3d', 'bev']):
        img = mmcv.imread(img)
        ori_img = mmcv.imread(ori_img)
        img_show = []
        if '3d' in views:
            img_pred_3d = ori_img.copy()
            draw_box_3d_pred(
                img_pred_3d,
                result['bbox_3d_results'],
                cam_intrinsic,
                score_thr=score_thr)
            img_show.append(img_pred_3d)
        if 'bev' in views:
            orientation = None
            gt_orientation = None
            if self.CLS_ORIENTATION is not None:
                orientation = []
                for i, bbox_3d_result in enumerate(result['bbox_3d_results']):
                    orientation.append(
                        np.array([self.CLS_ORIENTATION[i]] * len(bbox_3d_result)))
                if gt_labels is not None:
                    gt_orientation = [self.CLS_ORIENTATION[label] for label in gt_labels]
            viz_bev = show_bev(
                ori_img, result['bbox_results'], result['bbox_3d_results'],
                cam_intrinsic, width=ori_img.shape[1], height=img.shape[0], scale=bev_scale,
                pose_samples=result.get('pose_samples', None),
                pose_sample_weights=result.get('pose_sample_weights', None),
                orientation=orientation,
                gt_bboxes_3d=gt_bboxes_3d,
                gt_orientation=gt_orientation,
                score_thr=score_thr, thickness=2)
            img_show.append(viz_bev)
        if show:
            if len(img_show) == 1:
                img_show = img_show[0]
            elif len(img_show) == 2:
                img_show = np.concatenate(img_show, axis=0)
            else:
                raise ValueError('no view to show')
            mmcv.imshow(img_show, win_name, wait_time)
        if out_file is not None:
            if '3d' in views:
                mmcv.imwrite(img_pred_3d, out_file[:-4] + '_3d.jpg')
            if 'bev' in views:
                mmcv.imwrite(viz_bev, out_file[:-4] + '_bev.png')
            if '2d' in views:
                assert 'bbox_results' in result
                multi_cls_results = np.concatenate(result['bbox_results'], axis=0)
                labels = []
                for i, bbox_single in enumerate(result['bbox_results']):
                    labels += [i] * bbox_single.shape[0]
                labels = np.array(labels)
                imshow_det_bboxes(
                    ori_img,
                    multi_cls_results,
                    labels,
                    class_names=self.CLASSES,
                    score_thr=score_thr,
                    thickness=thickness,
                    show=False,
                    out_file=out_file[:-4] + '_2d.jpg')
            if 'score' in views:
                assert 'score' in result
                score = result['score'][:, :img.shape[0], :img.shape[1]].sum(axis=0)
                score = (score * 256).clip(min=0, max=255).astype(np.uint8)
                score = score[..., None] * 0.8 + img * 0.2
                mmcv.imwrite(score, out_file[:-4] + '_score.jpg')
            if 'pts' in views:
                assert 'x2d' in result and 'w2d' in result
                num_head = self.bbox_head.num_heads
                pts_obj, pts_head, pts_xy = deformable_point_vis(
                    ori_img, result, score_thr, num_head)
                mmcv.imwrite(pts_obj, out_file[:-4] + '_pts_obj.jpg')
                mmcv.imwrite(pts_head, out_file[:-4] + '_pts_head.jpg')
                mmcv.imwrite(pts_xy, out_file[:-4] + '_pts_xy.jpg')
            if 'orient' in views:
                assert 'orient_logprob' in result and 'bbox_results' in result
                dirname = out_file[:-4] + '/'
                os.makedirs(dirname, exist_ok=True)
                for cls_id in range(len(self.CLASSES)):
                    for i, (bbox_single, rot_logprob_single) in enumerate(zip(
                            result['bbox_results'][cls_id], result['orient_logprob'][cls_id])):
                        if bbox_single[4] < score_thr:
                            continue
                        filename = '{}_{:02d}'.format(self.CLASSES[cls_id], i)
                        x1, y1, x2, y2 = bbox_single[:4].round().astype(np.int64)
                        img_patch = ori_img[y1:y2, x1:x2]
                        mmcv.imwrite(img_patch, dirname + filename + '.jpg')
                        rot_bins = len(rot_logprob_single)
                        radian_div_pi = np.linspace(0, 2 * (rot_bins - 1) / rot_bins, num=rot_bins)
                        plt.figure(figsize=(4, 2))
                        plt.plot(radian_div_pi, np.exp(rot_logprob_single))
                        plt.xlim([0, 2])
                        plt.gca().set_ylim(bottom=0)
                        plt.xticks([0, 0.5, 1, 1.5, 2],
                                   ['0', '$\pi$/2', '$\pi$', '3$\pi$/2', '2$\pi$'])
                        plt.gca().xaxis.grid(True)
                        plt.xlabel('Yaw')
                        plt.ylabel('Density')
                        plt.tight_layout()
                        plt.savefig(dirname + filename + '.png')
                        plt.savefig(dirname + filename + '.eps')
                        plt.close()

        if not (show or out_file):
            return img_show
