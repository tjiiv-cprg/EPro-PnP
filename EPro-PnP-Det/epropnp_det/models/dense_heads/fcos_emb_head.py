"""
Copyright (C) 2010-2022 Alibaba Group Holding Limited.
This file is modified from
https://github.com/open-mmlab/mmdetection3d
"""

import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, bias_init_with_prob, normal_init

from mmdet.core import multi_apply, reduce_mean
from mmdet.models import HEADS, build_loss
from mmdet.models.dense_heads import AnchorFreeHead
from mmdet.models.dense_heads.base_dense_head import BaseDenseHead

INF = 1e8


@HEADS.register_module()
class FCOSEmbHead(AnchorFreeHead):

    def __init__(self,
                 num_classes,
                 in_channels,
                 feat_channels=256,
                 stacked_convs=2,
                 strides=[8, 16, 32, 64, 128],
                 regress_ranges=((-1, 48), (48, 96), (96, 192), (192, 384),
                                 (384, 1e8)),
                 cls_branch=(256, ),
                 centerness_branch=(64, ),
                 offset_branch=(256, ),
                 emb_branch=(256, ),
                 emb_channels=256,
                 centerness_alpha=2.5,
                 loss_cls=dict(
                     type='FocalLoss',
                     use_sigmoid=True,
                     gamma=2.0,
                     alpha=0.25,
                     loss_weight=1.0),
                 loss_rp=dict(
                     type='SmoothL1LossMod', beta=1.0, loss_weight=1.0),
                 loss_centerness=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=True,
                     loss_weight=1.0),
                 center_sampling=True,
                 center_sample_radius=1.5,
                 center_error_scale=0.2,
                 offset_cls_agnostic=True,
                 dcn_on_last_conv=True,
                 conv_bias='auto',
                 conv_cfg=None,
                 norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None):
        super(AnchorFreeHead, self).__init__(init_cfg)
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.stacked_convs = stacked_convs
        self.strides = strides
        self.dcn_on_last_conv = dcn_on_last_conv
        assert conv_bias == 'auto' or isinstance(conv_bias, bool)
        self.conv_bias = conv_bias
        self.regress_ranges = regress_ranges
        self.center_sampling = center_sampling
        self.center_sample_radius = center_sample_radius
        self.center_error_scale = center_error_scale
        self.offset_cls_agnostic = offset_cls_agnostic
        self.cls_branch = cls_branch
        self.centerness_branch = centerness_branch
        self.offset_branch = offset_branch
        self.emb_branch = emb_branch
        self.emb_channels = emb_channels
        self.centerness_alpha = centerness_alpha

        self.loss_cls = build_loss(loss_cls)
        self.loss_rp = build_loss(loss_rp)
        self.loss_centerness = build_loss(loss_centerness)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg

        self._init_layers()

    def _init_layers(self):
        super(FCOSEmbHead, self)._init_layers()

    def _init_predictor(self):
        self.conv_cls_prev = self._init_branch(
            conv_channels=self.cls_branch, conv_strides=(1, ) * len(self.cls_branch))
        self.conv_centerness_prev = self._init_branch(
            conv_channels=self.centerness_branch, conv_strides=(1, ) * len(self.centerness_branch))
        self.conv_emb_prev = self._init_branch(
            conv_channels=self.emb_branch, conv_strides=(1, ) * len(self.emb_branch))
        self.conv_offset_prev = self._init_branch(
            conv_channels=self.offset_branch, conv_strides=(1, ) * len(self.offset_branch))

        self.conv_cls = nn.Conv2d(self.cls_branch[-1], self.num_classes, 1)
        self.conv_centerness = nn.Conv2d(self.centerness_branch[-1], 1, 1)
        self.conv_offset = nn.Conv2d(
            self.offset_branch[-1],
            2 if self.offset_cls_agnostic else self.num_classes * 2,
            1)
        self.conv_emb = ConvModule(
            self.emb_branch[-1],
            self.emb_channels,
            1,
            stride=1,
            padding=0,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            bias=self.conv_bias)

    def _init_branch(self, conv_channels=(64, ), conv_strides=(1, )):
        conv_before_pred = nn.ModuleList()
        if isinstance(conv_channels, int):
            conv_channels = [self.feat_channels] + [conv_channels]
            conv_strides = [conv_strides]
        else:
            conv_channels = [self.feat_channels] + list(conv_channels)
            conv_strides = list(conv_strides)
        for i in range(len(conv_strides)):
            conv_before_pred.append(
                ConvModule(
                    conv_channels[i],
                    conv_channels[i + 1],
                    3,
                    stride=conv_strides[i],
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    bias=self.conv_bias))
        return conv_before_pred

    def init_weights(self):
        super(BaseDenseHead, self).init_weights()
        normal_init(self.conv_cls, std=0.01, bias=bias_init_with_prob(0.01))

    def forward(self, feats):
        """Forward features from the upstream network.

        Args:
            feats (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.

        Returns:
            tuple:
                cls_scores (list[Tensor]): Box scores for each scale level, \
                    each is a 4D-tensor, the channel number is \
                    num_points * num_classes.
                bbox_preds (list[Tensor]): Box energies / deltas for each \
                    scale level, each is a 4D-tensor, the channel number is \
                    num_points * 4.
                centernesses (list[Tensor]): centerness for each scale level, \
                    each is a 4D-tensor, the channel number is num_points * 1.
        """
        return multi_apply(self.forward_single, feats, self.strides)

    def forward_single(self, x, stride):
        num_img, _, h, w = x.size()

        cls_feat = x
        reg_feat = x

        for cls_layer in self.cls_convs:
            cls_feat = cls_layer(cls_feat)
        for reg_layer in self.reg_convs:
            reg_feat = reg_layer(reg_feat)
        emb_feat = reg_feat
        offset_feat = reg_feat
        centerness_feat = reg_feat

        for conv_centerness_prev_layer in self.conv_centerness_prev:
            centerness_feat = conv_centerness_prev_layer(centerness_feat)
        centerness = self.conv_centerness(centerness_feat)
        for conv_cls_prev_layer in self.conv_cls_prev:
            cls_feat = conv_cls_prev_layer(cls_feat)
        cls_score = self.conv_cls(cls_feat)
        for conv_emb_prev_layer in self.conv_emb_prev:
            emb_feat = conv_emb_prev_layer(emb_feat)
        obj_emb = self.conv_emb(emb_feat)
        for conv_offset_prev_layer in self.conv_offset_prev:
            offset_feat = conv_offset_prev_layer(offset_feat)
        points = self._get_points_single(x.shape[-2:], stride, x.dtype, x.device)  # (h * w, 2)
        offset = self.conv_offset(offset_feat) * stride
        if self.offset_cls_agnostic:
            center = offset + points.reshape(h, w, 2).permute(2, 0, 1)  # (num_img, 2, h, w)
        else:
            center = (offset.reshape(num_img, self.num_classes, 2, h, w)
                      + points.reshape(h, w, 2).permute(2, 0, 1)
                      ).reshape(num_img, self.num_classes * 2, h, w)
        return cls_score, center, centerness, obj_emb, points

    def loss(self,
             flatten_cls_scores,
             flatten_center,
             flatten_centerness,
             flatten_labels,
             flatten_gt_inds,
             flatten_centerness_targets,
             centers2d,
             gt_bboxes):
        # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
        pos_inds = ((flatten_labels >= 0)
                    & (flatten_labels < self.num_classes)).nonzero().reshape(-1)
        num_pos = flatten_cls_scores.new_tensor([len(pos_inds)])
        num_pos_reduce = max(reduce_mean(num_pos), 1.0)
        loss_cls = self.loss_cls(  # binary focal loss
            flatten_cls_scores,
            flatten_labels,
            avg_factor=num_pos_reduce)

        if self.offset_cls_agnostic:
            pos_center = flatten_center[pos_inds]
        else:
            pos_center = flatten_center.reshape(-1, self.num_classes, 2)[pos_inds, flatten_labels[pos_inds]]
        pos_centerness = flatten_centerness[pos_inds]
        pos_gt_inds = flatten_gt_inds[pos_inds]

        pos_centerness_targets = flatten_centerness_targets[pos_inds]
        pos_center_gt = centers2d[pos_gt_inds]
        pos_bbox_gt = gt_bboxes[pos_gt_inds]
        ref_length = pos_bbox_gt[:, 2:] - pos_bbox_gt[:, :2]  # (num_pts, 2) in [w, h]
        rel_center_error = (pos_center - pos_center_gt) / (
                self.center_error_scale * (ref_length + getattr(self.train_cfg, 'min_ref_length', 4.0)))
        loss_rp = self.loss_rp(
            rel_center_error, 0,
            weight=pos_centerness_targets[:, None],
            avg_factor=max(reduce_mean(pos_centerness_targets.sum()), 1e-6) * 2)
        loss_centerness = self.loss_centerness(
            pos_centerness, pos_centerness_targets,
            avg_factor=num_pos_reduce)

        loss_dict = dict(
            loss_cls=loss_cls,
            loss_rp=loss_rp,
            loss_centerness=loss_centerness)

        return loss_dict

    def get_preds(self,
                  cls_scores,
                  centernesses,
                  *args,
                  cfg=None):
        cfg = self.test_cfg if cfg is None else cfg
        bs = cls_scores[0].size(0)

        max_obj_per_batch = getattr(cfg, 'max_obj_per_img', 256) * bs
        min_fcos_score = getattr(cfg, 'min_fcos_score', 0.04)

        concat_cls_score = torch.cat([
            cls_score.permute(0, 2, 3, 1).reshape(bs, -1, self.num_classes)
            for cls_score in cls_scores], dim=1).sigmoid_()
        concat_centerness = torch.cat([
            centerness.permute(0, 2, 3, 1).reshape(bs, -1, 1)
            for centerness in centernesses], dim=1).sigmoid_()
        fcos_score = concat_cls_score * concat_centerness
        inds = (fcos_score >= min_fcos_score).nonzero()
        if inds.size(0) > max_obj_per_batch:
            inds = inds[fcos_score[inds.unbind(dim=-1)].topk(max_obj_per_batch, sorted=False)[1]]

        img_inds, point_inds, labels = inds.unbind(dim=-1)

        score = concat_cls_score[img_inds, point_inds, labels]

        concat_stride = torch.cat([
            centerness.new_full((centerness.shape[-2:].numel(), ), stride)
            for centerness, stride in zip(centernesses, self.strides)], dim=0)
        topk_strides = concat_stride[point_inds]  # (num_obj_per_batch, )

        topk_preds = []
        for mlvl_pred in args:
            concat_pred = torch.cat([
                pred.permute(0, 2, 3, 1).reshape(bs, -1, pred.size(1))
                for pred in mlvl_pred], dim=1)
            # (num_obj_per_batch, num_chn)
            topk_preds.append(concat_pred[img_inds, point_inds])

        return (img_inds, score, labels, topk_strides, *topk_preds)

    def _get_points_single(self,
                           featmap_size,
                           stride,
                           dtype,
                           device):
        """Get points according to feature map sizes."""
        y, x = super()._get_points_single(featmap_size, stride, dtype, device)
        points = torch.stack((x.reshape(-1) * stride, y.reshape(-1) * stride),
                             dim=-1) + stride // 2
        return points

    def get_targets(self, points, gt_bboxes_list, gt_labels_list, centers2d_list):
        """Compute regression, classification and centerness targets for points
        in multiple images.

        Args:
            points (list[Tensor]): Points of each fpn level, each has shape
                (num_points_per_lvl, 2).
            gt_bboxes_list (list[Tensor]): Ground truth bboxes of each image,
                each has shape (num_gt, 4).
            gt_labels_list (list[Tensor]): Ground truth labels of each box,
                each has shape (num_gt,).

        Returns:
            tuple:
                concat_lvl_labels (list[Tensor]): Labels of each level. \
                concat_lvl_bbox_targets (list[Tensor]): BBox targets of each \
                    level.
        """
        assert len(points) == len(self.regress_ranges)
        num_levels = len(points)
        # expand regress ranges to align with points
        expanded_regress_ranges = [
            points[i].new_tensor(self.regress_ranges[i])[None].expand_as(
                points[i]) for i in range(num_levels)
        ]
        # concat all levels points and regress ranges
        concat_regress_ranges = torch.cat(expanded_regress_ranges, dim=0)
        concat_points = torch.cat(points, dim=0)

        # the number of points per img, per lvl
        num_points_per_lvl = [lvl_points.size(0) for lvl_points in points]

        # get labels and bbox_targets of each image
        labels_list, centerness_targets_list, gt_inds_list = multi_apply(
            self._get_target_single,
            gt_bboxes_list,
            gt_labels_list,
            centers2d_list,
            points=concat_points,
            regress_ranges=concat_regress_ranges,
            num_points_per_lvl=num_points_per_lvl)

        # split to per img, per level
        labels_list = [labels.split(num_points_per_lvl, 0) for labels in labels_list]
        centerness_targets_list = [
            centerness_targets.split(num_points_per_lvl, 0) for centerness_targets in centerness_targets_list]
        num_gt_per_img = [gt_labels.size(0) for gt_labels in gt_labels_list]
        gt_inds_base = gt_inds_list[0].new_tensor([0] + num_gt_per_img[:-1]).cumsum(0)
        gt_inds_list = [(gt_inds + gt_inds_base[i]).split(num_points_per_lvl, 0)
                        for i, gt_inds in enumerate(gt_inds_list)]

        # concat per level image
        concat_lvl_labels = []
        concat_lvl_centerness_targets = []
        concat_lvl_gt_inds = []
        for i in range(num_levels):
            concat_lvl_labels.append(
                torch.cat([labels[i] for labels in labels_list]))
            concat_lvl_centerness_targets.append(
                torch.cat([centerness_targets[i] for centerness_targets in centerness_targets_list]))
            concat_lvl_gt_inds.append(
                torch.cat([gt_inds[i] for gt_inds in gt_inds_list]))
        return concat_lvl_labels, concat_lvl_centerness_targets, concat_lvl_gt_inds

    def _get_target_single(self, gt_bboxes, gt_labels, centers2d,
                           points, regress_ranges, num_points_per_lvl):
        num_points = points.size(0)
        num_gts = gt_labels.size(0)
        if num_gts == 0:
            return (gt_labels.new_full((num_points, ), self.num_classes),
                    gt_bboxes.new_zeros((num_points, )),
                    gt_labels.new_zeros((num_points, )))

        regress_ranges = regress_ranges[:, None, :].expand(
            num_points, num_gts, 2)
        gt_bboxes = gt_bboxes[None].expand(num_points, num_gts, 4)
        centers2d = centers2d[None].expand(num_points, num_gts, 2)

        xs, ys = points[:, 0], points[:, 1]
        xs = xs[:, None].expand(num_points, num_gts)
        ys = ys[:, None].expand(num_points, num_gts)

        delta_xys = torch.stack(((xs - centers2d[..., 0]),
                                 (ys - centers2d[..., 1])), dim=-1)

        left = xs - gt_bboxes[..., 0]
        right = gt_bboxes[..., 2] - xs
        top = ys - gt_bboxes[..., 1]
        bottom = gt_bboxes[..., 3] - ys
        bbox_targets = torch.stack((left, top, right, bottom), -1)

        assert self.center_sampling is True, 'Setting center_sampling to '\
            'False has not been implemented for FCOS3D.'
        # condition1: inside a `center bbox`
        radius = self.center_sample_radius
        center_xs = centers2d[..., 0]
        center_ys = centers2d[..., 1]
        center_gts = torch.zeros_like(gt_bboxes)
        stride = center_xs.new_zeros(center_xs.shape)

        # project the points on current lvl back to the `original` sizes
        lvl_begin = 0
        for lvl_idx, num_points_lvl in enumerate(num_points_per_lvl):
            lvl_end = lvl_begin + num_points_lvl
            stride[lvl_begin:lvl_end] = self.strides[lvl_idx] * radius
            lvl_begin = lvl_end

        center_gts[..., 0] = center_xs - stride
        center_gts[..., 1] = center_ys - stride
        center_gts[..., 2] = center_xs + stride
        center_gts[..., 3] = center_ys + stride

        cb_dist_left = xs - center_gts[..., 0]
        cb_dist_right = center_gts[..., 2] - xs
        cb_dist_top = ys - center_gts[..., 1]
        cb_dist_bottom = center_gts[..., 3] - ys
        center_bbox = torch.stack(
            (cb_dist_left, cb_dist_top, cb_dist_right, cb_dist_bottom), -1)
        inside_gt_bbox_mask = center_bbox.min(-1)[0] > 0

        # condition2: limit the regression range for each location
        max_regress_distance = bbox_targets.max(-1)[0]
        inside_regress_range = (
            (max_regress_distance >= regress_ranges[..., 0])
            & (max_regress_distance <= regress_ranges[..., 1]))

        # center-based criterion to deal with ambiguity
        dists = delta_xys.norm(dim=-1)
        dists[~inside_gt_bbox_mask] = INF
        dists[~inside_regress_range] = INF
        min_dist, min_dist_inds = dists.min(dim=1)

        labels = gt_labels[min_dist_inds]
        labels[min_dist == INF] = self.num_classes  # set as BG

        relative_dists = min_dist / (1.414 * stride[:, 0])
        # [N, 1] / [N, 1]
        centerness_targets = torch.exp(-self.centerness_alpha * relative_dists)

        return labels, centerness_targets, min_dist_inds

    def get_bboxes(self, *args, **kwargs):
        pass
