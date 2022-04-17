"""
Copyright (C) 2010-2022 Alibaba Group Holding Limited.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv.cnn import ConvModule, build_upsample_layer, kaiming_init, xavier_init, constant_init, Scale
from torch.nn.modules.batchnorm import _BatchNorm

from mmcv.ops import roi_align, batched_nms
from mmcv.cnn.bricks.transformer import build_positional_encoding, build_attention, build_transformer_layer
from mmdet.core import bbox2roi, reduce_mean, multi_apply
from mmdet.models import HEADS, build_head, build_loss
from mmdet.models.dense_heads.base_dense_head import BaseDenseHead

from ...core import build_proj_error_coder, build_dim_coder, bbox3d_overlaps_aligned_torch, \
    project_to_image, gen_unit_noc, batched_bev_nms, \
    build_center_target, bboxes_3d_to_2d
from ...ops import (GroupLinear, logsoftmax_across_rois, build_pnp, build_camera,
                    build_cost_fun, evaluate_pnp)
from ...runner.hooks.model_updater import rgetattr, rsetattr
from ...utils.timer import default_timers

default_timers.add_timer('FCOS head forward time')
default_timers.add_timer('FCOS head post-proc. time')
default_timers.add_timer('dense head time')
default_timers.add_timer('point & object subheads time')
default_timers.add_timer('PnP time')
default_timers.add_timer('PnP batch size')
default_timers.add_timer('post-proc. time')


@HEADS.register_module()
class DefromPnPHead(BaseDenseHead):

    def __init__(self,
                 num_classes=10,
                 in_channels=256,
                 lvl_feat_channels=(256, 128, 128),
                 strides=(4, 8, 16, 32, 64, 128),
                 output_stride=4,
                 dense_lvl_range=(0, 4),
                 det_lvl_range=(1, 6),
                 dense_channels=256,
                 embed_dims=256,
                 num_heads=8,
                 num_points=32,
                 num_pred_fcs=2,
                 upsample_cfg=dict(type='bilinear'),
                 detector=dict(type='FCOSEmbHead'),
                 proj_error_coder=dict(type='DistDimProjErrorCoder'),
                 dim_coder=dict(type='MultiClassLogDimCoder'),
                 positional_encoding=dict(
                     type='SinePositionalEncodingMod',
                     num_feats=128,
                     normalize=True,
                     offset=-0.5),
                 loss_pose=dict(
                     type='MonteCarloPoseLoss',
                     loss_weight=0.15,
                     momentum=0.01),
                 loss_proj=dict(
                     type='MVDGaussianMixtureNLLLoss',
                     loss_weight=0.5),
                 loss_dim=dict(
                     type='SmoothL1LossMod',
                     loss_weight=1.0),
                 loss_regr=dict(
                     type='SmoothL1LossMod',
                     beta=0.05,
                     loss_weight=0.25),
                 loss_score=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=True,
                     loss_weight=1.0),
                 loss_reg_pos=dict(
                     type='SmoothL1LossMod',
                     beta=1.0,
                     loss_weight=0.05),
                 loss_reg_orient=dict(
                     type='CosineAngleLoss',
                     loss_weight=0.05),
                 loss_velo=dict(
                     type='SmoothL1LossMod',
                     loss_weight=0.05),
                 loss_attr=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=False,
                     loss_weight=0.5),
                 attention_sampler=dict(
                     type='DeformableAttentionSampler',
                     embed_dims=256,
                     num_heads=8,
                     num_points=32,
                     stride=4),
                 num_pts_trans_layers=1,
                 pts_trans=dict(
                     type='BaseTransformerLayer',
                     attn_cfgs=[dict(
                                type='MultiheadAttention',
                                embed_dims=32,
                                num_heads=1)],
                     ffn_cfgs=dict(
                         type='FFN',
                         embed_dims=32,
                         feedforward_channels=256,
                         num_fcs=2,
                         ffn_drop=0.1,
                         act_cfg=dict(type='ReLU', inplace=True)),
                     operation_order=('self_attn', 'norm', 'ffn', 'norm'),
                     batch_first=True),
                 center_target=dict(
                     type='VolumeCenter',
                     output_stride=4,
                     render_stride=4,
                     min_box_size=4.0),
                 pnp=dict(
                     type='EProPnP4DoF',
                     mc_samples=512,
                     num_iter=4,
                     normalize=True,
                     solver=dict(
                         type='LMSolver',
                         num_iter=10,
                         normalize=True,
                         init_solver=dict(
                             type='RSLMSolver',
                             num_points=16,
                             num_proposals=64,
                             num_iter=3))),
                 camera=dict(type='PerspectiveCamera'),
                 cost_fun=dict(
                     type='AdaptiveHuberPnPCost',
                     relative_delta=0.5),
                 use_cls_emb=False,
                 dim_cls_agnostic=False,
                 pred_velo=True,
                 pred_attr=True,
                 num_attrs=9,
                 score_type='te',
                 conv_bias='auto',
                 conv_cfg=None,
                 norm_cfg=None,
                 train_cfg=None,
                 test_cfg=None):
        super(DefromPnPHead, self).__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.lvl_feat_channels = lvl_feat_channels
        self.num_heads = num_heads
        self.num_points = num_points
        self.num_pred_fcs = num_pred_fcs
        self.strides = strides
        self.output_stride = output_stride
        self.dense_lvl_range = dense_lvl_range
        self.det_lvl_range = det_lvl_range
        self.dense_channels = dense_channels
        self.embed_dims = embed_dims
        self.use_cls_emb = use_cls_emb
        self.dim_cls_agnostic = dim_cls_agnostic
        self.pred_velo = pred_velo
        self.pred_attr = pred_attr
        self.num_attrs = num_attrs
        assert score_type in ['te', 'iou']
        self.score_type = score_type
        assert conv_bias == 'auto' or isinstance(conv_bias, bool)
        self.conv_bias = conv_bias
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.upsample_cfg = upsample_cfg.copy()
        if self.upsample_cfg['type'] not in [
            'deconv', 'nearest', 'bilinear', 'carafe']:
            raise ValueError(
                f'Invalid upsample method {self.upsample_cfg["type"]}, '
                'accepted methods are "deconv", "nearest", "bilinear", '
                '"carafe"')
        self.upsample_method = self.upsample_cfg.get('type')

        detector.update(num_classes=self.num_classes,
                        in_channels=self.in_channels,
                        train_cfg=train_cfg,
                        test_cfg=test_cfg)
        self.detector = build_head(detector)
        self.attention_sampler = build_attention(attention_sampler)

        self.num_pts_trans_layer = num_pts_trans_layers
        self.obj_query_scale = nn.ModuleList(
            [Scale(0.1) for _ in range(num_pts_trans_layers)])
        self.pts_trans = nn.ModuleList(
            [build_transformer_layer(pts_trans) for _ in range(num_pts_trans_layers)])
        self.x2d_pos_enc = nn.Linear(2, self.pts_trans[0].embed_dims)
        self.corr_regs = nn.ModuleList(
            [GroupLinear(self.embed_dims, self.num_heads * 5, self.num_heads)
             for _ in range(num_pts_trans_layers + 1)])

        self.center_target = build_center_target(center_target)
        self.pnp = build_pnp(pnp)
        self.camera = build_camera(camera)
        self.cost_fun = build_cost_fun(cost_fun)

        self.proj_error_coder = build_proj_error_coder(proj_error_coder)
        self.dim_coder = build_dim_coder(dim_coder)
        self.positional_encoding = build_positional_encoding(positional_encoding)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.loss_proj = build_loss(loss_proj) if loss_proj is not None else None
        self.loss_dim = build_loss(loss_dim)
        self.loss_regr = build_loss(loss_regr) if loss_regr is not None else None
        self.loss_pose = nn.ModuleList([build_loss(loss_pose) for _ in range(num_pts_trans_layers)])
        self.loss_score = build_loss(loss_score)
        self.loss_reg_pos = build_loss(loss_reg_pos)
        self.loss_reg_orient = build_loss(loss_reg_orient)
        if self.pred_velo:
            self.loss_velo = build_loss(loss_velo)
        if self.pred_attr:
            self.loss_attr = build_loss(loss_attr)

        self._init_convs()
        self._init_upsamples()
        self._init_out_layers()

        self.train_cfg_backup = dict()
        for key, value in self.test_cfg.get('override_cfg', dict()).items():
            self.train_cfg_backup[key] = rgetattr(self, key)

    def _init_convs(self):
        self.convs = nn.ModuleList()
        for i, feat_channel in enumerate(self.lvl_feat_channels):
            chn = self.in_channels if i == 0 else self.lvl_feat_channels[i - 1]
            self.convs.append(
                ConvModule(
                    chn,
                    feat_channel,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    bias=False))

    def _init_upsamples(self):
        self.upsamples = []
        for stride in self.strides[self.dense_lvl_range[0]:self.dense_lvl_range[1]]:
            upsample_cfg_ = self.upsample_cfg.copy()
            scale_factor = stride // self.output_stride
            if scale_factor == 1:
                self.upsamples.append(None)
            else:
                if self.upsample_method == 'deconv':
                    upsample_cfg_.update(
                        in_channels=self.lvl_feat_channels[-1],
                        out_channels=self.lvl_feat_channels[-1],
                        kernel_size=scale_factor,
                        stride=scale_factor)
                    self.upsamples.append(build_upsample_layer(upsample_cfg_))
                elif self.upsample_method == 'carafe':
                    upsample_cfg_.update(
                        channels=self.lvl_feat_channels[-1], scale_factor=scale_factor)
                    self.upsamples.append(build_upsample_layer(upsample_cfg_))
                else:
                    # suppress warnings
                    align_corners = (None
                                     if self.upsample_method == 'nearest' else False)
                    upsample_cfg_.update(
                        scale_factor=scale_factor,
                        mode=self.upsample_method,
                        align_corners=align_corners)
                    self.upsamples.append(build_upsample_layer(upsample_cfg_))

    def _init_out_layers(self):
        in_channels = self.lvl_feat_channels[-1] * len(self.upsamples)
        self.conv_upsampled = ConvModule(
            in_channels,
            self.dense_channels,
            1,
            stride=1,
            padding=0,
            conv_cfg=self.conv_cfg,
            norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
            bias=self.conv_bias)
        self.k_proj = nn.Conv2d(self.dense_channels + self.embed_dims, self.embed_dims, 1)
        self.v_proj = nn.Conv2d(self.dense_channels, self.embed_dims, 1)
        self.query_scale = Scale(0.1)
        self.query_proj = nn.Linear(self.embed_dims, self.embed_dims)
        pred_fc = []
        for _ in range(self.num_pred_fcs):
            pred_fc.append(nn.Linear(self.embed_dims, self.embed_dims))
            pred_fc.append(nn.ReLU())
        self.pred_fc = nn.Sequential(*pred_fc)
        self.dim_branch = nn.Linear(
            self.embed_dims, 3 if self.dim_cls_agnostic else self.num_classes * 3)  # [l, h, w]
        self.score_branch = nn.Linear(self.embed_dims, 1)
        self.scale_branch = nn.Linear(self.embed_dims, 2)
        if self.use_cls_emb:
            self.cls_emb = nn.Parameter(torch.zeros([self.num_classes, self.embed_dims]))
        if self.pred_velo:
            self.velo_branch = nn.Linear(self.embed_dims, 2)
        if self.pred_attr:
            self.attr_branch = nn.Linear(self.embed_dims, self.num_attrs)

    def init_weights(self):
        for m in self.children():
            if hasattr(m, 'init_weights'):
                m.init_weights()
        for m in self.convs.modules():
            if isinstance(m, nn.Conv2d):
                kaiming_init(m)
            elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                constant_init(m, 1)
            if hasattr(m, 'conv_offset'):
                constant_init(m.conv_offset, 0)
        for m in self.conv_upsampled.modules():
            if isinstance(m, nn.Conv2d):
                kaiming_init(m)
            elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                constant_init(m, 1)
        kaiming_init(self.k_proj, nonlinearity='linear')
        kaiming_init(self.v_proj, nonlinearity='linear')
        xavier_init(self.query_proj, distribution='uniform')
        for i, corr_reg in enumerate(self.corr_regs):
            corr_reg.weight.data *= 0.01
            corr_reg.bias.data.view(
                self.num_heads, corr_reg.out_features // self.num_heads
            )[..., :3] = gen_unit_noc(self.num_heads) / 2
        constant_init(self.scale_branch, 0)
        self._is_init = True

    def train(self, mode=True):
        if mode:
            for key, value in self.train_cfg_backup.items():
                rsetattr(self, key, value)
        else:
            for key, value in self.test_cfg.get('override_cfg', dict()).items():
                if self.training:
                    self.train_cfg_backup[key] = rgetattr(self, key)
                rsetattr(self, key, value)
        super(DefromPnPHead, self).train(mode)
        return self

    def forward_dense_single(self, x):
        for conv in self.convs:
            x = conv(x)
        return x

    def forward_det_dense(self, mlvl_feats, img_metas):
        with default_timers['FCOS head forward time']:
            batch_size = mlvl_feats[0].size(0)
            input_img_h, input_img_w = img_metas[0]['batch_input_shape']
            img_masks = mlvl_feats[0].new_ones(
                (batch_size, input_img_h, input_img_w))
            for img_id in range(batch_size):
                img_h, img_w, _ = img_metas[img_id]['img_shape']
                img_masks[img_id, :img_h, :img_w] = 0

            detector_feats = mlvl_feats[self.det_lvl_range[0]:self.det_lvl_range[1]]
            mlvl_cls_score, mlvl_center, mlvl_centerness, mlvl_obj_emb, mlvl_points = self.detector(detector_feats)

        with default_timers['dense head time']:
            mlvl_feats = [self.forward_dense_single(lvl_feats)
                          for lvl_feats in mlvl_feats[self.dense_lvl_range[0]:self.dense_lvl_range[1]]]
            concat_feats = []
            for lvl_feats, upsample in zip(mlvl_feats, self.upsamples):
                concat_feats.append(lvl_feats if upsample is None else upsample(lvl_feats))
            concat_feats = self.conv_upsampled(torch.cat(concat_feats, dim=1))

            mask = F.interpolate(img_masks[None], size=concat_feats.shape[-2:]).to(torch.bool).squeeze(0)
            # (n, embed_dim, h, w)
            key = self.k_proj(
                torch.cat((concat_feats, self.positional_encoding(mask)), dim=1))
            value = self.v_proj(concat_feats)

        return (mlvl_cls_score, mlvl_center, mlvl_centerness, mlvl_obj_emb, mlvl_points,
                key, value)

    def forward_correspondence(
            self, v_samples, x2d_samples, mask_samples, obj_query, sample_flips):
        num_obj = v_samples.size(0)
        num_multihead_points = self.num_heads * self.num_points
        head_emb_dim = self.embed_dims // self.num_heads
        v_samples = v_samples.transpose(-1, -2).reshape(num_obj, num_multihead_points, head_emb_dim)
        x2d = x2d_samples.transpose(-1, -2).reshape(num_obj, num_multihead_points, 2)
        mask_samples = mask_samples.transpose(-1, -2)

        x2d_flip = x2d.detach().clone()
        x2d_flip[sample_flips, :, 0] = -x2d_flip[sample_flips, :, 0]
        x2d_std, x2d_mean = torch.std_mean(x2d_flip, dim=1, keepdim=True)
        pos_enc = self.x2d_pos_enc((x2d_flip - x2d_mean) / x2d_std.clamp(min=1.0))
        obj_query = obj_query.expand(-1, -1, self.num_points, -1).reshape(
            num_obj, num_multihead_points, head_emb_dim)
        noc_list = []
        w2d_list = []
        for i, (pts_trans, scale) in enumerate(zip(self.pts_trans, self.obj_query_scale)):
            v_samples = v_samples + scale(obj_query)
            if num_obj > 0:
                v_samples = pts_trans(v_samples, query_pos=pos_enc, key_pos=pos_enc)
            else:  # avoid MMCV transformer error for empty batch
                if self.training:
                    v_samples = F.pad(v_samples, [0, 0, 0, 0, 0, 1])
                    pos_enc = F.pad(pos_enc, [0, 0, 0, 0, 0, 1])
                    v_samples = pts_trans(v_samples, query_pos=pos_enc, key_pos=pos_enc)
                    v_samples = v_samples[:0]
                else:
                    v_samples = v_samples
            v_samples_ = v_samples.reshape(
                num_obj, self.num_heads, self.num_points, head_emb_dim
            ).transpose(1, 2).reshape(
                num_obj, self.num_points, self.embed_dims)
            regr_samples = self.corr_regs[i + 1](v_samples_).reshape(
                num_obj, self.num_points, self.num_heads, 5).transpose(2, 1)
            # (num_obj_sample, num_head, num_point, [3, 2])
            noc, w2d = regr_samples.split([3, 2], dim=-1)
            noc_flip = noc.clone()
            noc_flip[sample_flips, :, :, 2] = -noc[sample_flips, :, :, 2]  # flip correction
            w2d = w2d.reshape(
                num_obj, num_multihead_points, 2
            ).softmax(dim=1).reshape(num_obj, self.num_heads, self.num_points, 2)
            w2d = w2d * mask_samples
            noc_list.append(noc_flip.reshape(num_obj, num_multihead_points, 3))
            w2d_list.append(w2d.reshape(num_obj, num_multihead_points, 2))

        return noc_list, w2d_list, x2d

    def forward_subheads(
            self, obj_center, obj_emb, key, value,
            img_dense_x2d_small, img_dense_x2d_mask_small,
            obj_strides, obj_img_inds, obj_labels, img_flips, img_shapes):
        head_emb_dim = self.embed_dims // self.num_heads
        num_obj = obj_img_inds.size(0)
        obj_flips = img_flips[obj_img_inds]
        if self.use_cls_emb:
            obj_emb = obj_emb + self.cls_emb[obj_labels]

        # deformable attn & sampling
        if obj_center.size(-1) > 2:  # offset_cls_agnostic=False
            obj_center = obj_center.reshape(
                num_obj, self.num_classes, 2
            )[torch.arange(num_obj, device=obj_center.device), obj_labels]
        batch_mlvl_positional_encodings = self.positional_encoding.points_to_enc(
            obj_center, img_shapes[obj_img_inds])
        query = self.query_proj(
            self.query_scale(obj_emb) + batch_mlvl_positional_encodings
        ).reshape(num_obj, self.num_heads, 1, head_emb_dim)
        output, v_samples, a_samples, mask_samples, x2d_samples = \
            self.attention_sampler(
                query, obj_emb, key, value,
                img_dense_x2d_small, img_dense_x2d_mask_small,
                obj_center, obj_strides, obj_img_inds)

        scale = self.scale_branch(output).exp()  # (num_obj, 2)
        score_pred = self.score_branch(output).squeeze(-1)  # (num_obj, )

        output_ = self.pred_fc(output)
        dim_enc = self.dim_branch(output_)
        if not self.dim_cls_agnostic:
            dim_enc = dim_enc.reshape(
                num_obj, self.num_classes, 3
            )[torch.arange(num_obj, device=dim_enc.device), obj_labels]
        dim_dec = self.dim_coder.decode(dim_enc, obj_labels)

        if self.pred_velo:
            velo = self.velo_branch(output_)
            velo_flip = velo.clone()
            velo_flip[obj_flips, 0] = -velo[obj_flips, 0]
        else:
            velo_flip = None
        if self.pred_attr:
            attr = self.attr_branch(output_)
        else:
            attr = None

        noc_list, w2d_list, x2d = self.forward_correspondence(
            v_samples, x2d_samples, mask_samples, query, obj_flips)

        return (query, scale, score_pred, dim_enc, dim_dec, velo_flip, attr,
                noc_list, w2d_list, x2d)

    def forward_test(self,
                     img_dense_x2d, img_dense_x2d_mask,
                     key, value,
                     topk_img_inds,
                     topk_labels,
                     topk_strides,
                     topk_obj_emb,
                     topk_center,
                     img_shapes, img_flips):
        img_dense_x2d_small = F.avg_pool2d(img_dense_x2d, self.output_stride, self.output_stride)
        img_dense_x2d_mask_small = F.avg_pool2d(img_dense_x2d_mask, self.output_stride, self.output_stride)

        (query, scale, score_pred, dim_enc, dim_dec, velo, attr,
         noc_list, w2d_list, x2d
         ) = self.forward_subheads(
            topk_center, topk_obj_emb, key, value,
            img_dense_x2d_small, img_dense_x2d_mask_small,
            topk_strides, topk_img_inds, topk_labels, img_flips, img_shapes)
        noc = noc_list[-1]
        w2d = w2d_list[-1]
        w2d *= scale[:, None, :]
        score_3d = score_pred.sigmoid()

        return noc, x2d, w2d, dim_dec, score_3d, velo, attr

    def test_post(self, x3d, x2d, w2d, ori_shapes, cam_intrinsic,
                  dim_decoded, img_inds, score, score_3d, labels, velo, attr,
                  mlvl_cls_score, mlvl_centerness):
        num_img = ori_shapes.size(0)

        ori_shapes_ = ori_shapes[img_inds]
        cam_intrinsic_ = cam_intrinsic[img_inds]

        self.camera.set_param(cam_intrinsic_, img_shape=ori_shapes_)
        self.cost_fun.set_param(x2d.detach(), w2d)

        mc_scoring_ratio = getattr(self.test_cfg, 'mc_scoring_ratio', 0.0)
        debug = getattr(self.test_cfg, 'debug', [])

        with default_timers['PnP time']:
            if mc_scoring_ratio > 0 or 'mc' in debug:
                pose_opt, _, _, pose_samples, pose_sample_logweights, _ = self.pnp.monte_carlo_forward(
                    x3d, x2d, w2d, self.camera, self.cost_fun, fast_mode=True)
                pose_sample_weights = pose_sample_logweights.softmax(dim=0)
            else:
                pose_opt = self.pnp(
                    x3d, x2d, w2d, self.camera, self.cost_fun, fast_mode=True)[0]
        if default_timers['PnP batch size'].enabled:
            default_timers['PnP batch size'].times.append(x3d.size(0))

        with default_timers['post-proc. time']:
            if mc_scoring_ratio > 0:
                if self.score_type == 'te':
                    sample_dev = (pose_samples[..., [0, 2]] - pose_opt[:, [0, 2]]).norm(dim=-1)
                    score_3d_mc = ((-sample_dev.log2() + 2.5) / 4).clamp(min=0, max=1)
                    score_3d_mc = (score_3d_mc * pose_sample_weights).sum(dim=0)  # (num_obj, )
                    score_3d = score_3d ** (1 - mc_scoring_ratio) * score_3d_mc ** mc_scoring_ratio
                else:
                    raise NotImplementedError
            if 'orient' in debug:
                orient_bins = getattr(self.test_cfg, 'orient_bins', 128)
                orient_grid = torch.linspace(
                    0, 2 * np.pi * (orient_bins - 1) / orient_bins,
                    steps=orient_bins, device=x3d.device)
                # (orient_bins, num_obj, 4)
                pose_grid = pose_opt[None].expand(orient_bins, -1, -1).clone()
                pose_grid[..., 3] = orient_grid[None, :, None]
                cost = evaluate_pnp(
                    x3d, x2d, w2d, pose_grid, self.camera, self.cost_fun, out_cost=True)[1]
                orient_logprob = cost.neg().log_softmax(dim=0) + np.log(orient_bins / (2 * np.pi))
                orient_logprob = orient_logprob.transpose(1, 0).cpu().numpy()
            # convert to numpy output
            # (num_obj, num_sample, 4), (num_obj, num_sample)
            if 'mc' in debug:
                pose_samples = pose_samples.transpose(1, 0).cpu().numpy()
                pose_sample_weights = pose_sample_weights.transpose(1, 0).cpu().numpy()
            if 'pts' in debug:
                x3d = x3d.cpu().numpy()
                x2d = x2d.cpu().numpy()
                w2d = w2d.cpu().numpy()

            bbox_2d, bbox_2d_mask = bboxes_3d_to_2d(
                torch.cat((dim_decoded, pose_opt), dim=-1),
                cam_intrinsic_,
                ori_shapes_)
            bbox_2d_result, bbox_3d_result = self.get_bbox_3d_result(
                dim_decoded, pose_opt, score, score_3d, labels, bbox_2d, bbox_2d_mask,
                num_img, img_inds, velo=velo, attr=attr, to_np=True)

            results = []

            for i, (bbox_2d_per_img, bbox_3d_per_img) in enumerate(
                    zip(bbox_2d_result, bbox_3d_result)):
                result_dict = dict(bbox_results=bbox_2d_per_img,
                                   bbox_3d_results=bbox_3d_per_img)
                if 'mc' in debug or 'orient' in debug or 'pts' in debug:
                    pose_samples_multi_cls = []
                    pose_sample_weights_multi_cls = []
                    orient_logprob_multi_cls = []
                    x3d_multi_cls = []
                    x2d_multi_cls = []
                    w2d_multi_cls = []
                    for bboxes_3d in bbox_3d_per_img:
                        keep_inds = bboxes_3d[:, -1]
                        keep_inds = keep_inds.round().astype(np.int64)
                        if 'mc' in debug:
                            pose_samples_multi_cls.append(pose_samples[keep_inds])
                            pose_sample_weights_multi_cls.append(pose_sample_weights[keep_inds])
                        if 'orient' in debug:
                            orient_logprob_multi_cls.append(orient_logprob[keep_inds])
                        if 'pts' in debug:
                            x3d_multi_cls.append(x3d[keep_inds])
                            x2d_multi_cls.append(x2d[keep_inds])
                            w2d_multi_cls.append(w2d[keep_inds])
                    if 'mc' in debug:
                        result_dict.update(
                            pose_samples=pose_samples_multi_cls,
                            pose_sample_weights=pose_sample_weights_multi_cls)
                    if 'orient' in debug:
                        result_dict.update(
                            orient_logprob=orient_logprob_multi_cls)
                    if 'pts' in debug:
                        result_dict.update(
                            x3d=x3d_multi_cls,
                            x2d=x2d_multi_cls,
                            w2d=w2d_multi_cls)
                results.append(result_dict)
            if 'score' in debug:
                h = mlvl_cls_score[0].size(-2) * self.strides[self.det_lvl_range[0]]
                w = mlvl_cls_score[0].size(-1) * self.strides[self.det_lvl_range[0]]
                scores = torch.zeros(
                    (num_img, self.num_classes, h, w),
                    device=mlvl_cls_score[0].device, dtype=torch.float32)
                for lvl_cls_score, lvl_centerness, stride in zip(
                        mlvl_cls_score, mlvl_centerness, self.detector.strides):
                    lvl_score = lvl_cls_score.sigmoid() * lvl_centerness.sigmoid()
                    scores += F.interpolate(lvl_score, scale_factor=stride, mode='nearest')[..., :h, :w]
                for result, score in zip(results, scores.cpu().numpy()):
                    result.update(score=score)

        return results

    def simple_test(self,
                    mlvl_feats,
                    img_metas,
                    img_dense_x2d=None,
                    img_dense_x2d_mask=None,
                    cam_intrinsic=None,
                    **kwargs):
        img_dense_x2d, img_dense_x2d_mask, cam_intrinsic = img_dense_x2d[0], img_dense_x2d_mask[0], cam_intrinsic[0]
        cam_intrinsic = torch.stack(cam_intrinsic, dim=0)
        img_shapes = cam_intrinsic.new_tensor([img_meta['img_shape'][:2] for img_meta in img_metas])
        ori_shapes = cam_intrinsic.new_tensor([img_meta['ori_shape'][:2] for img_meta in img_metas])
        img_flips = cam_intrinsic.new_tensor(
            [img_meta['flip'] for img_meta in img_metas], dtype=torch.bool)

        (mlvl_cls_score, mlvl_center, mlvl_centerness, mlvl_obj_emb, _,
         key, value) = self.forward_det_dense(mlvl_feats, img_metas)

        with default_timers['FCOS head post-proc. time']:
            topk_img_inds, topk_scores, topk_labels, topk_strides, topk_obj_emb, topk_center = \
                self.detector.get_preds(mlvl_cls_score, mlvl_centerness, mlvl_obj_emb, mlvl_center)

        with default_timers['point & object subheads time']:
            noc, x2d, w2d, dim_decoded, score_3d, velo, attr = self.forward_test(
                img_dense_x2d, img_dense_x2d_mask,
                key, value,
                topk_img_inds,
                topk_labels,
                topk_strides,
                topk_obj_emb,
                topk_center,
                img_shapes, img_flips)
            x3d = noc * dim_decoded[:, None]

        return self.test_post(
            x3d, x2d, w2d, ori_shapes, cam_intrinsic,
            dim_decoded, topk_img_inds, topk_scores, score_3d, topk_labels, velo, attr,
            mlvl_cls_score, mlvl_centerness)

    def aug_test(self,
                 mlvl_feats,
                 img_metas,
                 img_dense_x2d=None,
                 img_dense_x2d_mask=None,
                 cam_intrinsic=None):
        # Note: support only horizontal flip aug
        cam_intrinsic = torch.stack(cam_intrinsic[0], dim=0)
        img_shapes = cam_intrinsic.new_tensor([img_meta['img_shape'][:2] for img_meta in img_metas[0]])
        ori_shapes = cam_intrinsic.new_tensor([img_meta['ori_shape'][:2] for img_meta in img_metas[0]])
        img_flips_aug = [cam_intrinsic.new_tensor(
            [img_meta['flip'] for img_meta in img_metas_single], dtype=torch.bool)
            for img_metas_single in img_metas]

        (mlvl_cls_score_aug, mlvl_center_aug, mlvl_centerness_aug, mlvl_obj_emb_aug, _,
         key_aug, value_aug) = multi_apply(self.forward_det_dense, mlvl_feats, img_metas)

        mlvl_cls_score = [(a + b.flip(-1)) / 2 for a, b in zip(*mlvl_cls_score_aug)]
        mlvl_centerness = [(a + b.flip(-1)) / 2 for a, b in zip(*mlvl_centerness_aug)]

        (topk_img_inds, topk_scores, topk_labels, topk_strides,
         topk_obj_emb_0, topk_obj_emb_1,
         topk_center_0, topk_center_1) = self.detector.get_preds(
            mlvl_cls_score, mlvl_centerness,
            mlvl_obj_emb_aug[0], [b.flip(-1) for b in mlvl_obj_emb_aug[1]],
            mlvl_center_aug[0], [b.flip(-1) for b in mlvl_center_aug[1]])

        (noc_aug, x2d_aug, w2d_aug,
         dim_decoded_aug, score_3d_aug, velo_aug, attr_aug) = multi_apply(
            self.forward_test,
            img_dense_x2d, img_dense_x2d_mask,
            key_aug, value_aug,
            [topk_img_inds, topk_img_inds],
            [topk_labels, topk_labels],
            [topk_strides, topk_strides],
            [topk_obj_emb_0, topk_obj_emb_1],
            [topk_center_0, topk_center_1],
            [img_shapes, img_shapes], img_flips_aug)
        noc = torch.cat(noc_aug, dim=1)
        x2d = torch.cat(x2d_aug, dim=1)
        w2d = torch.cat(w2d_aug, dim=1)
        dim_decoded = (dim_decoded_aug[0] + dim_decoded_aug[1]) / 2
        score_3d = (score_3d_aug[0] + score_3d_aug[1]) / 2
        x3d = noc * dim_decoded[:, None]
        if velo_aug[0] is not None:
            velo = (velo_aug[0] + velo_aug[1]) / 2
        else:
            velo = None
        if attr_aug[0] is not None:
            attr = (attr_aug[0] + attr_aug[1]) / 2
        else:
            attr = None

        return self.test_post(
            x3d, x2d, w2d, ori_shapes, cam_intrinsic,
            dim_decoded, topk_img_inds, topk_scores, score_3d, topk_labels, velo, attr,
            mlvl_cls_score, mlvl_centerness)

    def extract_rois(self, rois, img_dense_x2d, key, value, roi_shape=(28, 28)):
        """
        Args:
            rois: shape (bs, 5)
            img_dense_x2d: shape (num_img, 2, h_img, w_img)
            key: (num_img, embed_dim, h, w)
            value: (num_img, embed_dim, h, w)

        Returns:
            Tuple[Tensor]:
                x2d_roi: (bs, 2, rh, rw)
                key_roi: (bs, embed_dim, rh, rw)
                value_roi: (bs, embed_dim, rh, rw)
        """
        spatial_scale = 1.0 / self.output_stride
        rh, rw = roi_shape
        x2d_roi = roi_align_wrapper(
            img_dense_x2d, rois, (rh, rw), 1.0, 0, 'avg', True)
        key_roi = roi_align_wrapper(
            key, rois, (rh, rw), spatial_scale, 0, 'avg', True)
        value_roi = roi_align_wrapper(
            value, rois, (rh, rw), spatial_scale, 0, 'avg', True)
        return x2d_roi, key_roi, value_roi

    def forward_train(self,
                      mlvl_feats,
                      img_metas,
                      gt_bboxes,
                      gt_labels=None,
                      gt_bboxes_ignore=None,
                      gt_bboxes_3d=None,
                      gt_x3d=None,
                      gt_x2d=None,
                      gt_attr=None,
                      gt_velo=None,
                      img_dense_x2d=None,
                      img_dense_x2d_mask=None,
                      cam_intrinsic=None):
        # ===== prepare img metas and g.t. =====
        device = mlvl_feats[0].device
        cam_intrinsic = torch.stack(cam_intrinsic, dim=0)
        img_shapes = cam_intrinsic.new_tensor([img_meta['img_shape'][:2] for img_meta in img_metas])
        ori_shapes = cam_intrinsic.new_tensor([img_meta['ori_shape'][:2] for img_meta in img_metas])
        img_flips = cam_intrinsic.new_tensor(
            [img_meta['flip'] for img_meta in img_metas], dtype=torch.bool)

        gt_bboxes_ = torch.cat(gt_bboxes, dim=0)
        gt_bboxes_3d_ = torch.cat(gt_bboxes_3d, dim=0)
        gt_labels_ = torch.cat(gt_labels, dim=0)
        img_dense_x2d_small = F.avg_pool2d(img_dense_x2d, self.output_stride, self.output_stride)
        img_dense_x2d_mask_small = F.avg_pool2d(img_dense_x2d_mask, self.output_stride, self.output_stride)

        gt_img_inds = []
        for i, gt_bboxes_3d_single in enumerate(gt_bboxes_3d):
            gt_img_inds += [i] * gt_bboxes_3d_single.size(0)
        gt_img_inds = torch.tensor(gt_img_inds, device=device, dtype=torch.long)

        # ===== get center targets and filter g.t. boxes =====
        (centers2d_, gt_bboxes_, centers2d, gt_bboxes,
         valid_mask, num_obj_per_img) = self.center_target.get_centers_2d(
            gt_bboxes_, gt_bboxes_3d_, gt_img_inds, img_dense_x2d_small, img_dense_x2d_mask_small,
            cam_intrinsic, ori_shapes.max(dim=0)[0])

        gt_bboxes_3d_ = gt_bboxes_3d_[valid_mask]
        gt_labels_ = gt_labels_[valid_mask]
        if self.pred_velo:
            gt_velo_ = torch.cat(gt_velo, dim=0)[valid_mask]
        if self.pred_attr:
            gt_attr_ = torch.cat(gt_attr, dim=0)[valid_mask]
        if self.loss_regr is not None:
            assert gt_x3d is not None and gt_x2d is not None
            gt_x3d_ = []
            gt_x2d_ = []
            for gt_x3d_single, gt_x2d_single, valid_mask_single in zip(
                    list_flatten(gt_x3d), list_flatten(gt_x2d), valid_mask):
                if valid_mask_single:
                    gt_x3d_.append(gt_x3d_single)
                    gt_x2d_.append(gt_x2d_single)

        gt_labels = gt_labels_.split(num_obj_per_img, dim=0)
        rois = bbox2roi(gt_bboxes)
        gt_img_inds = rois[:, 0].to(torch.int64)
        gt_flips_ = img_flips[gt_img_inds]

        # ===== FCOS detector and dense head =====
        (mlvl_cls_score, mlvl_center, mlvl_centerness, mlvl_obj_emb, mlvl_points,
         key, value) = self.forward_det_dense(mlvl_feats, img_metas)

        # ===== detector loss =====
        mlvl_labels, mlvl_centerness_targets, mlvl_gt_inds = self.detector.get_targets(
            mlvl_points, gt_bboxes, gt_labels, centers2d)
        flatten_cls_scores = torch.cat([
            cls_score.permute(0, 2, 3, 1).reshape(-1, self.num_classes)
            for cls_score in mlvl_cls_score], dim=0)
        flatten_center = torch.cat([
            center.permute(0, 2, 3, 1).reshape(-1, center.size(1))
            for center in mlvl_center], dim=0)
        flatten_centerness = torch.cat([
            centerness.permute(0, 2, 3, 1).reshape(-1)
            for centerness in mlvl_centerness], dim=0)
        flatten_obj_emb = torch.cat([
            obj_emb.permute(0, 2, 3, 1).reshape(-1, self.embed_dims)
            for obj_emb in mlvl_obj_emb], dim=0)
        flatten_labels = torch.cat(mlvl_labels, dim=0)
        flatten_centerness_targets = torch.cat(mlvl_centerness_targets, dim=0)
        flatten_gt_inds = torch.cat(mlvl_gt_inds, dim=0)
        flatten_strides = torch.cat(
            [torch.full_like(lvl_labels, stride)
             for lvl_labels, stride in zip(mlvl_labels, self.detector.strides)], dim=0)

        losses = self.detector.loss(  # detector losses
            flatten_cls_scores,
            flatten_center,
            flatten_centerness,
            flatten_labels,
            flatten_gt_inds,
            flatten_centerness_targets,
            centers2d_,
            gt_bboxes_)

        # ===== obj sampling =====
        num_img = key.size(0)
        num_obj_samples = getattr(self.train_cfg, 'num_obj_samples_per_img', 48) * num_img
        fg_mask = flatten_labels < self.num_classes
        (sample_gt_inds, sample_weights, sample_uniform_weights,
         obj_emb_samples, center_samples, stride_samples) = obj_sampler(
            num_obj_samples, fg_mask, flatten_centerness_targets, flatten_gt_inds,
            flatten_obj_emb, flatten_center, flatten_strides,
            uniform_mix_ratio=getattr(self.train_cfg, 'uniform_mix_ratio', 0.5))
        sample_img_inds = gt_img_inds[sample_gt_inds]
        sample_labels = gt_labels_[sample_gt_inds]
        ori_shape_samples = ori_shapes[sample_img_inds]
        cam_intrinsic_samples = cam_intrinsic[sample_img_inds]
        bbox_3d_targets = gt_bboxes_3d_[sample_gt_inds]  # (num_obj_actual, 7)
        num_obj_actual = sample_gt_inds.size(0)

        # ===== subheads =====
        (query_samples, scale, score_pred, dim_pred, dim_decoded, velo, attr,
         noc_list, w2d_list, x2d) = self.forward_subheads(
            center_samples, obj_emb_samples, key, value,
            img_dense_x2d_small, img_dense_x2d_mask_small,
            stride_samples, sample_img_inds, sample_labels, img_flips, img_shapes)

        # ===== dim loss =====
        dim_targets = self.dim_coder.encode(
            bbox_3d_targets[:, :3], sample_labels)  # (num_obj_actual, 3)
        loss_dim = self.loss_dim(
            dim_pred, dim_targets,
            weight=sample_weights[:, None], avg_factor=num_obj_samples * 3)

        # ===== pose loss =====
        norm_factor = (scale * sample_weights[:, None]).sum() / max(scale.size(0) * 2, 1)
        self.camera.set_param(cam_intrinsic_samples, img_shape=ori_shape_samples)
        for stage_id, (noc, w2d) in enumerate(zip(noc_list, w2d_list)):
            x3d = noc * dim_decoded[:, None]
            w2d_scaled = w2d * scale[:, None, :]
            self.cost_fun.set_param(x2d.detach(), w2d_scaled)
            _, _, _, _, pose_sample_logweights, cost_tgt = self.pnp.monte_carlo_forward(
                x3d, x2d, w2d_scaled, self.camera, self.cost_fun,
                pose_init=bbox_3d_targets[:, 3:], force_init_solve=True)
            loss_pose = self.loss_pose[stage_id](
                pose_sample_logweights, cost_tgt, norm_factor,
                weight=sample_weights,
                avg_factor=num_obj_samples)
            losses.update({'loss_pose_{}'.format(stage_id): loss_pose})

        # ===== 3d score loss & derivative regularization loss =====
        if num_obj_actual > 0:
            self.cost_fun.delta = self.cost_fun.delta.detach()
            pose_opt, _, _, pose_opt_plus = self.pnp(
                noc * dim_decoded[:, None].detach(),
                x2d,
                w2d * scale[:, None, :].detach(),
                self.camera, self.cost_fun,
                with_pose_opt_plus=True)
            if self.score_type == 'iou':
                ious = bbox3d_overlaps_aligned_torch(
                    torch.cat((pose_opt[:, :3], dim_decoded, pose_opt[:, 3:]), dim=-1),
                    bbox_3d_targets[:, [3, 4, 5, 0, 1, 2, 6]]).reshape(-1)
                metric = {'mean_iou': (ious * sample_weights).sum() / num_obj_actual}
                score_targets = (2 * ious - 0.5).clamp(min=0, max=1)
            else:  # self.score_type == 'te'
                te = (pose_opt[:, [0, 2]] - bbox_3d_targets[:, [3, 5]]).norm(dim=1)
                metric = {'ate': (te * sample_weights).sum() / num_obj_actual}
                score_targets = ((-te.log2() + 2.5) / 4).clamp(min=0, max=1)
            loss_score = self.loss_score(
                score_pred, score_targets.detach(),
                weight=sample_uniform_weights, avg_factor=num_obj_samples)
            loss_reg_pos = self.loss_reg_pos(
                (pose_opt_plus[:, :3] - bbox_3d_targets[:, 3:6]).norm(dim=-1), -1,
                weight=sample_weights, avg_factor=num_obj_samples)
            loss_reg_orient = self.loss_reg_orient(
                pose_opt_plus[:, 3], bbox_3d_targets[:, 6],
                weight=sample_weights, avg_factor=num_obj_samples)
        else:
            if self.score_type == 'iou':
                metric = {'mean_iou': torch.tensor(0.0, device=device)}
            else:
                metric = {'ate': torch.tensor(0.0, device=device)}
            loss_score = score_pred.sum()
            loss_reg_pos = noc.sum() + x2d.sum() + w2d.sum()
            loss_reg_orient = loss_reg_pos
        losses.update(metric)

        losses.update({'loss_dim': loss_dim,
                       'loss_score': loss_score,
                       'loss_reg_pos': loss_reg_pos,
                       'loss_reg_orient': loss_reg_orient,
                       'norm_factor': self.loss_pose[-1].norm_factor.detach()})

        # ===== auxiliary loss =====
        if self.loss_proj is not None or self.loss_regr is not None:
            # roi sampling
            gt_active_inds, sample_gt_act_inds = torch.unique(
                sample_gt_inds, return_inverse=True, sorted=False)
            num_gt_act = gt_active_inds.size(0)
            gt_flips_act = gt_flips_[gt_active_inds]
            gt_bboxes_3d_act = gt_bboxes_3d_[gt_active_inds]
            gt_img_inds_act = gt_img_inds[gt_active_inds]
            rois_act = rois[gt_active_inds]

            rh, rw = getattr(self.train_cfg, 'roi_shape', (28, 28))
            x2d_roi, key_roi, value_roi = self.extract_rois(
                rois_act, img_dense_x2d, key, value, roi_shape=(rh, rw))
            # (num_gt_act, 1, rh * rw, 2)
            x2d_tgt = x2d_roi.reshape(num_gt_act, 1, 2, rh * rw).transpose(-1, -2)
            cam_intrinsic_gt = cam_intrinsic[gt_img_inds_act]
            ori_shapes_gt = ori_shapes[gt_img_inds_act]

            # dense regression
            # (num_gt_act, num_head, rh * rw, 5 or 6)
            regr_res = self.corr_regs[0](
                value_roi.reshape(num_gt_act, self.embed_dims, rh * rw).transpose(-1, -2)
            ).reshape(num_gt_act, rh * rw, self.num_heads, 5).transpose(2, 1)
            noc_roi, logstd_roi = regr_res.split([3, 2], dim=-1)
            noc_roi = noc_roi.clone()
            noc_roi[gt_flips_act, :, :, 2] = -noc_roi[gt_flips_act, :, :, 2]  # flip correction

            # (num_gt_act, num_obj_actual)
            sample_to_act = torch.arange(num_gt_act, device=device)[:, None] == sample_gt_act_inds
            sample_to_act = F.normalize(sample_to_act * sample_weights, p=1, dim=-1)
            dim_decoded_act = (sample_to_act @ dim_decoded).detach()  # (num_gt_act, 3)

            x3d_roi = noc_roi * dim_decoded_act[:, None, None, :]  # (num_gt_act, num_head, rh * rw, 3)
            x2d_proj = project_to_image(
                x3d_roi.reshape(num_gt_act, self.num_heads * rh * rw, 3), gt_bboxes_3d_act[:, 3:],
                cam_intrinsic_gt,
                ori_shapes_gt,
                z_min=self.camera.z_min,
                allowed_border=self.camera.allowed_border
            ).reshape(num_gt_act, self.num_heads, rh * rw, 2)
            proj_error = self.proj_error_coder.encode(
                x2d_proj - x2d_tgt,
                gt_bboxes_3d_act[:, None, 5:6],
                gt_bboxes_3d_act[:, None, :3],
                cam_intrinsic_gt[:, 0, 0, None, None]
            ).reshape(num_gt_act, self.num_heads, rh, rw, 2)
            # reprojection loss
            head_emb_dim = self.embed_dims // self.num_heads
            query_gt_act = (sample_to_act @ query_samples.flatten(1)).reshape(
                num_gt_act, self.num_heads, 1, head_emb_dim)
            # (num_gt_act, num_head, 1, h_out * w_out) = (num_gt_act, num_head, 1, head_emb_dim)
            # @ (num_gt_act, num_head, head_emb_dim, h_out * w_out)
            attn_gt_act = query_gt_act @ key_roi.reshape(
                num_gt_act, self.num_heads, head_emb_dim, rh * rw) / np.sqrt(head_emb_dim)
            attn_gt_act = attn_gt_act.reshape(num_gt_act, self.num_heads, rh, rw)
            attn_gt_act_logsoftmax = logsoftmax_across_rois(
                attn_gt_act, rois_act, extra_dim=1)

            # proj loss
            if self.loss_proj is not None:
                losses.update(
                    {'loss_proj': self.loss_proj(
                        proj_error, 0,
                        rois=rois_act,
                        logstd=logstd_roi.reshape(proj_error.size()),
                        logmixweight=attn_gt_act_logsoftmax,
                        avg_factor=max(reduce_mean(proj_error.new_tensor(num_gt_act)), 1.0) * rh * rw)})

            # regr loss
            if self.loss_regr is not None:
                x3d_tgt = x2d_roi.new_zeros((num_gt_act, rh * rw, 4))
                x2d_roi_start = x2d_roi[..., 0, 0]  # (num_gt_act, 2)
                x2d_roi_range = x2d_roi[..., -1, -1] - x2d_roi_start
                roi_wh = x2d_roi.new_tensor([rw, rh])
                for i, act_ind in enumerate(gt_active_inds):
                    gt_x3d_act = gt_x3d_[act_ind]  # (pn, 3)
                    gt_x3d_act = F.pad(gt_x3d_act, [0, 1], mode='constant', value=1.0)
                    gt_x2d_act = gt_x2d_[act_ind]  # (pn, 2)
                    roi_inds = (((gt_x2d_act - x2d_roi_start[i]) / x2d_roi_range[i]).clamp(min=0, max=1) * (
                        roi_wh - 1)).round().long()  # (pn, 2)
                    roi_inds = roi_inds[:, 1] * rw + roi_inds[:, 0]  # (pn, )
                    x3d_tgt[i].scatter_(dim=0, index=roi_inds[:, None].expand(-1, 4),
                                        src=gt_x3d_act, reduce='add')
                x3d_tgt_weight = x3d_tgt[..., 3].clamp(max=1.0)  # 0, 1 binary mask (num_gt_act, rh * rw)
                x3d_tgt = x3d_tgt[..., :3] / x3d_tgt[..., 3:].clamp(min=1.0)  # (num_gt_act, rh * rw, 3)

                max_dim = gt_bboxes_3d_act[:, :3].max(dim=-1)[0]
                # (num_gt_act, num_head, rh * rw)
                regr_error = (x3d_roi - x3d_tgt[:, None]).norm(dim=-1) / max_dim[:, None, None]
                x3d_weight = attn_gt_act.reshape(
                    num_gt_act, self.num_heads, rh * rw
                ).softmax(dim=1) * x3d_tgt_weight[:, None, :]
                losses.update(
                    {'loss_regr': self.loss_regr(
                      regr_error, -1, weight=x3d_weight,
                      avg_factor=reduce_mean(x3d_weight.sum()).clamp(min=1e-4))})

        # ===== velo & attr loss =====
        if self.pred_velo:
            velo_targets = gt_velo_[sample_gt_inds]
            nan_mask = velo_targets.isnan()
            velo_targets.masked_fill_(nan_mask, 0.0)
            velo_weights = sample_weights[:, None] * (~nan_mask)
            losses.update(
                {'loss_velo': self.loss_velo(
                    velo, velo_targets,
                    weight=velo_weights,
                    avg_factor=reduce_mean(velo_weights.sum().clamp(min=1.0)))})
        if self.pred_attr:
            attr_targets = gt_attr_[sample_gt_inds]
            losses.update(
                {'loss_attr': self.loss_attr(
                    attr, attr_targets, weight=sample_weights, avg_factor=num_obj_samples)})

        return losses

    def get_bbox_3d_result(self, dimensions, pose, scores, scores_3d, labels, bbox_2d, bbox_2d_mask,
                           num_img, img_inds, velo=None, attr=None, to_np=False):
        device = dimensions.device
        bbox_3d = [dimensions, pose, (scores * scores_3d).unsqueeze(1)]
        if velo is not None and attr is not None:
            bbox_3d += [velo, attr]
        bbox_3d.append(
            torch.arange(dimensions.size(0), dtype=torch.float32, device=device).unsqueeze(1))
        bbox_3d = torch.cat(bbox_3d, dim=1)

        bbox_2d = bbox_2d[bbox_2d_mask]
        bbox_3d = bbox_3d[bbox_2d_mask]
        labels = labels[bbox_2d_mask]
        img_inds = img_inds[bbox_2d_mask]

        if bbox_2d.size(0) > 0:
            nms_batch_inds = img_inds * self.num_classes + labels
            bbox_2d, keep_inds_nms2d = batched_nms(
                bbox_2d, bbox_3d[:, 7].contiguous(), nms_batch_inds,
                nms_cfg=getattr(self.test_cfg, 'nms_iou2d', dict(type='nms', iou_threshold=0.8)))
            bbox_2d[:, 4] = scores[bbox_2d_mask][keep_inds_nms2d]  # 2D score
            bbox_3d = bbox_3d[keep_inds_nms2d]
            labels = labels[keep_inds_nms2d]
            img_inds = img_inds[keep_inds_nms2d]

            nms_batch_inds = img_inds * self.num_classes + labels
            bbox_3d, keep_inds_nmsbev = batched_bev_nms(
                bbox_3d, nms_batch_inds,
                nms_thr=getattr(self.test_cfg, 'nms_ioubev_thr', 0.25))
            bbox_2d = bbox_2d[keep_inds_nmsbev]
            labels = labels[keep_inds_nmsbev]
            img_inds = img_inds[keep_inds_nmsbev]
        else:
            bbox_2d = bbox_2d.reshape(0, 5)

        # (num_cls, num_obj)
        label_masks = labels == torch.arange(self.num_classes, device=device)[:, None]
        # (num_img, num_obj)
        img_masks = img_inds == torch.arange(num_img, device=device)[:, None]
        # (num_img, num_cls, num_obj)
        img_label_masks = img_masks[:, None, :] & label_masks
        if to_np:
            bbox_2d = bbox_2d.cpu().numpy()
            bbox_3d = bbox_3d.cpu().numpy()
            img_label_masks = img_label_masks.cpu().numpy()

        bbox_2d_result = []
        bbox_3d_result = []
        for i in range(num_img):
            bbox_2d_per_img = []
            bbox_3d_per_img = []
            for j in range(self.num_classes):
                mask = img_label_masks[i, j]
                bbox_2d_per_img.append(bbox_2d[mask])
                bbox_3d_per_img.append(bbox_3d[mask])
            bbox_2d_result.append(bbox_2d_per_img)
            bbox_3d_result.append(bbox_3d_per_img)
        return bbox_2d_result, bbox_3d_result

    def loss(self, **kwargs):
        pass

    def get_bboxes(self, **kwargs):
        pass


def obj_sampler(num_obj_samples,
                fg_mask,
                flatten_centerness_targets,
                flatten_gt_inds,
                *args,
                uniform_mix_ratio=0.5,
                eps=1e-5):
    """
    Sample from the dense objects to reduce the Monte Carlo overhead in training the EPro-PnP

    Args:
        num_obj_samples (int): Target sample count. If fg_mask_count == 0, the actual count
            will be zero
        fg_mask (torch.Tensor): Shape (num_total_point, ), bool
        flatten_centerness_targets (torch.Tensor): Shape (num_total_point, )
        flatten_gt_inds (torch.Tensor): Shape (num_total_point, ), int64
        *args (Tuple[torch.Tensor]): Shape (num_total_point, *), optional

    Returns:
        Tuple[torch.Tensor]:
            sample_gt_inds: Shape (num_obj_sample, ), int64
            sample_weights: Shape (num_obj_sample, ), balanced sample weights, where the
                cummulative weights of samples assiciated with each g.t. are equal
            sample_uniform_weights: Shape (num_obj_sample, ), balanced uniform sample weights,
                where the cummulative weights of samples assiciated with each g.t. are equal,
                and the weights of samples of the same g.t. are uniform. If uniform_mix_ratio
                == 0.0, then sample_uniform_weights and sample_weights are equivalent
            optional_outputs: Shape (num_obj_sample, *)
    """
    fg_mask_count = fg_mask.count_nonzero()

    num_uniform_samples = min(int(round(num_obj_samples * uniform_mix_ratio)), fg_mask_count)
    num_replace_samples = num_obj_samples - num_uniform_samples
    uniform_mix_ratio = num_uniform_samples / num_obj_samples

    # (num_total_point, )
    prob = flatten_centerness_targets * fg_mask
    prob /= prob.sum().clamp(min=eps)
    prob_uniform = fg_mask / fg_mask_count.clamp(min=1)  # uniform
    prob_mix = prob_uniform * uniform_mix_ratio + prob * (1 - uniform_mix_ratio)
    
    if fg_mask_count > 0:
        sample_point_inds_uniform = torch.multinomial(
            prob_uniform, num_uniform_samples, replacement=False)
        sample_point_inds_replace = torch.multinomial(
            prob, num_replace_samples, replacement=True)
        sample_point_inds = torch.cat(
            (sample_point_inds_uniform, sample_point_inds_replace), dim=0)
        sample_gt_inds = flatten_gt_inds[sample_point_inds]
        sample_prob_weights = prob[sample_point_inds] / prob_mix[sample_point_inds]
        
        gt_mask = sample_gt_inds == torch.arange(
            sample_gt_inds.max() + 1, device=sample_gt_inds.device)[:, None]  # (num_gt, num_obj_sample)
        
        gt_prob_sum = (sample_prob_weights * gt_mask).sum(dim=1)
        gt_weights = 1 / gt_prob_sum.clamp(min=eps)
        sample_weights = sample_prob_weights * gt_weights[sample_gt_inds]
        sample_weights /= sample_weights.mean().clamp(min=eps)
       
        gt_counts = torch.count_nonzero(gt_mask, dim=1)
        gt_weights = 1 / gt_counts.clamp(min=1)
        sample_uniform_weights = gt_weights[sample_gt_inds]
        sample_uniform_weights /= sample_uniform_weights.mean().clamp(min=eps)
        
    else:
        sample_point_inds = sample_gt_inds = flatten_gt_inds[[]]
        sample_weights = sample_uniform_weights = prob[[]]

    ret = (sample_gt_inds, sample_weights, sample_uniform_weights)
    extra_ret = []
    for arg in args:
        extra_ret.append(arg[sample_point_inds])
    return ret + tuple(extra_ret)


def roi_align_wrapper(x, rois, output_size, spatial_scale=1.0,
                      sampling_ratio=0, pool_mode='avg', aligned=True):
    if rois.size(0) > 0:
        output = roi_align(x, rois, output_size, spatial_scale,
                           sampling_ratio, pool_mode, aligned)
    else:
        output = x.new_zeros((0, x.size(1)) + output_size)
        if x.requires_grad:
            output += x.sum()
    return output


def list_flatten(t):
    return [item for sublist in t for item in sublist]
