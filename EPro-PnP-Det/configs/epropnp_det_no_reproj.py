"""
Copyright (C) 2010-2022 Alibaba Group Holding Limited.
"""

model = dict(
    type='EProPnPDet',
    backbone=dict(
        type='ResNet',
        depth=101,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='pytorch',
        dcn=dict(type='DCNv2', deform_groups=1, fallback_on_stride=False),
        stage_with_dcn=(False, False, True, True),
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet101')),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=6,
        add_extra_convs='on_output',
        relu_before_extra_convs=True),
    bbox_head=dict(
        type='DefromPnPHead',
        num_classes=10,
        in_channels=256,
        strides=(4, 8, 16, 32, 64, 128),
        output_stride=4,
        dense_lvl_range=(0, 4),
        det_lvl_range=(1, 6),
        dense_channels=256,
        embed_dims=256,
        num_heads=8,
        num_points=32,
        detector=dict(
            type='FCOSEmbHead',
            feat_channels=256,
            stacked_convs=2,
            emb_channels=256,
            strides=[8, 16, 32, 64, 128],
            regress_ranges=((-1, 48), (48, 96), (96, 192), (192, 384),
                            (384, 1e8)),
            loss_cls=dict(
                type='FocalLoss',
                use_sigmoid=True,
                gamma=2.0,
                alpha=0.25,
                loss_weight=1.0),
            loss_rp=dict(  # reference point loss
                type='SmoothL1LossMod', beta=1.0, loss_weight=1.0),
            loss_centerness=dict(
                type='CrossEntropyLoss',
                use_sigmoid=True,
                loss_weight=1.0)),
        attention_sampler=dict(
            type='DeformableAttentionSampler',
            embed_dims=256,
            num_heads=8,
            num_points=32,
            stride=4),
        center_target=dict(
            type='VolumeCenter',
            output_stride=4,
            render_stride=4,
            min_box_size=4.0),
        dim_coder=dict(
            type='MultiClassLogDimCoder',
            target_means=[
                (4.62, 1.73, 1.96),
                (6.94, 2.84, 2.52),
                (12.56, 3.89, 2.94),
                (11.22, 3.50, 2.95),
                (6.68, 3.21, 2.85),
                (1.70, 1.29, 0.61),
                (2.11, 1.46, 0.78),
                (0.73, 1.77, 0.67),
                (0.41, 1.08, 0.41),
                (0.50, 0.99, 2.52)],
            target_stds=[
                (0.46, 0.24, 0.16),
                (2.11, 0.84, 0.45),
                (4.50, 0.77, 0.54),
                (2.06, 0.49, 0.33),
                (3.23, 0.93, 1.07),
                (0.26, 0.35, 0.16),
                (0.33, 0.29, 0.17),
                (0.19, 0.19, 0.14),
                (0.14, 0.27, 0.13),
                (0.17, 0.15, 0.62)]),
        positional_encoding=dict(
            type='SinePositionalEncodingMod',
            num_feats=128,
            normalize=True,
            offset=-0.5),
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
        loss_pose=dict(  # main pose loss
            type='MonteCarloPoseLoss',
            loss_weight=0.15,
            momentum=0.01),
        loss_proj=None,
        loss_dim=dict(  # dimension (size) loss
            type='SmoothL1LossMod',
            loss_weight=1.0),
        loss_regr=None,
        loss_score=dict(  # 3D score loss
            type='CrossEntropyLoss',
            use_sigmoid=True,
            loss_weight=1.0),
        loss_reg_pos=dict(  # derivative regularization loss for position
            type='SmoothL1LossMod',
            beta=1.0,
            loss_weight=0.05),
        loss_reg_orient=dict(  # derivative regularization loss for orientation
            type='CosineAngleLoss',
            loss_weight=0.05),
        loss_velo=dict(
            type='SmoothL1LossMod',
            loss_weight=0.05),
        loss_attr=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=0.5),
        pred_velo=True,
        pred_attr=True,
        num_attrs=9,
        score_type='te'),
    train_cfg=dict(
        num_obj_samples_per_img=48,
        roi_shape=(28, 28)),  # RoI shape for the reprojection-based auxiliary loss
    test_cfg=dict(
        override_cfg={'pnp.solver.num_iter': 5},
        mc_scoring_ratio=0.0,  # 1.0 for Monte Carlo scoring
        nms_iou2d=dict(type='nms', iou_threshold=0.8),
        nms_ioubev_thr=0.25))
dataset_type = 'NuScenes3DDataset'
data_root = 'data/nuscenes/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile3D',
         with_img_dense_x2d=True),
    dict(type='LoadAnnotations3D',
         with_bbox=True,
         with_label=True,
         with_bbox_3d=True,
         with_coord_3d=False,
         with_truncation=True,
         with_attr=True,
         with_velo=True),
    dict(type='RandomFlip3D', flip_ratio=0.5),
    dict(type='Crop3D', crop_box=(0, 228, 1600, 900), trunc_ignore_thres=0.8),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad3D', size_divisor=32),
    dict(type='DefaultFormatBundle3D'),
    dict(type='Collect',
         keys=['img', 'gt_bboxes', 'gt_bboxes_ignore', 'gt_labels',
               'gt_bboxes_3d', 'img_dense_x2d', 'img_dense_x2d_mask', 'cam_intrinsic',
               'gt_attr', 'gt_velo']),
]
test_pipeline = [
    dict(type='LoadImageFromFile3D',
         with_img_dense_x2d=True),
    dict(type='MultiScaleFlipAug',
         scale_factor=1.0,
         flip=False,
         transforms=[
             dict(type='RandomFlip3D', flip_ratio=0.5),
             dict(type='Crop3D', crop_box=(0, 228, 1600, 900)),
             dict(type='Normalize', **img_norm_cfg),
             dict(type='Pad3D', size_divisor=32),
             dict(type='DefaultFormatBundle3D'),
             dict(type='Collect', keys=[
                 'img', 'img_dense_x2d', 'img_dense_x2d_mask', 'cam_intrinsic']),
         ])
]
data = dict(
    samples_per_gpu=3,
    workers_per_gpu=1,
    train=dict(
        type=dataset_type,
        ann_file='nuscenes_annotations_train.pkl',
        pipeline=train_pipeline,
        data_root=data_root,
        trunc_ignore_thres=0.8,
        min_visibility=2,
        filter_empty_gt=True),
    val=dict(
        type=dataset_type,
        samples_per_gpu=6,
        ann_file='nuscenes_annotations_val.pkl',
        pipeline=test_pipeline,
        data_root=data_root,
        filter_empty_gt=False),
    test=dict(
        type=dataset_type,
        samples_per_gpu=6,
        ann_file='nuscenes_annotations_test.pkl',
        pipeline=test_pipeline,
        data_root=data_root,
        filter_empty_gt=False))
evaluation = dict(
    interval=1,
    metric='NDS')
# optimizer
optimizer = dict(
    type='AdamW',
    lr=1e-4,
    weight_decay=0.0001,
    paramwise_cfg=dict(
        custom_keys={
            'sampling_offsets': dict(lr_mult=0.1)
        }))
optimizer_config = dict(
    type='OptimizerHookMod',
    grad_clip=dict(max_norm=5.0, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    step=[10, 11])
runner = dict(type='EpochBasedRunner', max_epochs=12)
checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=20,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
custom_hooks = [dict(type='EmptyCacheHook')]
find_unused_parameters = True
