model = dict(
    type='CascadeRCNN',
    num_stages=3,
    pretrained=None,
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        style='pytorch',
        dcn=dict(modulated=False, deformable_groups=1, fallback_on_stride=False),
        stage_with_dcn=(False, True, True, True)
    ),
    neck=dict(type='FPN', in_channels=[256, 512, 1024, 2048], out_channels=256, num_outs=5),
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_scales=[8],
        anchor_ratios=[0.5, 1.0, 2.0],
        anchor_strides=[4, 8, 16, 32, 64],
        target_means=[.0, .0, .0, .0],
        target_stds=[1.0, 1.0, 1.0, 1.0],
        use_sigmoid_cls=True
    ),
    bbox_roi_extractor=dict(
        type='SingleRoIExtractor',
        roi_layer=dict(type='RoIAlign', out_size=7, sample_num=2),
        out_channels=256,
        featmap_strides=[4, 8, 16, 32]
    ),
    bbox_head=[
        dict(
            type='SharedFCBBoxHead',
            num_fcs=2,
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=2,
            target_means=[0., 0., 0., 0.],
            target_stds=[0.1, 0.1, 0.2, 0.2],
            reg_class_agnostic=True
        ),
        dict(
            type='SharedFCBBoxHead',
            num_fcs=2,
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=2,
            target_means=[0., 0., 0., 0.],
            target_stds=[0.05, 0.05, 0.1, 0.1],
            reg_class_agnostic=True
        ),
        dict(
            type='SharedFCBBoxHead',
            num_fcs=2,
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=2,
            target_means=[0., 0., 0., 0.],
            target_stds=[0.033, 0.033, 0.067, 0.067],
            reg_class_agnostic=True
        )
    ]
)
# model training and testing settings
train_cfg = dict(
    rpn=dict(
        assigner=dict(type='MaxIoUAssigner', pos_iou_thr=0.7, neg_iou_thr=0.3, min_pos_iou=0.3, ignore_iof_thr=-1),
        sampler=dict(type='RandomSampler', num=256, pos_fraction=0.5, neg_pos_ub=-1, add_gt_as_proposals=False),
        allowed_border=0,
        pos_weight=-1,
        smoothl1_beta=1 / 9.0,
        debug=False
    ),
    rcnn=[
        dict(
            assigner=dict(type='MaxIoUAssigner', pos_iou_thr=0.5, neg_iou_thr=0.5, min_pos_iou=0.5, ignore_iof_thr=-1),
            sampler=dict(type='RandomSampler', num=512, pos_fraction=0.25, neg_pos_ub=-1, add_gt_as_proposals=True),
            pos_weight=-1,
            debug=False
        ),
        dict(
            assigner=dict(type='MaxIoUAssigner', pos_iou_thr=0.6, neg_iou_thr=0.6, min_pos_iou=0.6, ignore_iof_thr=-1),
            sampler=dict(type='RandomSampler', num=512, pos_fraction=0.25, neg_pos_ub=-1, add_gt_as_proposals=True),
            pos_weight=-1,
            debug=False
        ),
        dict(
            assigner=dict(type='MaxIoUAssigner', pos_iou_thr=0.7, neg_iou_thr=0.7, min_pos_iou=0.7, ignore_iof_thr=-1),
            sampler=dict(type='RandomSampler', num=512, pos_fraction=0.25, neg_pos_ub=-1, add_gt_as_proposals=True),
            pos_weight=-1,
            debug=False
        )
    ],
    stage_loss_weights=[1, 0.5, 0.25]
)
test_cfg = dict(
    rpn=dict(nms_across_levels=False, nms_pre=3000, nms_post=3000, max_num=3000, nms_thr=0.7, min_bbox_size=0),
    rcnn=dict(score_thr=0.04, nms=dict(type='soft_nms', iou_thr=0.75, min_score=0.04), max_per_img=60),
    keep_all_stages=False
)
# dataset settings
dataset_type = 'CustomDataset'
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
height = 2048
width = 3008
min_height = 1600
min_width = 2624
data = dict(
    imgs_per_gpu=1,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        ann_file='/wdata/mmdetection/training_fix.pickle',
        img_prefix='/wdata/training_resize',
        img_scale=[(min_width, min_height), (width, height)],
        img_norm_cfg=img_norm_cfg,
        size_divisor=32,
        flip_ratio=0.0,
        with_mask=False,
        with_crowd=False,
        with_label=True,
        extra_aug=dict(
            type='Compose',
            transforms=[
                dict(
                    always_apply=False,
                    p=0.5,
                    type='HorizontalFlip',
                ),
                dict(
                    p=0.5,
                    transforms=[
                        dict(always_apply=False, loc=0, p=0.5, scale=[2.5, 12.75], type='IAAAdditiveGaussianNoise'),
                        dict(always_apply=False, p=0.5, type='GaussNoise', var_limit=[10.0, 50.0])
                    ],
                    type='OneOf'
                ),
                dict(
                    p=0.5,
                    transforms=[
                        dict(always_apply=False, blur_limit=[3, 5], p=0.5, type='MotionBlur'),
                        dict(always_apply=False, blur_limit=[3, 5], p=0.5, type='MedianBlur'),
                        dict(always_apply=False, blur_limit=[3, 5], p=0.5, type='Blur')
                    ],
                    type='OneOf'
                ),
                dict(
                    slant_lower=-10,
                    slant_upper=10,
                    drop_length=20,
                    drop_width=1,
                    drop_color=(200, 200, 200),
                    blur_value=7,
                    brightness_coefficient=0.7,
                    rain_type=None,
                    always_apply=False,
                    p=0.25,
                    type='RandomRain'
                ),
                dict(
                    always_apply=False,
                    brightness_limit=[-0.3, 0.3],
                    contrast_limit=[-0.3, 0.3],
                    p=0.5,
                    type='RandomBrightnessContrast'
                ),
                dict(always_apply=False, p=0.5, quality_lower=60, quality_upper=99, type='JpegCompression'),
                dict(
                    p=0.5,
                    shift_limit=0.15,
                    rotate_limit=0,
                    border_mode=0,
                    scale_limit=0.25,
                    type='ShiftScaleRotate'
                ),
                dict(
                    always_apply=False,
                    p=0.25,
                    limit=10,
                    interpolation=1,
                    border_mode=0,
                    type='Rotate'
                )
            ],
            p=1.0,
            bbox_params=dict(format='pascal_voc', min_visibility=0.75, label_fields=['labels'])
        )
    ),
    test=dict(
        type=dataset_type,
        ann_file='/wdata/mmdetection/test.pickle',
        img_prefix='/data/test',
        img_scale=(3040, 2080),
        img_norm_cfg=img_norm_cfg,
        size_divisor=32,
        flip_ratio=1,
        with_mask=False,
        with_label=False,
        test_mode=True
    )
)
# optimizer
optimizer = dict(type='SGD', lr=0.075, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(policy='step', warmup='linear', warmup_iters=500, warmup_ratio=1 / 3, step=[7, 11])
checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=25,
    hooks=[
        dict(type='TextLoggerHook'),
    ])
# runtime settings
total_epochs = 15
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = '/wdata/cascade_rcnn_dconv_c3-c5_r50_fpn/'
load_from = '/wdata/cascade_rcnn_dconv_c3-c5_r50_fpn_1x_20190125-dfa53166_part.pth'
resume_from = None
workflow = [('train', 1)]
