_base_ = [
    './_base_/models/cascade_rcnn_r50_fpn.py',
    './_base_/datasets/coco_detection.py',
    './_base_/schedules/schedule_2x.py', './_base_/default_runtime.py'
]

num_classes = 8
model = dict(
    type='LCPDetector',
    init_cfg=dict(type='Pretrained', checkpoint='./checkpoints/cascade_rcnn_r50_city.pth'),
    backbone=dict(
        type='LCPResNet',
        frozen_stages=-1,
        init_cfg=None),
    crux_learner_cfg=dict(
        num_blocks=3,
        mask_thr=0.5,
        k=10
    ),
    kl_loss_weight=1,
    roi_head=dict(
        bbox_head=[
            dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=num_classes,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.1, 0.1, 0.2, 0.2]),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0,
                               loss_weight=1.0)),
            dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=num_classes,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.05, 0.05, 0.1, 0.1]),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0,
                               loss_weight=1.0)),
            dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=num_classes,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.033, 0.033, 0.067, 0.067]),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0))
        ]
    )
)

img_prefix = './data/city/foggy_city_images/'
img_df_prefix = './data/city/city_images/'
train_ann = './data/city/annotations/instances_trainval2017.json'
test_ann = './data/city/annotations/instances_test2017.json'
classes = ('bicycle', 'bus', 'person', 'train', 'truck', 'motorcycle', 'car', 'rider')

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=False),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=4,
    train=dict(
        _delete_=True,
        type='LCPDataset',
        dataset=dict(
            type='CocoDataset',
            ann_file=train_ann,
            img_prefix=img_prefix,
            classes=classes,
            pipeline=[
                dict(type='LoadImageFromFile'),
                dict(type='LoadAnnotations', with_bbox=True)
            ]),
    img_df_prefix=img_df_prefix,
    pipeline=[
        dict(type='Resize', img_scale=(1333, 800), keep_ratio=False),
        dict(type='RandomFlip', flip_ratio=0.5),
        dict(type='Normalize', **img_norm_cfg),
        dict(type='Pad', size_divisor=32),
        dict(type='LCPFormatBundle'),
        dict(type='Collect', keys=['img', 'img_df', 'gt_bboxes', 'gt_labels'])
    ]),
    val=dict(
        classes=classes,
        ann_file=test_ann,
        img_prefix=img_prefix,
        pipeline=test_pipeline),
    test=dict(
        classes=classes,
        ann_file=test_ann,
        img_prefix=img_prefix,
        pipeline=test_pipeline))
evaluation = dict(interval=1, metric='bbox', classwise=True, save_best='auto')

optimizer = dict(
    type='SGD', lr=0.002, momentum=0.9, weight_decay=1e-4,
)

lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[16, 22])

custom_hooks = [
    dict(type='NumClassCheckHook'),
    dict(type='TrainModeControllerHook', num_epoch=[6, 6, 12], train_mode=['catch_up_learner', 'crux_learner', 'alter'])
]

runner = dict(max_epochs=24)


checkpoint_config = dict(interval=1, max_keep_ckpts=5)
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ]
)