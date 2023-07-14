_base_ = [
    '../_base_/models/faster_rcnn_r50_fpn.py',
    '../_base_/default_runtime.py'
]
model = dict(
    backbone=dict(init_cfg=None),
    roi_head=dict(
        bbox_head=dict(
            type='Shared2FCBBoxHead',
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=8,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0., 0., 0., 0.],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=False,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0))))

dataset_type = 'CityscapesDataset_fuda'
data_root_source = '../data/cityscapes/'
data_root_target = '../data/cityscapes_foggy/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='Resize', img_scale=[(1024, 400), (1024, 512)], keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1024, 512),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

data = dict(
    samples_per_gpu=8,
    workers_per_gpu=16,
    # cityscapes
    train_source=dict(
        dict(
            type='RepeatDataset',
            times=1,
            dataset=dict(
                type=dataset_type,
                DA_mode=True,
                ann_file=[data_root_source +
                         'annotations/instancesonly_filtered_gtFine_train.json',
                          data_root_target +
                         'annotations/instancesonly_filtered_gtFine_train.json'
                          ],
                img_prefix=[data_root_source + 'leftImg8bit/train/',
                          data_root_target + 'leftImg8bit_foggy/train/'
                            ],
                pipeline=train_pipeline)
        )),


    val=dict(
        type=dataset_type,
        DA_mode=True,
        ann_file=[data_root_source +
                  'annotations/instancesonly_filtered_gtFine_val.json',
                  data_root_target +
                  'annotations/instancesonly_filtered_gtFine_val.json'
                  ],
        img_prefix=[data_root_source + 'leftImg8bit/val/',
                    data_root_target + 'leftImg8bit_foggy/val/'
                    ],
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        DA_mode=True,
        ann_file=[data_root_source +
                  'annotations/instancesonly_filtered_gtFine_val.json',
                  data_root_target +
                  'annotations/instancesonly_filtered_gtFine_val.json'
                  ],
        img_prefix=[data_root_source + 'leftImg8bit/val/',
                    data_root_target + 'leftImg8bit_foggy/val/'
                    ],
        pipeline=test_pipeline)
)

