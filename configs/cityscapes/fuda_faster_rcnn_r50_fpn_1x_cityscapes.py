_base_ = [
    '../_base_/models/fuda_faster_rcnn_r50_fpn.py',
    # '../_base_/datasets/cityscapes_detection.py',
    '../_base_/default_runtime.py'
]

model = dict(
    backbone=dict(init_cfg=None),
    roi_head=dict(
        type='IUARoIHead',
        Uncertain_map_extractor=dict(
            type='UncertaintyMapExtractor',
            roi_layer=dict(type='RoIAlign', output_size=1, sampling_ratio=0),
            out_channels=1,
            featmap_strides=[4, 8, 16, 32]),
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=dict(
            type='IUABBoxHead',
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

# dataset for FUDA

dataset_type_train = 'CityscapesDataset_fuda'
dataset_type_val = 'CityscapesDataset'
data_root_source = '../data/cityscapes/'
data_root_target = '../data/cityscapes_foggy/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

train_pipeline = [
    dict(type='LoadDaImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='Resize', img_scale=[(2048, 800), (2048, 1024)], keep_ratio=True),
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
        img_scale=(2048, 1024),
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
    samples_per_gpu=1,
    workers_per_gpu=2,
    # cityscapes
    train_source=dict(
        dict(
            type='RepeatDataset',
            times=8,
            dataset=dict(
                type=dataset_type_train,
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
        type=dataset_type_val,
        ann_file=data_root_target +
                 'annotations/instancesonly_filtered_gtFine_val.json',
        img_prefix=data_root_target + 'leftImg8bit_foggy/val/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type_val,
        ann_file=data_root_target +
                 'annotations/instancesonly_filtered_gtFine_val.json',
        img_prefix=data_root_target + 'leftImg8bit_foggy/val/',
        pipeline=test_pipeline))


evaluation = dict(interval=1, metric='bbox')

# optimizer
# lr is set for a batch size of 8
optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    # [7] yields higher performance than [6]
    step=[7])
runner = dict(
    type='EpochBasedRunner', max_epochs=20)  # actual epoch = 8 * 8 = 64
log_config = dict(interval=100)
# For better, more stable performance initialize from COCO
load_from = 'https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'  # noqa

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# USER SHOULD NOT CHANGE ITS VALUES.
# base_batch_size = (8 GPUs) x (1 samples per GPU)
auto_scale_lr = dict(base_batch_size=1)
