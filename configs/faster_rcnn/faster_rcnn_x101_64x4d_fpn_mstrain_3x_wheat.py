_base_ = './faster_rcnn_r50_fpn_mstrain_3x_wheat.py'

model = dict(
    backbone=dict(
        type='ResNeXt',
        depth=101,
        groups=64,
        base_width=4,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        style='pytorch',
        init_cfg=dict(
            type='Pretrained', checkpoint='open-mmlab://resnext101_64x4d')))

# set path to pretrained detector to init weights from
load_from = 'checkpoints/faster_rcnn/faster_rcnn_x101_64x4d_fpn_mstrain_3x_coco.pth'

# Set up working dir to save files and logs.
work_dir = './experiments/faster_rcnn/x101_64x4d_fpn'