_base_ = './faster_rcnn_r50_fpn_mstrain_3x_wheat.py'

data = dict(
    samples_per_gpu=12)

model = dict(
    backbone=dict(
        depth=101,
        init_cfg=dict(type='Pretrained',
                      checkpoint='torchvision://resnet101')))

# set path to pretrained detector to init weights from
load_from = 'checkpoints/faster_rcnn/faster_rcnn_r101_fpn_mstrain_3x_coco.pth'

# Set up working dir to save files and logs.
work_dir = './experiments/faster_rcnn/r101_fpn'