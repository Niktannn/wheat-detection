_base_ = [
    '../common/mstrain_3x_coco.py', '../_base_/models/faster_rcnn_r50_fpn.py'
]

dataset_type = 'WheatDataset'
data_root = 'data/'

data = dict(
    samples_per_gpu=20,
    train=dict(
        dataset = dict(
            type = dataset_type,
            ann_file = data_root+'train.txt',
            img_prefix = data_root+'train')
    ),
    test=dict(
        type = dataset_type,
        ann_file = data_root+'test.txt',
        img_prefix = data_root+'test'
    ),
    val=dict(
        type = dataset_type,
        ann_file = data_root+'val.txt',
        img_prefix = data_root+'train'
    ) 
)

model = dict(
    roi_head = dict(
        bbox_head = dict(
            num_classes = 1
        )
    )
)


load_from = 'checkpoints/faster_rcnn/faster_rcnn_r50_fpn_mstrain_3x_coco.pth'

work_dir = './experiments/faster_rcnn/r50_fpn'

optimizer = dict(
    lr = 0.02 / 8
)

lr_config = dict(
    warmup_iters = 500
)

runner = dict(type='EpochBasedRunner', max_epochs=3)

evaluation = dict(
    metric = 'mAP',
    interval = 1)
checkpoint_config = dict(interval=2)

device = 'cuda'
gpu_ids = range(1)

log_config = dict(
    interval=100,
    hooks = [
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')])

seed = 0