_base_='./yolox_s_8x8_300e_coco.py'

dataset_type = 'WheatDataset'
data_root = 'data/'

data = dict(
    samples_per_gpu=16,
    test=dict(
        type=dataset_type,
        ann_file = data_root + 'test.txt',
        img_prefix = data_root + 'test'
        ),
    val = dict(
        type=dataset_type,
        ann_file = data_root + 'train.txt',
        img_prefix=data_root + 'train/'
        ),
    train=dict(
        dataset=dict(
            type=dataset_type,
            ann_file = data_root + 'train.txt',
            img_prefix = data_root + 'train/'
            )
        )
    )

train_dataset = dict(
    dataset = dict(
        type=dataset_type,
        ann_file = data_root + 'train.txt',
        img_prefix = data_root + 'train/'
    )
)

model = dict(bbox_head = dict(num_classes = 1))

load_from = 'checkpoints/yolox/yolox_s_8x8_300e_coco.pth'

work_dir = './experiments/yolox/yolox_s'

optimizer = dict(lr = 0.01 / 4)
lr_config = dict(warmup_iters = 5 * 2)
auto_scale_lr = dict(enable=False)

interval = 5

evaluation = dict(
    interval=interval,
    metric = 'mAP',
    save_best= 'mAP'
)

checkpoint_config = dict(interval=interval)

device = 'cuda'
gpu_ids = range(1)

log_config = dict(
    interval=40,
    hooks = [
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')])