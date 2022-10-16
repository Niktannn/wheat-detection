import argparse
import time
import os
import logging

import torch
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint, wrap_fp16_model

from mmdet.datasets import (build_dataloader, build_dataset,
                            replace_ImageToTensor)
from mmdet.models import build_detector

import wheat_dataset
from .split_data import write_ann_file

def fps_bench(config, checkpoint, max_iters=2000, num_warmup=10, print_logs=False, log_interval=50):
    if isinstance(config, str):
        cfg = Config.fromfile(config)
    elif not isinstance(config, Config):
        raise TypeError('config must be a filename or Config object, '
                        f'but got {type(config)}')
    else:
        cfg=config

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True

    # build the dataloader
    samples_per_gpu = cfg.data.test.pop('samples_per_gpu', 1)
    if samples_per_gpu > 1:
        # Replace 'ImageToTensor' to 'DefaultFormatBundle'
        cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)

    all_imgs_ids = [os.path.splitext(os.path.basename(f))[0] for f in os.listdir('data/train')]
    all_ids_file = 'data/all.txt'
    write_ann_file(all_ids_file, all_imgs_ids)

    cfg.data.test.ann_file= all_ids_file
    cfg.data.test.img_prefix='data/train'

    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False)

    # build the model and load checkpoint
    cfg.model.train_cfg = None
    model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))

    # suppress any output except error messages
    logger = logging.getLogger("mechanize")
    logger.setLevel(logging.ERROR)
    load_checkpoint(model, checkpoint, logger=logger, map_location='cpu')

    model = MMDataParallel(model, device_ids=[0])

    model.eval()

    pure_inf_time = 0

    # benchmark with 2000 image and take the average
    for i, data in enumerate(data_loader):

        torch.cuda.synchronize()
        start_time = time.perf_counter()

        with torch.no_grad():
            model(return_loss=False, rescale=True, **data)

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start_time

        if i >= num_warmup:
            pure_inf_time += elapsed
            if (i + 1) % log_interval == 0:
                fps = (i + 1 - num_warmup) / pure_inf_time
                if print_logs:
                    print(f'Done image [{i + 1:<3}/ 2000], fps: {fps:.1f} img / s')

        if (i + 1) == max_iters:
            pure_inf_time += elapsed
            fps = (i + 1 - num_warmup) / pure_inf_time
            if print_logs:
                print(f'Overall fps: {fps:.1f} img / s')
            break

    os.remove(all_ids_file)

    return fps