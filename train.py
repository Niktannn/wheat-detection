import argparse
import numpy as np
import mmcv
import random
import os
import torch

import wheat_dataset

from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.apis import train_detector, set_random_seed

def train(config, resume_from=None, max_epochs=None, work_dir=None, seed=0,
          dataset=None):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    #load config
    if isinstance(config, str):
        config = mmcv.Config.fromfile(config)
    elif not isinstance(config, mmcv.Config):
        raise TypeError('config must be a filename or Config object, '
                        f'but got {type(config)}')
    
    #change config
    if resume_from:
      config.load_from = None
      config.resume_from = resume_from
    if work_dir:
        config.work_dir = work_dir
    if max_epochs:
        config.runner.max_epochs = max_epochs
    
    config.seed = seed
    # Set seed thus the results are more reproducible
    set_random_seed(seed, deterministic=True)

    # Build dataset
    if dataset:
        datasets = [dataset]
    else:
        datasets = [build_dataset(config.data.train)]

    # Build the detector
    model = build_detector(config.model)
    # Add an attribute for visualization convenience
    model.CLASSES = datasets[0].CLASSES
    
    # Create work_dir
    mmcv.mkdir_or_exist(os.path.abspath(config.work_dir))

    train_detector(model, datasets, config, distributed=False, validate=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='path to mmdetection model config')
    parser.add_argument('--resume_from', type=str, default=None, 
                        help='path to load weights from to resume training')
    parser.add_argument('--max_epochs', type=int, default=None, 
                        help='max number of epochs to train')
    parser.add_argument('--work_dir', type=str, help='path to working directory where to save logs')
    
    opt = parser.parse_args()
    train(opt.config, opt.resume_from, opt.max_epochs, opt.work_dir)