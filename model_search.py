import os
import random
import numpy as np
import pandas as pd
import argparse
import mmcv
from sklearn.model_selection import train_test_split, ParameterGrid, StratifiedKFold
from train import train
from evaluate import evaluate
from tqdm.notebook import tqdm
import json
from mmdet.datasets import build_dataset

seed = 0


backbones_by_name = {
    'ResNet50'      : 'faster_rcnn_r50_fpn_mstrain_3x_wheat.py',
    'ResNet101'     : 'faster_rcnn_r101_fpn_mstrain_3x_wheat.py',
    'ResNext101'    : 'faster_rcnn_x101_64x4d_fpn_mstrain_3x_wheat.py'
}


bbox_losses_by_name = {
    'L1'        : {},
    'IoULoss'   : { 'model.roi_head.bbox_head.reg_decoded_bbox' : True,
                    'model.roi_head.bbox_head.loss_bbox.type' : 'IoULoss',
                    'model.roi_head.bbox_head.loss_bbox.loss_weight' : 10.0}
}


optimizers_by_name = {
    'SGD'   : { 'optimizer.type' : 'SGD'},
    'Adam'  : { 'optimizer._delete_' : True,
                'optimizer.type' : 'Adam'}
}

schedulers_by_name = {
    'step' : {
        'lr_config.policy'  : 'step',
        'lr_config.step'    : [9,11]
    },
    'cosine' : {
        'lr_config._delete_'        : True,
        'lr_config.policy'          : 'CosineAnnealing',
        'lr_config.min_lr_ratio'    : 1e-5
    }
}

warmup_by_name = {
    'fast' : {
        'lr_config.warmup_iters' : 100,
        'lr_config.warmup_ratio' : 0.1
    },
    'slow' : {
        'lr_config.warmup_iters' : 1000,
        'lr_config.warmup_ratio' : 0.001
    }
}


frcnn_grid = [
{
    'backbone'      : ['ResNet50', 'ResNet101'],
    'loss_bbox'     : ['L1','IoULoss'],
    'optimizer'     : ['SGD'],
    'lr'            : [0.01],
    'momentum'      : [0.9],
    'scheduler'     : ['step'],
    'warmup'        : ['linear'],
    'warmup_speed'  : ['fast']
},
{   'backbone'      : ['ResNet50', 'ResNet101'],
    'loss_bbox'     : ['L1', 'IoULoss'],
    'optimizer'     : ['Adam'],
    'lr'            : [0.005],
    'scheduler'     : ['cosine'],
    'warmup'        : ['linear'],
    'warmup_speed'  : ['fast']
}
]



def write_ann_file(file, ids):
    with open(file, 'w') as f:
        for id in ids:
            f.write(f"{id}\n")


def draw_table(data, title=['mAP'], width=[100, 6]):
    row_format = '|' + '|'.join([("{:>" + str(w) + "}") for w in width]) + '|'
    row_format_bet = '+' + '+'.join([("{:>" + str(w) + "}") for w in width]) + '+'

    print(row_format_bet.format(
        "-" * width[0], *["-" * width[i + 1] for i, _ in enumerate(title)]))
    print(row_format.format("", *title))
    print(row_format_bet.format(
        "-" * width[0], *["-" * width[i + 1] for i, _ in enumerate(title)]))
    for key in data:
        if len(key) > width[0]:
            row_name = '...' + key[len(key) - width[0] + 3:]
        else:
            row_name = key
        print(row_format.format(row_name, *[round(x, 2) for x in data[key]]))
        print(row_format_bet.format(
            "-" * width[0], *["-" * width[i + 1] for i, _ in enumerate(title)]))


def faster_rcnn_model_search(seed=0, max_epochs=3, train_size=None):
    train_data = pd.read_csv('data/annotations.csv')

    train_data['source']= pd.factorize(train_data['source'])[0]
    train_data['source'] = train_data['source'] + 1

    imgs_regions = train_data.groupby('image_id')['source'].first()
    img_ids = imgs_regions.index.to_numpy(dtype=str)
    regions = imgs_regions.to_numpy(dtype=np.int64)

    non_labeled = 0
    for f in os.listdir('data/train'):
        id = os.path.splitext(os.path.basename(f))[0]
        if id not in img_ids:
            non_labeled += 1
            np.append(img_ids, id)
            np.append(regions, 0)

    if train_size:
        partial_size = min(train_size, len(img_ids))
        img_ids = img_ids[:partial_size]
        regions = regions[:partial_size]


    param_grid = ParameterGrid(frcnn_grid)
    scores = {}
    best_score = 0.

    n_splits = 3
    skf = StratifiedKFold(n_splits=n_splits, random_state=seed, shuffle=True)

    fold_datasets = []
    base_dir = os.path.join('model_search', 'faster_rcnn')
    mmcv.mkdir_or_exist(base_dir)
    cfg = mmcv.Config.fromfile('configs/faster_rcnn/' + backbones_by_name['ResNet50'])

    fold_num = 0

    test_ids = [os.path.splitext(os.path.basename(f))[0] for f in os.listdir('data/test')]
    test_ann_file = os.path.join(base_dir, f'test.txt')
    write_ann_file(test_ann_file, test_ids)

    for train_index, val_index in skf.split(img_ids, regions):
        train_ids, val_ids = img_ids[train_index], img_ids[val_index]

        train_ann_file = os.path.join(base_dir, f'train_{str(fold_num)}.txt')
        val_ann_file = os.path.join(base_dir, f'val_{str(fold_num)}.txt')

        write_ann_file(train_ann_file, train_ids)
        write_ann_file(val_ann_file, val_ids)

        cfg.data.train.dataset.ann_file = train_ann_file

        fold_datasets.append(build_dataset(cfg.data.train))
        fold_num+=1

    for params in tqdm(param_grid):
        print(f'evaluating parameters:\n{params}')

        sum_map = 0.

        for fold_num in range(n_splits):
            cfg = mmcv.Config.fromfile('configs/faster_rcnn/' + backbones_by_name[params['backbone']])
            cfg.merge_from_dict({**bbox_losses_by_name[params['loss_bbox']],
                                 **optimizers_by_name[params['optimizer']],
                                 **schedulers_by_name[params['scheduler']]
                                 })
            cfg.optimizer.lr = params['lr']
            if params['optimizer'] == 'SGD':
                cfg.optimizer.momentum = params['momentum']

            cfg.lr_config.warmup = params['warmup']
            cfg.merge_from_dict(warmup_by_name[params['warmup_speed']])

            work_dir = os.path.join(base_dir,
                                    '_'.join([str(p) for p in params.values()]),
                                    'fold_' + str(fold_num)
                                    )
            mmcv.mkdir_or_exist(work_dir)

            train_ann_file = os.path.join(base_dir, f'train_{str(fold_num)}.txt')
            val_ann_file = os.path.join(base_dir, f'val_{str(fold_num)}.txt')

            cfg.work_dir = work_dir
            cfg.data.train.dataset.ann_file = train_ann_file
            cfg.data.test.ann_file = test_ann_file
            cfg.data.val.ann_file = val_ann_file

            if train_size:
                cfg.log_config.interval = train_size

            train(cfg, seed=seed, max_epochs=max_epochs, dataset=fold_datasets[fold_num])

            with open(os.path.join(work_dir, "None.log.json"), 'r') as json_file:
                map = max([json.loads(line)['mAP'] for line in json_file if json.loads(line)['mode']=='val'])

            sum_map += map

        score = sum_map / n_splits
        scores[str(params)] = [score]
        if score > best_score:
            best_score = score
            cfg.dump(os.path.join(base_dir, 'best_config.json'))
        print(score)
        print('====================================')

    draw_table(scores)