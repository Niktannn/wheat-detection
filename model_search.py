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
import shutil
from mmdet.datasets import build_dataset

from utils.split_data import write_ann_file
import csv
from enum import Enum

class Mode(Enum):
  RCNN = 'faster_rcnn'
  YOLO = 'yolox'

#######################################################################
####################### MODEL SEARCH PARAMETERS #######################
#######################################################################

backbones_by_name = {
    'ResNet50'      : 'faster_rcnn/faster_rcnn_r50_fpn_mstrain_3x_wheat.py',
    'ResNet101'     : 'faster_rcnn/faster_rcnn_r101_fpn_mstrain_3x_wheat.py',
    'ResNext101'    : 'faster_rcnn/faster_rcnn_x101_64x4d_fpn_mstrain_3x_wheat.py',
    'yolox-tiny'    : 'yolox/yolox_tiny_8x8_300e_wheat.py',
    'yolox-small'   : 'yolox/yolox_s_8x8_300e_wheat.py'
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

############################################################
##################### Faster-RCNN GRID #####################
############################################################

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
}
]

############################################################
######################## YOLOX GRID ########################
############################################################

yolo_grid = [{ 'backbone'   : ['yolox-tiny', 'yolox-small'] }]

n_splits = 3

faster_rcnn_configs = [
    'configs/faster_rcnn/faster_rcnn_r50_fpn_mstrain_3x_wheat.py',
    'configs/faster_rcnn/faster_rcnn_r101_fpn_mstrain_3x_wheat.py']
yolox_configs = [
    'configs/yolox/yolox_tiny_8x8_300e_wheat.py',
    'configs/yolox/yolox_s_8x8_300e_wheat.py'
]

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
        print(row_format.format(row_name, *[round(x, 3) for x in data[key]]))
        print(row_format_bet.format(
            "-" * width[0], *["-" * width[i + 1] for i, _ in enumerate(title)]))


def write_annotations_kfold(skf, base_dir, img_ids, regions, cfg):
    fold_datasets = []
    fold_num = 0
    for train_index, val_index in skf.split(img_ids, regions):
        train_ids, val_ids = img_ids[train_index], img_ids[val_index]

        train_ann_file = os.path.join(base_dir, f'train_{str(fold_num)}.txt')
        val_ann_file = os.path.join(base_dir, f'val_{str(fold_num)}.txt')

        write_ann_file(train_ann_file, train_ids)
        write_ann_file(val_ann_file, val_ids)

        cfg.data.train.dataset.ann_file = train_ann_file

        fold_datasets.append(build_dataset(cfg.data.train))
        fold_num+=1
    return fold_datasets


def faster_rcnn_model_search(seed=0, max_epochs=3, train_size=None):
    model_search(seed, max_epochs, train_size, mode=Mode.RCNN)


def yolo_model_search(seed=0, max_epochs=50, train_size=None):
    model_search(seed, max_epochs, train_size, mode=Mode.YOLO)


def model_search(seed=0, max_epochs=3, train_size=None, mode=Mode.RCNN):
    train_data = pd.read_csv('data/annotations.csv')

    train_data['source'] = pd.factorize(train_data['source'])[0]
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

    param_grid = ParameterGrid({})
    if mode == Mode.RCNN:
        param_grid = ParameterGrid(frcnn_grid)
    elif mode == Mode.YOLO:
        param_grid = ParameterGrid(yolo_grid)
    scores = {}
    best_score = 0.

    skf = StratifiedKFold(n_splits=n_splits, random_state=seed, shuffle=True)

    base_dir = os.path.join('model_search', 'faster_rcnn' if mode == Mode.RCNN else 'yolox')
    mmcv.mkdir_or_exist(base_dir)

    test_ids = [os.path.splitext(os.path.basename(f))[0] for f in os.listdir('data/test')]
    test_ann_file = os.path.join(base_dir, f'test.txt')
    write_ann_file(test_ann_file, test_ids)

    if mode == Mode.RCNN:
        fold_datasets = write_annotations_kfold(skf, base_dir, img_ids, regions,
                                                mmcv.Config.fromfile('configs/' + backbones_by_name['ResNet50']))

    for params in tqdm(param_grid):
        print(f'evaluating parameters:\n{params}')
        sum_map = 0.

        if mode == Mode.YOLO:
            fold_datasets = write_annotations_kfold(skf, base_dir, img_ids, regions,  mmcv.Config.fromfile(
                'configs/' + backbones_by_name[params['backbone']]))

        for fold_num in range(n_splits):
            cfg = mmcv.Config.fromfile('configs/' + backbones_by_name[params['backbone']])

            if mode == Mode.RCNN:
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
                map = max([json.loads(line)['mAP'] for line in json_file
                           if 'mode' in json.loads(line) and json.loads(line)['mode']=='val'])

            sum_map += map

        score = sum_map / n_splits
        scores[str(params)] = [score]
        if score > best_score:
            best_score = score
            cfg.dump(os.path.join(base_dir, 'best_config.json'))
        print(score)
        print('====================================')

    draw_table(scores)


def get_models_info(mode):
    if mode == Mode.RCNN:
        param_grid = ParameterGrid(frcnn_grid)
    elif mode == Mode.YOLO:
        param_grid = ParameterGrid(yolo_grid)
    else:
        raise ValueError("unknown mode, shoud be 'rcnn' or 'yolo'")

    scores = {}
    checkpoints = {}

    for params in param_grid:
        base_dir = os.path.join('model_search',
                                'faster_rcnn' if mode == Mode.RCNN else 'yolox',
                                '_'.join([str(p) for p in params.values()])
                                )
        fold_scores = np.zeros(n_splits, dtype=np.float)

        for fold_num in range(n_splits):
            work_dir = os.path.join(base_dir, 'fold_' + str(fold_num))
            with open(os.path.join(work_dir, "None.log.json"), 'r') as json_file:
                fold_scores[fold_num] = max([json.loads(line)['mAP']
                                             for line in json_file
                                             if 'mode' in json.loads(line) and
                                             json.loads(line)['mode'] == 'val'])
        best_fold = np.argmax(fold_scores)
        checkpoint = os.path.join(base_dir, 'fold_' + str(best_fold), 'latest.pth')

        score = np.mean(fold_scores)
        if mode == Mode.RCNN:
            name = f"Faster-RCNN {params['loss_bbox']}"
        else:
            name = 'YOLOX'
        scores.setdefault(name, [])
        scores[name].append(score)

        checkpoints.setdefault(name, [])
        checkpoints[name].append(checkpoint)

    return scores, checkpoints

def aggregate_results(file='models_info.csv'):
    lines = []
    scores_table = {}

    for mode in [Mode.RCNN, Mode.YOLO]:
        if mode == Mode.RCNN:
            param_grid = ParameterGrid(frcnn_grid)
        else:
            param_grid = ParameterGrid(yolo_grid)

        for params in param_grid:
            base_dir = os.path.join('model_search',
                                    'faster_rcnn' if mode == Mode.RCNN else 'yolox',
                                    '_'.join([str(p) for p in params.values()])
                                    )
            fold_scores = np.zeros(n_splits, dtype=np.float)

            for fold_num in range(n_splits):
                work_dir = os.path.join(base_dir, 'fold_' + str(fold_num))
                with open(os.path.join(work_dir, "None.log.json"), 'r') as json_file:
                    fold_scores[fold_num] = max([json.loads(line)['mAP']
                                                 for line in json_file
                                                 if 'mode' in json.loads(line) and
                                                 json.loads(line)['mode'] == 'val'])
            best_fold = np.argmax(fold_scores)
            checkpoint = os.path.join(base_dir, 'fold_' + str(best_fold), 'latest.pth')
            log_file = os.path.join(base_dir, 'fold_' + str(best_fold), 'None.log.json')
            tf_logs = os.path.join(base_dir, 'fold_' + str(best_fold), 'tf_logs')

            score = round(np.mean(fold_scores),3)

            if mode == Mode.RCNN:
                name = f"Faster-RCNN {params['backbone']} {params['loss_bbox']}"
                file_name = f"faster_rcnn_{params['backbone']}_{params['loss_bbox']}"
                dst_checkpoint = os.path.join('checkpoints', "faster_rcnn",
                                              f"{file_name}_wheat.pth")
                dst_log = os.path.join('logs', "faster_rcnn", file_name)
            else:
                name = f"YOLOX {params['backbone']}"
                file_name = params['backbone']
                dst_checkpoint = os.path.join('checkpoints', "yolox",
                                              f"{file_name}_wheat.pth")
                dst_log = os.path.join('logs', "yolox", file_name)
            shutil.copy(checkpoint, dst_checkpoint)

            os.makedirs(dst_log, exist_ok = True)
            shutil.copy(log_file, dst_log)
            shutil.copytree(tf_logs, os.path.join(dst_log,'tf_logs'), dirs_exist_ok=True)

            scores_table[name] = [score]
            lines.append([name, score, dst_checkpoint, "configs/" + backbones_by_name[params['backbone']], dst_log])


    header = ['name', 'score', 'checkpoint', 'config', 'log']
    with open(file, 'w', newline='', encoding='UTF8') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(header)
        for line in lines:
            writer.writerow(line)

    draw_table(scores_table)