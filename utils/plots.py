from utils.get_flops import get_flops
import json
import mmcv
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import ParameterGrid
import numpy as np

from model_search import faster_rcnn_configs, yolox_configs

from utils.benchmark import fps_bench
from model_search import get_models_info
from model_search import Mode


def get_acc_size(mode):
    sizes = []
    flops = []

    if mode == Mode.RCNN:
        configs = faster_rcnn_configs
    elif mode == Mode.YOLO:
        configs = yolox_configs
    else:
        raise ValueError("unknown mode, shoud be 'rcnn' or 'yolo'")

    for cfg in configs:
        config = mmcv.Config.fromfile(cfg)
        flop, size = get_flops(config, (1024,))
        flops.append(flop)
        sizes.append(size)

    scores, _ = get_models_info(mode)

    return sizes, scores


def get_acc_fps(mode):
    if mode == Mode.RCNN:
        configs = faster_rcnn_configs
    elif mode == Mode.YOLO:
        configs = yolox_configs
    else:
        raise ValueError("unknown mode, shoud be 'rcnn' or 'yolo'")

    scores, checkpoints = get_models_info(mode)
    speeds = {}

    for (name, ckpts) in checkpoints.items():
        speeds[name] = []
        for i, cfg in enumerate(configs):
            config = mmcv.Config.fromfile(cfg)
            checkpoint = ckpts[i]
            speeds[name].append(fps_bench(config, checkpoint, max_iters=100))

    return speeds, scores


def acc_size_plot():
    rcnn_params, rcnn_scores = get_acc_size(Mode.RCNN)
    yolox_params, yolox_scores = get_acc_size(Mode.YOLO)
    rcnn_params = [float(p[:-2]) for p in rcnn_params]
    yolox_params = [float(p[:-2]) for p in yolox_params]

    plt.figure(figsize=(14, 10))
    for j, (name, scores) in enumerate(rcnn_scores.items()):
        # print(name, scores)
        plt.plot(rcnn_params, scores, '--o', markersize=15, label=name)
        for i, backbone in enumerate(['ResNet50', 'ResNet101']):
            x, y = rcnn_params[i], scores[i]
            tx, ty = x , y
            tx -=3
            if j == 0:
                ty+=0.0005
            else:
                ty-=0.0005
            plt.annotate(f'{backbone}',
                         (x, y),
                         (tx, ty),
                         fontsize=14)

    for name, scores in yolox_scores.items():
        plt.plot(yolox_params, scores, '--o', markersize=15, label=name)
        for i, backbone in enumerate(['yolox-tiny', 'yolox-small']):
            plt.annotate(f'{backbone}', (yolox_params[i], scores[i]), fontsize=14)


    plt.xlabel('model parameters(M)')
    plt.ylabel('mAP')
    plt.legend(fontsize=12)
    plt.show()

def acc_fps_plot():
    rcnn_fps, rcnn_scores = get_acc_fps(Mode.RCNN)
    yolox_fps, yolox_scores = get_acc_fps(Mode.YOLO)

    plt.figure(figsize=(14, 10))
    for j, (name, scores) in enumerate(rcnn_scores.items()):
        plt.plot(rcnn_fps[name], scores, '--o', markersize=15, label=name)
        for i, backbone in enumerate(['ResNet50', 'ResNet101']):
            x, y = rcnn_fps[name][i], scores[i]
            tx, ty = x, y
            if j == 0:
                ty += 0.0005
            else:
                ty -= 0.0005
            plt.annotate(f'{backbone}',
                         (x, y),
                         (tx, ty),
                         fontsize=14)
    for name, scores in yolox_scores.items():
        plt.plot(yolox_fps[name], scores, '--o', markersize=15, label=name)
        for i, backbone in enumerate(['yolox-tiny', 'yolox-small']):
            plt.annotate(f'{backbone}',
                         (yolox_fps[name][i], scores[i]),
                         (yolox_fps[name][i]-3, scores[i]),
                         fontsize=14)

    plt.xlabel('FPS')
    plt.ylabel('mAP')
    plt.legend(loc='upper left', fontsize=12)
    plt.show()