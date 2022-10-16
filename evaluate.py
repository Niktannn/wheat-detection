import mmcv
import os
import argparse
import time
import logging

from mmdet.apis import init_detector
from mmdet.models import build_detector
from mmdet.datasets import build_dataloader, build_dataset
from mmdet.apis import single_gpu_test
from mmdet.utils import (build_dp, compat_cfg,
                         replace_cfg_vals,
                         update_data_root)
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint)

import wheat_dataset
from utils.split_data import write_ann_file


def evaluate(config, ckpt,
             show=False, show_dir=None, show_score_thr=0.3,
             eval_metrics=('mAP',),
             out=None
             ):
    if isinstance(config, (str)):
        config = mmcv.Config.fromfile(config)
    elif not isinstance(config, mmcv.Config):
        raise TypeError('config must be a filename or Config object, '
                        f'but got {type(config)}')

    all_imgs_ids = [os.path.splitext(os.path.basename(f))[0] for f in os.listdir('data/train')]
    all_ids_file = 'data/all.txt'
    write_ann_file(all_ids_file, all_imgs_ids)

    config.data.test.ann_file = all_ids_file
    config.data.test.img_prefix = 'data/train'

    dataset = build_dataset(config.data.test)

    data_loader = build_dataloader(dataset,
                                   samples_per_gpu=config.data.samples_per_gpu,
                                   workers_per_gpu=2,
                                   dist=False,
                                   shuffle=False)

    # model = init_detector(config, ckpt, device='cuda:0')
    # build the model and load checkpoint
    config.model.train_cfg = None
    model = build_detector(config.model, test_cfg=config.get('test_cfg'))

    # suppress any output except error messages
    logger = logging.getLogger("mechanize")
    logger.setLevel(logging.ERROR)
    checkpoint = load_checkpoint(model, ckpt, logger=logger, map_location='cpu')

    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = dataset.CLASSES

    model = build_dp(model, config.device, device_ids=config.gpu_ids)

    results = single_gpu_test(model,
                              data_loader,
                              show=show,
                              out_dir=show_dir,
                              show_score_thr=show_score_thr)

    eval_kwargs = config.get('evaluation', {}).copy()
    # hard-code way to remove EvalHook args
    for key in [
        'interval', 'tmpdir', 'start', 'gpu_collect', 'save_best',
        'rule', 'dynamic_intervals'
    ]:
        eval_kwargs.pop(key, None)
    eval_kwargs.update(dict(metric=eval_metrics))
    metric = dataset.evaluate(results, **eval_kwargs)
    if out:
        metric_dict = dict(config=config.dump(), metric=metric)
        mmcv.mkdir_or_exist(os.path.abspath(out))
        timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        json_file = os.path.join(out, f'eval_{timestamp}.json')
        mmcv.dump(metric_dict, json_file)

    os.remove(all_ids_file)

    return metric


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str,
                        help='path to mmdetection model config')
    parser.add_argument('--ckpt', type=str, default=None,
                        help='path to load weights from')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument('--show-dir',
                        help='directory where painted images will be saved')
    parser.add_argument('--show-score-thr', type=float, default=0.3,
                        help='score threshold (default: 0.3)')
    parser.add_argument('--out', type=str,
                        help='path to directory where results will be saved')

    opt = parser.parse_args()
    evaluate(opt.config, opt.ckpt,
             opt.show, opt.show_dir, opt.show_score_thr,
             eval_metrics=('mAP',),
             out=opt.out
             )