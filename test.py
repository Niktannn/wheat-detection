import os
import uuid
import argparse
import mmcv
import logging

from mmdet.apis import init_detector
from mmdet.models import build_detector
from mmdet.datasets import (build_dataloader, build_dataset,
                            replace_ImageToTensor)
from mmdet.apis import single_gpu_test

from mmdet.utils import (build_dp, compat_cfg,
                         replace_cfg_vals,
                         update_data_root)
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)

import wheat_dataset

def test(config, ckpt, imgs_path = None,
         res_file=None,
         show=False, show_dir=None, show_score_thr=0.3,
         seed=0
         ):
    if isinstance(config, (str)):
        config = mmcv.Config.fromfile(config)
    elif not isinstance(config, mmcv.Config):
        raise TypeError('config must be a filename or Config object, '
                        f'but got {type(config)}')
    if imgs_path:
        tmp_annotations_file = f'ann_{str(uuid.uuid4())}.txt'
        test_files = [os.path.splitext(os.path.basename(f))[0] 
                      for f in os.listdir(imgs_path)]
        with open(tmp_annotations_file, 'w') as f:
            for id in test_files:
                f.write(f"{id}\n")
        config.data.test.ann_file = tmp_annotations_file
        config.data.test.img_prefix = imgs_path
    
    
    config.data.test.test_mode = True

    dataset = build_dataset(config.data.test)
    data_loader = build_dataloader(dataset, 
                                   samples_per_gpu=1, 
                                   workers_per_gpu=1, 
                                   dist=False, 
                                   shuffle=False)
    
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
    
    if res_file:
        mmcv.dump(results, res_file)

    if imgs_path:
        os.remove(tmp_annotations_file)
    
    # return results

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, 
                        help='path to mmdetection model config')
    parser.add_argument('--ckpt', type=str, default=None, 
                        help='path to load weights from')
    parser.add_argument('--imgs_path', type=str, 
                        help='path to images to test (default testing on config.data.test)')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument('--show-dir', 
                        help='directory where painted images will be saved')
    parser.add_argument('--show-score-thr', type=float, default=0.3,
                        help='score threshold (default: 0.3)')
    parser.add_argument('--out', help='output result file in pickle format')
    opt = parser.parse_args()
    test(opt.config, opt.ckpt, opt.imgs_path, 
          opt.out,
          opt.show, opt.show_dir, opt.show_score_thr,
          )