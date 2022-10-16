import torch
from mmcv import Config

from mmdet.models import build_detector
from mmcv.cnn import get_model_complexity_info


def get_flops(config, shape):
    if len(shape) == 1:
        h = w = shape[0]
    elif len(shape) == 2:
        h, w = shape
    else:
        raise ValueError('invalid input shape')

    if isinstance(config, str):
        config = Config.fromfile(config)
    elif not isinstance(config, Config):
        raise TypeError('config must be a filename or Config object, '
                        f'but got {type(config)}')

    input_shape = (3, h, w)
    model = build_detector(
        config.model,
        train_cfg=config.get('train_cfg'),
        test_cfg=config.get('test_cfg'))
    if torch.cuda.is_available():
        model.cuda()
    model.eval()

    if hasattr(model, 'forward_dummy'):
        model.forward = model.forward_dummy
    else:
        raise NotImplementedError(
            'FLOPs counter is currently not currently supported with {}'.
            format(model.__class__.__name__))

    flops, params = get_model_complexity_info(model, input_shape, 
        print_per_layer_stat=False)
    return flops, params
