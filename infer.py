import argparse
import logging
from pathlib import Path

import torch.backends.cudnn
import yaml

import scripts
from config import from_dict

if __name__ == '__main__':
    # args parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', default='config/default.yaml', help='config file path')
    parser.add_argument('--save_dir', default='runs/tmp', help='fusion result save folder')
    args = parser.parse_args()

    # init config
    config = yaml.safe_load(Path(args.cfg).open('r'))
    config = from_dict(config)  # convert dict to object
    config = config

    # init logger
    log_f = '%(asctime)s | %(filename)s[line:%(lineno)d] | %(levelname)s | %(message)s'
    logging.basicConfig(level=config.debug.log, format=log_f)

    # init device & anomaly detector
    torch.backends.cudnn.benchmark = True
    torch.autograd.set_detect_anomaly(True)

    # choose inference script
    logging.info(f'enter {config.strategy} inference mode')
    match config.strategy:
        case 'fuse':
            infer_p = getattr(scripts, 'InferF')
            # check pretrained weights
            if config.fuse.pretrained is None:
                logging.warning('no pretrained weights specified, use official pretrained weights')
                config.fuse.pretrained = 'https://github.com/JinyuanLiu-CV/TarDAL/releases/download/v1.0.0/tardal-dt.pth'
        case 'fuse & detect':
            infer_p = getattr(scripts, 'InferFD')
            # check pretrained weights
            if config.fuse.pretrained is None:
                logging.warning('no pretrained weights specified, use official pretrained weights')
                config.fuse.pretrained = 'https://github.com/JinyuanLiu-CV/TarDAL/releases/download/v1.0.0/tardal-ct.pth'
        case 'detect':
            raise NotImplementedError('detect mode is useless during inference period, please use fuse & detect mode')
        case _:
            raise ValueError(f'unknown strategy: {config.strategy}')

    # create script instance
    infer = infer_p(config, args.save_dir)
    infer.run()
