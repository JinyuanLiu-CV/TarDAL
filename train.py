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
    parser.add_argument('--auth', help='wandb auth api key')
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

    # choose train script
    logging.info(f'enter {config.strategy} train mode')
    match config.strategy:
        case 'fuse':
            train_p = getattr(scripts, 'TrainF')
        case 'detect':
            if config.loss.bridge.fuse != 0:
                logging.warning('overwrite fuse loss weight to 0')
                config.loss.bridge.fuse = 0
            train_p = getattr(scripts, 'TrainFD')
        case 'fuse & detect':
            train_p = getattr(scripts, 'TrainFD')
        case _:
            raise ValueError(f'unknown strategy: {config.strategy}')

    # create script instance
    train = train_p(config, wandb_key=args.auth)
    train.run()
