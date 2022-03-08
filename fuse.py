import argparse
from argparse import Namespace

import torch
from pathlib import Path

from modules.generator import Generator
from pipeline.eval import Eval


def parse_opt() -> Namespace:
    parser = argparse.ArgumentParser()

    # universal opt
    parser.add_argument('--src', type=str, help='fusion data root path')
    parser.add_argument('--dst', type=str, help='fusion images save path')
    parser.add_argument('--weights', type=str, default='weights/tardal.pt', help='pretrained weights path')
    parser.add_argument('--color', action='store_true', help='colorize fused images with visible color channels')
    parser.add_argument('--from_checkpoint', action='store_true', help='load from checkpoint')

    # fusion opt
    parser.add_argument('--dim', type=int, default=32, help='feature dimension')
    parser.add_argument('--depth', type=int, default=3, help='network dense depth')
    parser.add_argument('--cudnn', action='store_true', help='accelerate network forward with cudnn')
    parser.add_argument('--eval', action='store_true', help='use eval mode for new pytorch models')
    parser.add_argument('--half', action='store_true', help='use half mode for new pytorch models')

    return parser.parse_args()


def img_filter(x: Path) -> bool:
    return x.suffix in ['.png', '.bmp', '.jpg']


if __name__ == '__main__':
    config = parse_opt()

    # init model
    net = Generator(dim=config.dim, depth=config.depth)

    # load pretrained weights
    ck_pt = torch.load(config.weights)
    net.load_state_dict(ck_pt if not config.from_checkpoint else ck_pt['g'])

    # images
    root = Path(config.src)
    ir_paths = [x for x in sorted((root / 'ir').glob('*')) if img_filter]
    vi_paths = [x for x in sorted((root / 'vi').glob('*')) if img_filter]

    # fuse
    f = Eval(net, cudnn=config.cudnn, half=config.half, eval=config.eval)
    f(ir_paths, vi_paths, Path(config.dst), config.color)
