import argparse
from pathlib import Path
from typing import Optional

import cv2
import torch
from kornia import image_to_tensor, tensor_to_image
from torch import Tensor
from torchvision.utils import draw_bounding_boxes
from tqdm import tqdm

import loader
from loader.utils.reader import label_read


def data_preview(img_f: str | Path, lbl_f: str | Path, dst_f: str | Path, dataset: Optional[str] = None):
    # create dst
    dst_f = Path(dst_f)
    dst_f.mkdir(parents=True, exist_ok=True)
    # images list
    img_f, lbl_f = Path(img_f), Path(lbl_f)
    img_l = sorted([x.stem for x in img_f.glob('*.png')])
    # dataset settings
    classes, palette = [], []
    if dataset is not None:
        dataset = getattr(loader, dataset)
        classes = dataset.classes
        palette = dataset.palette
    t_l = tqdm(img_l)
    for stem in t_l:
        t_l.set_description(f'draw on {stem}')
        lbl = label_read(lbl_f / f'{stem}.txt')
        img = image_to_tensor(cv2.imread(str(img_f / f'{stem}.png')))
        lbl[:, 1:] *= Tensor([img.shape[-1], img.shape[-2], img.shape[-1], img.shape[-2]])
        boxes = [x[1:] for x in lbl]
        if dataset is not None:
            cls = [classes[int(x[0])] for x in lbl]
            colors = [palette[int(x[0])] for x in lbl]
            img = draw_bounding_boxes(img, torch.stack(boxes, dim=0), cls, colors, width=3)
        else:
            img = draw_bounding_boxes(img, torch.stack(boxes, dim=0), width=3)
        cv2.imwrite(str(dst_f / f'{stem}.png'), tensor_to_image(img))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('data preview')
    parser.add_argument('--img', help='image folder')
    parser.add_argument('--lbl', help='label folder')
    parser.add_argument('--dst', help='mask output folder (we will create it if not exists)')
    parser.add_argument('--cls', required=False, help='dataset type (random if not specified)')
    config = parser.parse_args()
    data_preview(img_f=config.img, lbl_f=config.lbl, dst_f=config.dst, dataset=config.cls)
