from pathlib import Path
from typing import Tuple

import cv2
import numpy
import torch
from kornia import image_to_tensor, tensor_to_image
from kornia.color import rgb_to_ycbcr, bgr_to_rgb, rgb_to_bgr
from torch import Tensor
from torchvision.ops import box_convert


def gray_read(img_path: str | Path) -> Tensor:
    img_n = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    img_t = image_to_tensor(img_n).float() / 255
    return img_t


def ycbcr_read(img_path: str | Path) -> Tuple[Tensor, Tensor]:
    img_n = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    img_t = image_to_tensor(img_n).float() / 255
    img_t = rgb_to_ycbcr(bgr_to_rgb(img_t))
    y, cbcr = torch.split(img_t, [1, 2], dim=0)
    return y, cbcr


def label_read(label_path: str | Path) -> Tensor:
    target = numpy.loadtxt(str(label_path), dtype=numpy.float32)
    labels = torch.from_numpy(target).view(-1, 5)  # (cls, cx, cy, w, h)
    labels[:, 1:] = box_convert(labels[:, 1:], 'cxcywh', 'xyxy')  # (cls, x1, y1, x2, y2)
    return labels


def img_write(img_t: Tensor, img_path: str | Path):
    if img_t.shape[0] == 3:
        img_t = rgb_to_bgr(img_t)
    img_n = tensor_to_image(img_t.squeeze().cpu()) * 255
    cv2.imwrite(str(img_path), img_n)


def label_write(pred_i: Tensor, txt_path: str | Path):
    for *pos, conf, cls in pred_i.tolist():
        line = (cls, *pos, conf)
        with txt_path.open('a') as f:
            f.write(('%g ' * len(line)).rstrip() % line + '\n')
