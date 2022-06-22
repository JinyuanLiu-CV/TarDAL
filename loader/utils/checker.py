import logging
import sys
from pathlib import Path
from typing import List

from torch import Tensor, Size
from tqdm import tqdm

from config import ConfigDict
from loader.utils.reader import label_read, gray_read
from pipeline.iqa import IQA
from pipeline.saliency import Saliency


def check_image(root: Path, img_list: List[str]):
    assert (root / 'ir').exists() and (root / 'vi').exists(), f'ir and vi folders are required'
    for img_name in img_list:
        if not (root / 'ir' / img_name).exists() or not (root / 'vi' / img_name).exists():
            logging.fatal(f'empty img {img_name} in {root.name}')
            sys.exit(1)
    logging.info('find all images on list')


def check_iqa(root: Path, img_list: List[str], config: ConfigDict):
    iqa_cache = True
    if (root / 'iqa').exists():
        for img_name in img_list:
            if not (root / 'iqa' / 'ir' / img_name).exists() or not (root / 'iqa' / 'vi' / img_name).exists():
                iqa_cache = False
                break
    else:
        iqa_cache = False
    if iqa_cache:
        logging.info(f'find iqa cache in folder, skip information measurement')
    else:
        logging.info(f'find no iqa cache in folder, start information measurement')
        iqa = IQA(url=config.iqa.url)
        iqa.inference(src=root, dst=root / 'iqa')


def check_labels(root: Path, img_list: List[str]) -> List[Tensor]:
    assert (root / 'labels').exists(), f'labels folder is required'
    labels = []
    for img_name in img_list:
        label_name = Path(img_name).stem + '.txt'
        if not (root / 'labels' / label_name).exists():
            logging.fatal(f'empty label {label_name} in {root.name}')
            sys.exit(1)
        labels.append(label_read(root / 'labels' / label_name))
    logging.info('find all labels on list')
    return labels


def check_mask(root: Path, img_list: List[str], config: ConfigDict):
    mask_cache = True
    if (root / 'mask').exists():
        for img_name in img_list:
            if not (root / 'mask' / img_name).exists():
                mask_cache = False
                break
    else:
        mask_cache = False
    if mask_cache:
        logging.info('find mask cache in folder, skip saliency detection')
    else:
        logging.info('find no mask cache in folder, start saliency detection')
        saliency = Saliency(url=config.saliency.url)
        saliency.inference(src=root / 'ir', dst=root / 'mask')


def get_max_size(root: Path, img_list: List[str]):
    max_h, max_w = -1, -1
    logging.info('find suitable size for prediction')
    img_l = tqdm(img_list)
    for img_name in img_l:
        img_l.set_description('finding suitable size')
        img = gray_read(root / 'ir' / img_name)
        max_h = max(max_h, img.shape[1])
        max_w = max(max_w, img.shape[2])
    logging.info(f'max size in dataset: H:{max_h} x W:{max_w}')
    return Size((max_h, max_w))
