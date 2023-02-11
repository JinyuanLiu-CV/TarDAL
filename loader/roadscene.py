import logging
from pathlib import Path
from typing import Literal, List

import torch
from kornia.geometry import resize
from torch import Tensor, Size
from torch.utils.data import Dataset
from torchvision.transforms import Resize

from config import ConfigDict
from loader.utils.checker import check_mask, check_image, check_iqa, get_max_size
from loader.utils.reader import gray_read, ycbcr_read, img_write


class RoadScene(Dataset):
    type = 'fuse'  # dataset type: 'fuse' or 'fuse & detect'
    color = True  # dataset visible format: false -> 'gray' or true -> 'color'

    def __init__(self, root: str | Path, mode: Literal['train', 'val', 'pred'], config: ConfigDict):
        super().__init__()
        root = Path(root)
        self.root = root
        self.mode = mode

        # read corresponding list
        img_list = Path(root / 'meta' / f'{mode}.txt').read_text().splitlines()
        logging.info(f'load {len(img_list)} images from {root.name}')
        self.img_list = img_list

        # check images
        check_image(root, img_list)

        # more check
        match mode:
            case 'train' | 'val':
                # check mask cache
                check_mask(root, img_list, config)
                # check iqa cache
                check_iqa(root, img_list, config)
            case _:
                # get max shape
                self.max_size = get_max_size(root, img_list)

        # choose transform
        match mode:
            case 'train' | 'val':
                self.transform_fn = Resize(size=config.train.image_size)
            case _:
                self.transform_fn = Resize(size=self.max_size)

    def __len__(self) -> int:
        return len(self.img_list)

    def __getitem__(self, index: int) -> dict:
        # choose get item method
        match self.mode:
            case 'train' | 'val':
                return self.train_val_item(index)
            case _:
                return self.pred_item(index)

    def train_val_item(self, index: int) -> dict:
        # image name, like '003.png'
        name = self.img_list[index]
        logging.debug(f'train-val mode: loading item {name}')

        # load infrared and visible
        ir = gray_read(self.root / 'ir' / name)
        vi, cbcr = ycbcr_read(self.root / 'vi' / name)

        # load mask
        mask = gray_read(self.root / 'mask' / name)

        # load information measurement
        ir_w = gray_read(self.root / 'iqa' / 'ir' / name)
        vi_w = gray_read(self.root / 'iqa' / 'vi' / name)

        # transform (resize)
        t = torch.cat([ir, vi, mask, ir_w, vi_w, cbcr], dim=0)
        ir, vi, mask, ir_w, vi_w, cbcr = torch.split(self.transform_fn(t), [1, 1, 1, 1, 1, 2], dim=0)

        # merge data
        sample = {'name': name, 'ir': ir, 'vi': vi, 'ir_w': ir_w, 'vi_w': vi_w, 'mask': mask, 'cbcr': cbcr}

        # return as expected
        return sample

    def pred_item(self, index: int) -> dict:
        # image name, like '003.png'
        name = self.img_list[index]
        logging.debug(f'pred mode: loading item {name}')

        # load infrared and visible
        ir = gray_read(self.root / 'ir' / name)
        vi, cbcr = ycbcr_read(self.root / 'vi' / name)

        # transform (resize)
        s = ir.shape[1:]
        t = torch.cat([ir, vi, cbcr], dim=0)
        ir, vi, cbcr = torch.split(self.transform_fn(t), [1, 1, 2], dim=0)

        # merge data
        sample = {'name': name, 'ir': ir, 'vi': vi, 'cbcr': cbcr, 'shape': s}

        # return as expected
        return sample

    @staticmethod
    def pred_save(fus: Tensor, names: List[str | Path], shape: List[Size]):
        for img_t, img_p, img_s in zip(fus, names, shape):
            img_t = resize(img_t, img_s)
            img_write(img_t, img_p)

    @staticmethod
    def collate_fn(data: List[dict]) -> dict:
        # keys
        keys = data[0].keys()
        # merge
        new_data = {}
        for key in keys:
            k_data = [d[key] for d in data]
            new_data[key] = k_data if isinstance(k_data[0], str) or isinstance(k_data[0], Size) else torch.stack(k_data)
        # return as expected
        return new_data
