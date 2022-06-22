import logging
import random
from pathlib import Path
from typing import Literal, List, Optional

import torch
from kornia.geometry import vflip, hflip, resize
from torch import Tensor, Size
from torch.utils.data import Dataset
from torchvision.ops import box_convert
from torchvision.transforms import Resize
from torchvision.utils import draw_bounding_boxes

from config import ConfigDict
from loader.utils.checker import check_mask, check_image, check_labels, check_iqa, get_max_size
from loader.utils.reader import gray_read, ycbcr_read, label_read, img_write, label_write
from tools.scenario_reader import scenario_counter, generate_meta


class M3FD(Dataset):
    type = 'fuse & detect'  # dataset type: 'fuse' or 'fuse & detect'
    color = True  # dataset visible format: false -> 'gray' or true -> 'color'
    classes = ['People', 'Car', 'Bus', 'Lamp', 'Motorcycle', 'Truck']
    palette = ['#FF0000', '#C1C337', '#2FA7B4', '#F541C4', '#F84F2C', '#7D2CC8']

    generate_meta_lock = False  # generate meta once

    def __init__(self, root: str | Path, mode: Literal['train', 'val', 'pred'], config: ConfigDict):
        super().__init__()
        root = Path(root)
        self.root = root
        self.mode = mode
        self.config = config

        # check json meta config
        if M3FD.generate_meta_lock is False:
            if Path(root / 'meta' / 'scenario.json').exists():
                logging.info('found scenario.json, generating train & val list.')
                scenario_counter(root / 'meta' / 'scenario.json')
                generate_meta(root)
                M3FD.generate_meta_lock = True
            else:
                logging.warning('not found scenario.json, using current train & val list.')

        # read corresponding list
        img_list = Path(root / 'meta' / f'{mode}.txt').read_text().splitlines()
        logging.info(f'load {len(img_list)} images from {root.name}')
        self.img_list = img_list

        # check images
        check_image(root, img_list)

        # check labels
        self.labels = check_labels(root, img_list)

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
        # image name, like '028.png'
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

        # load label
        label_p = Path(name).stem + '.txt'
        labels = label_read(self.root / 'labels' / label_p)

        # concat images for transform(s)
        t = torch.cat([ir, vi, mask, ir_w, vi_w, cbcr], dim=0)

        # transform (resize)
        resize_fn = Resize(size=self.config.train.image_size)
        t = resize_fn(t)

        # transform (flip up-down)
        if random.random() < self.config.dataset.detect.flip_ud:
            t = vflip(t)
            if len(labels):
                labels[:, 2] = 1 - labels[:, 2]

        # transform (flip left-right)
        if random.random() < self.config.dataset.detect.flip_lr:
            t = hflip(t)
            if len(labels):
                labels[:, 1] = 1 - labels[:, 1]

        # transform labels (cls, x1, y1, x2, y2) -> (0, cls, ...)
        labels_o = torch.zeros((len(labels), 6))
        if len(labels):
            labels_o[:, 1:] = labels

        # unpack images
        ir, vi, mask, ir_w, vi_w, cbcr = torch.split(t, [1, 1, 1, 1, 1, 2], dim=0)

        # merge data
        sample = {
            'name': name,
            'ir': ir, 'vi': vi,
            'ir_w': ir_w, 'vi_w': vi_w, 'mask': mask, 'cbcr': cbcr,
            'labels': labels_o
        }

        # return as expected
        return sample

    def pred_item(self, index: int) -> dict:
        # image name, like '028.png'
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
    def pred_save(fus: Tensor, names: List[str | Path], shape: List[Size], pred: Optional[Tensor] = None, save_txt: bool = False):
        if pred is None:
            return M3FD.pred_save_no_boxes(fus, names, shape)
        return M3FD.pred_save_with_boxes(fus, names, shape, pred, save_txt)

    @staticmethod
    def pred_save_no_boxes(fus: Tensor, names: List[str | Path], shape: List[Size]):
        for img_t, img_p, img_s in zip(fus, names, shape):
            img_t = resize(img_t, img_s)
            img_write(img_t, img_p)

    @staticmethod
    def pred_save_with_boxes(fus: Tensor, names: List[str | Path], shape: List[Size], pred: Tensor, save_txt: bool = False):
        for img_t, img_p, img_s, pred_i in zip(fus, names, shape, pred):
            # reshape target
            cur_s = img_t.shape[1:]
            scale_x, scale_y = cur_s[1] / img_s[1], cur_s[0] / img_s[0]
            pred_i[:, :4] *= Tensor([scale_x, scale_y, scale_x, scale_y]).to(pred_i.device)
            # reshape image
            img_t = resize(img_t, img_s)
            img = (img_t.clamp_(0, 1) * 255).to(torch.uint8)
            # draw bounding box
            pred_x = list(filter(lambda x: x[4] > 0.6, pred_i))
            boxes = [x[:4] for x in pred_x]
            cls_idx = [int(x[5].cpu().numpy()) for x in pred_x]
            labels = [f'{M3FD.classes[cls]}: {x[4].cpu().numpy():.2f}' for cls, x in zip(cls_idx, pred_x)]
            colors = [M3FD.palette[cls] for cls, x in zip(cls_idx, pred_x)]
            if len(boxes):
                img = draw_bounding_boxes(img, torch.stack(boxes, dim=0), labels, colors, width=2)
            img = img.float() / 255
            # save labeled images
            img_p = Path(img_p.parent) / 'images' / img_p.name
            img_write(img, img_p)
            # save label txt
            if save_txt:
                txt_p = Path(str(img_p.parent).replace('images', 'labels')) / (img_p.stem + '.txt')
                txt_p.unlink(missing_ok=True)
                txt_p.touch()
                pred_i[:, :4] /= Tensor([img_s[1], img_s[0], img_s[1], img_s[0]]).to(pred_i.device)
                pred_i[:, :4] = box_convert(pred_i[:, :4], 'xyxy', 'cxcywh')
                label_write(pred_i, txt_p)

    @staticmethod
    def collate_fn(data: List[dict]) -> dict:
        # keys
        keys = data[0].keys()
        # merge
        new_data = {}
        for key in keys:
            k_data = [d[key] for d in data]
            match key:
                case 'name' | 'shape':
                    # (name, name)
                    new_data[key] = k_data
                case 'labels':
                    # (labels, image_index)
                    for i, lb in enumerate(k_data):
                        lb[:, 0] = i
                    new_data[key] = torch.cat(k_data, dim=0)
                case _:
                    # (img, img)
                    new_data[key] = torch.stack(k_data, dim=0)
        # return as expected
        return new_data
