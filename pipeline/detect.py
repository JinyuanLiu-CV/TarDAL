import logging
import sys
from pathlib import Path
from typing import Literal, List, Tuple

import numpy
import torch
import torch.backends.cudnn
from numpy import number
from torch import Tensor, nn
from torchvision.ops import box_convert
from torchvision.utils import draw_bounding_boxes

from functions.get_param_groups import get_param_groups
from module.detect.models.common import Conv
from module.detect.models.yolo import Model
from module.detect.utils.general import labels_to_class_weights, non_max_suppression
from module.detect.utils.loss import ComputeLoss
from module.detect.utils.metrics import box_iou


class Detect:
    r"""
    Init detect pipeline to detect objects from fused images.
    """

    def __init__(self, config, mode: Literal['train', 'inference'], nc: int, classes: List[str], labels: List[Tensor]):
        # attach hyper parameters
        self.config = config
        self.mode = mode  # fuse computation mode: train(grad+graph), eval(graph), inference(x)

        # init device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logging.info(f'deploy {config.detect.model} on device {str(device)}')
        self.device = device

        # init yolo model
        model_t = config.detect.model
        config_p = Path(__file__).parents[1] / 'module' / 'detect' / 'models' / f'{model_t}.yaml'
        net = Model(cfg=config_p, ch=config.detect.channels, nc=nc).to(self.device)
        logging.info(f'init {model_t} with (nc: {nc})')
        self.net = net

        # init hyperparameters
        hyp = config.loss.detect
        nl = net.model[-1].nl  # number of detection layers

        # model parameters
        hyp['box'] *= 3 / nl  # scale to layers
        hyp['cls'] *= nc / 80 * 3 / nl  # scale to classes and layers
        hyp['obj'] *= (config.train.image_size[0] / 640) ** 2 * 3 / nl  # scale to image size and layers
        hyp['label_smoothing'] = False  # label smoothing

        # attach constants
        net.nc = nc  # attach number of classes to model
        net.hyp = hyp  # attach hyper parameters to model
        net.class_weights = labels_to_class_weights(labels, nc).to(self.device)  # attach class weights
        net.names = classes

        # load pretrained parameters (optional)
        d_ckpt = config.detect.pretrained
        if d_ckpt is not None:
            if 'http' in d_ckpt:
                ckpt_p = Path.cwd() / 'weights' / 'v1' / 'tardal.pth'
                url = d_ckpt
                logging.info(f'download pretrained parameters from {url}')
                try:
                    ckpt = torch.hub.load_state_dict_from_url(url, model_dir=ckpt_p.parent, map_location='cpu')
                except Exception as err:
                    logging.fatal(f'connect to {url} failed: {err}, try download pretrained weights manually')
                    sys.exit(1)
            else:
                ckpt = torch.load(d_ckpt, map_location='cpu')
            self.load_ckpt(ckpt)

        # criterion (reference: YOLOv5 official)
        self.loss = ComputeLoss(net)

    def load_ckpt(self, ckpt: dict):
        ckpt = ckpt if 'detect' not in ckpt else ckpt['detect']
        self.net.load_state_dict(ckpt)

    def load_ckpt_fuse(self, ckpt: dict):
        ckpt = ckpt if 'detect' not in ckpt else ckpt['detect']
        # fuse conv & bn
        self.net.fuse()
        # compatibility updates
        for m in self.net.modules():
            t = type(m)
            if t in (nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU, Detect, Model):
                m.inplace = True  # torch 1.7.0 compatibility
                if t is Detect and not isinstance(m.anchor_grid, list):
                    delattr(m, 'anchor_grid')
                    setattr(m, 'anchor_grid', [torch.zeros(1)] * m.nl)
            elif t is Conv:
                m._non_persistent_buffers_set = set()  # torch 1.6.0 compatibility
            elif t is nn.Upsample and not hasattr(m, 'recompute_scale_factor'):
                m.recompute_scale_factor = None  # torch 1.11.0 compatibility
        # return as expect
        self.net.load_state_dict(ckpt)

    def save_ckpt(self) -> dict:
        ckpt = {'detect': self.net.state_dict()}
        return ckpt

    def forward(self, imgs: Tensor) -> Tensor:
        self.net.train()
        pred = self.net(imgs)
        return pred

    @torch.no_grad()
    def eval(self, imgs: Tensor, targets: Tensor, stats: List, preview: bool = False) -> Tuple[int, Tensor | None]:
        self.net.eval()

        # forward
        preds, _ = self.net(imgs)  # (xyxy, conf, cls) [h, w]

        # convert pred format
        batch_size, _, height, width = imgs.shape
        targets[:, 2:] *= torch.tensor((width, height, width, height), device=self.device)  # (id, cls, xyxy) [1, 1] -> [h, w]
        preds = non_max_suppression(preds, conf_thres=0.001, iou_thres=0.6, labels=[], multi_label=True)  # (xyxy, conf, cls) [h, w]

        # const
        iou_v = torch.linspace(0.5, 0.95, 10).to(self.device)  # iou vector for mAP@0.5:0.95
        n_iou = iou_v.numel()

        # record
        seen = 0

        # statistics per images
        for si, pred in enumerate(preds):
            labels = targets[targets[:, 0] == si, 1:]  # (cls, xyxy) [h, w]

            num_l, num_p = labels.shape[0], pred.shape[0]
            correct = torch.zeros(num_p, n_iou, dtype=torch.bool, device=self.device)
            seen += 1

            # no pred result
            if num_p == 0:
                if num_l:
                    stats.append((correct, *torch.zeros((2, 0), device=self.device), labels[:, 0]))
                continue

            # predictions
            pred_n = pred.clone()

            # evaluate
            if num_l:
                t_box = labels[:, 1:5]  # (xyxy) [h, w]
                labels_n = torch.cat((labels[:, 0:1], t_box), 1)  # (xyxy, cls) [h, w]
                correct = self.process_batch(pred_n, labels_n, iou_v)

            # update stats matrix
            stats.append((correct, pred[:, 4], pred[:, 5], labels[:, 0]))  # (correct, conf, pcls, tcls)

        # preview
        if preview:
            prv = self.preview(imgs, preds)
            return seen, prv

        # return as expected
        return seen, None

    @torch.inference_mode()
    def inference(self, imgs: Tensor) -> Tensor:
        self.net.eval()
        # forward
        preds, _ = self.net(imgs)
        # convert pred format
        batch_size, _, height, width = imgs.shape
        preds = non_max_suppression(preds, conf_thres=0.001, iou_thres=0.6, multi_label=True)  # [xyxy, conf, cls]
        # return as expected
        return preds

    def criterion(self, imgs: Tensor, targets: Tensor) -> Tuple[Tensor, List[number]]:
        """
        criterion on detector
        """

        logging.debug('criterion on yolo')

        # forward
        pred = self.forward(imgs)  # (bs, 3, 80, 80, class + 5)

        # calculate loss
        targets[:, 2:] = box_convert(targets[:, 2:], 'xyxy', 'cxcywh')  # (idx, cls, x1, y1, x2, y2) -> (idx, cls, cx, cy, w, h)
        loss, loss_items = self.loss(pred, targets.to(self.device))

        return loss, [x.item() for x in loss_items]

    @staticmethod
    def preview(imgs: Tensor, preds: Tensor, conf_th: float = 0.6):
        imgs_mk = []
        # preds: (xyxy, conf, cls)

        # mark on images
        for img, pred in zip(imgs, preds):
            pred = list(filter(lambda x: x[4] > conf_th, pred))
            logging.debug(f'detect {len(pred)} on images')
            img = (img * 255).type(torch.uint8)
            boxes = [x[:4] for x in pred]
            cls = [int(x[5].cpu().numpy()) for x in pred]
            labels = [f'{[cls]}: {x[4].cpu().numpy():.2f}' for cls, x in zip(cls, pred)]
            if len(boxes):
                img = draw_bounding_boxes(img, torch.stack(boxes, dim=0), labels=labels, width=2)
            imgs_mk.append((img / 255).float().to(imgs.device))

        # fill or crop to 9 images
        if len(imgs_mk) > 9:
            imgs_mk = imgs_mk[:9]
        elif len(imgs_mk) < 9:
            zeros = [torch.zeros_like(imgs_mk[0], device=imgs[0].device) for _ in range(9 - len(imgs_mk))]
            imgs_mk = imgs_mk + zeros

        # merge images(9, 3, h, w) to one image (3, 3h, 3w)
        imgs_mk = torch.stack(imgs_mk, dim=0)
        imgs_c = []
        for i in range(3):
            t = [imgs_mk[i * 3 + j] for j in range(3)]  # [(3, h, w), (3, h, w), (3, h, w)]
            imgs_c.append(torch.cat(t, dim=2))  # (3, h, 3w)
        imgs_one = torch.cat(imgs_c, dim=1)  # (3, 3h, 3w)

        # return as expected
        return imgs_one

    def param_groups(self) -> tuple[List, List, List]:
        group = [], [], []
        tmp = get_param_groups(self.net)
        for idx in range(3):
            group[idx].extend(tmp[idx])
        return group

    @staticmethod
    def process_batch(detections, labels, iou_v):
        """
        Return correct predictions' matrix. Both sets of boxes are in (x1, y1, x2, y2) format.
        Arguments:
            detections (Array[N, 6]), x1, y1, x2, y2, conf, class
            labels (Array[M, 5]), class, x1, y1, x2, y2
            iou_v (Array[10]), iou thresholds
        Returns:
            correct (Array[N, 10]), for 10 IoU levels
        """
        correct = torch.zeros(detections.shape[0], iou_v.shape[0], dtype=torch.bool, device=iou_v.device)
        iou = box_iou(labels[:, 1:], detections[:, :4])
        x = torch.where((iou >= iou_v[0]) & (labels[:, 0:1] == detections[:, 5]))  # IoU above threshold and classes match
        if x[0].shape[0]:
            matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()  # [label, detection, iou]
            if x[0].shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[numpy.unique(matches[:, 1], return_index=True)[1]]
                # matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[numpy.unique(matches[:, 0], return_index=True)[1]]
            matches = torch.Tensor(matches).to(iou_v.device)
            correct[matches[:, 1].long()] = matches[:, 2:3] >= iou_v
        return correct
