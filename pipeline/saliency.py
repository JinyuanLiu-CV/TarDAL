import logging
import socket
import sys
import warnings
from pathlib import Path

import cv2
import torch.hub
from kornia import image_to_tensor, tensor_to_image
from torchvision.transforms import Resize, Compose, Normalize
from tqdm import tqdm

from module.saliency.u2net import U2NETP


class Saliency:
    r"""
    Init saliency detection pipeline to generate mask from infrared images.
    """

    def __init__(self, url: str):
        # init device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logging.info(f'deploy u2net on device {str(device)}')
        self.device = device

        # init u2net small (u2netp)
        net = U2NETP(in_ch=1, out_ch=1)
        logging.info(f'init u2net small model with (1 -> 1)')
        self.net = net

        # download pretrained parameters
        ckpt_p = Path.cwd() / 'weights' / 'v1' / 'u2netp.pth'
        logging.info(f'download pretrained u2net weights from {url}')
        socket.setdefaulttimeout(5)
        try:
            logging.info(f'starting download of pretrained weights from {url}')
            ckpt = torch.hub.load_state_dict_from_url(url, model_dir=ckpt_p.parent, map_location='cpu')
        except Exception as err:
            logging.fatal(f'load {url} failed: {err}, try download pretrained weights manually')
            sys.exit(1)
        net.load_state_dict(ckpt)
        logging.info(f'load pretrained u2net weights from {str(ckpt_p)}')

        # move to device
        net.to(device)

        # more parameters
        self.transform_fn = Compose([Resize(size=(320, 320)), Normalize(mean=0.485, std=0.229)])

    @torch.inference_mode()
    def inference(self, src: str | Path, dst: str | Path):
        # create save folder
        dst = Path(dst)
        dst.mkdir(parents=True, exist_ok=True)
        logging.debug(f'create save folder {str(dst)}')

        # forward
        self.net.eval()
        warnings.filterwarnings(action='ignore', lineno=780)
        img_list = sorted(Path(src).rglob('*.png'))
        logging.info(f'load {len(img_list)} images from {str(src)}')
        process = tqdm(img_list)
        for img_p in process:
            process.set_description(f'generate mask for {img_p.name} to {str(dst)}')
            img = self._imread(img_p).to(self.device)
            reverse_fn = Resize(size=img.shape[-2:])
            img = self.transform_fn(img)
            mask = self.net(img.unsqueeze(0))[0]
            mask = (mask - mask.min()) / (mask.max() - mask.min())
            mask = reverse_fn(mask).squeeze()
            cv2.imwrite(str(dst / img_p.name), tensor_to_image(mask) * 255)

    @staticmethod
    def _imread(img_p: str | Path):
        img = cv2.imread(str(img_p), cv2.IMREAD_GRAYSCALE)
        img = image_to_tensor(img).float() / 255
        return img
