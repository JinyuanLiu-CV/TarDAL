import logging
import sys
from pathlib import Path
from typing import Literal, List, Tuple, Optional

import torch
import torch.backends.cudnn
from kornia.filters import spatial_gradient
from kornia.losses import MS_SSIMLoss, ssim_loss
from numpy import number
from torch import Tensor
from torch.nn.functional import l1_loss

from functions.div_loss import div_loss
from functions.get_param_groups import get_param_groups
from module.fuse.discriminator import Discriminator
from module.fuse.generator import Generator


class Fuse:
    r"""
    Init fuse pipeline to generate fused images from infrared and visible images.
    """

    def __init__(self, config, mode: Literal['train', 'inference']):
        # attach hyper parameters
        self.config = config
        self.mode = mode  # fuse computation mode: train(grad+graph), eval(graph), inference(x)
        modules = []

        # init device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logging.info(f'deploy tardal-fuse on device {str(device)}')
        self.device = device

        # init tardal generator
        f_dim, f_depth = config.fuse.dim, config.fuse.depth
        generator = Generator(dim=f_dim, depth=f_depth)
        modules.append(generator)
        logging.info(f'init generator with (dim: {f_dim} depth: {f_depth})')
        self.generator = generator

        # init tardel discriminator during train mode
        if mode == 'train':
            f_size = config.train.image_size
            dis_t = Discriminator(dim=f_dim, size=f_size)
            dis_d = Discriminator(dim=f_dim, size=f_size)
            modules += [dis_t, dis_d]
            logging.info(f'init discriminators with (dim: {f_dim} size: {f_size})')
            self.dis_t, self.dis_d = dis_t, dis_d

        # load pretrained parameters (optional)
        f_ckpt = config.fuse.pretrained
        if f_ckpt is not None:
            if 'http' in f_ckpt:
                ckpt_p = Path.cwd() / 'weights' / 'v1' / 'tardal.pth'
                url = f_ckpt
                logging.info(f'download pretrained parameters from {url}')
                try:
                    ckpt = torch.hub.load_state_dict_from_url(url, model_dir=ckpt_p.parent, map_location='cpu')
                except Exception as err:
                    logging.fatal(f'connect to {url} failed: {err}, try download pretrained weights manually')
                    sys.exit(1)
            else:
                ckpt = torch.load(f_ckpt, map_location='cpu')
            self.load_ckpt(ckpt)

        # criterion
        if config.loss.fuse.src_fn == 'v1':
            ms_ssim_loss = MS_SSIMLoss()
            modules.append(ms_ssim_loss)
            self.ms_ssim_loss = ms_ssim_loss

        # move to device
        _ = [x.to(device) for x in modules]

        # more parameters
        # WGAN div hyper parameters
        self.wk, self.wp = 2, 6

    def load_ckpt(self, ckpt: dict):
        f_ckpt = ckpt if 'fuse' not in ckpt else ckpt['fuse']

        # check eval mode
        if self.config.inference.use_eval is None:
            if 'use_eval' in f_ckpt:
                logging.warning(f'overwriting inference.use_eval {self.config.inference.use_eval} with {f_ckpt["use_eval"]}')
                self.config.inference.use_eval = f_ckpt['use_eval']
            else:
                logging.warning(f'no use_eval settings found, using default (true)')
                self.config.inference.use_eval = True
        if 'use_eval' in f_ckpt:
            f_ckpt.pop('use_eval')

        # load state dict
        self.generator.load_state_dict(f_ckpt)
        if self.mode == 'train' and 'disc' in ckpt:
            self.dis_t.load_state_dict(ckpt['disc']['t'])
            self.dis_d.load_state_dict(ckpt['disc']['d'])

    def save_ckpt(self) -> dict:
        ckpt = {'fuse': self.generator.state_dict()}
        if self.mode == 'train':
            ckpt |= {'disc': {'t': self.dis_t.state_dict(), 'd': self.dis_t.state_dict()}}
        return ckpt

    def forward(self, ir: Tensor, vi: Tensor) -> Tensor:
        self.generator.train()
        fus = self.generator(ir, vi)
        return fus

    @torch.no_grad()
    def eval(self, ir: Tensor, vi: Tensor) -> Tensor:
        self.generator.eval()
        fus = self.generator(ir, vi)
        return fus

    @torch.inference_mode()
    def inference(self, ir: Tensor, vi: Tensor) -> Tensor:
        if self.config.inference.use_eval:
            self.generator.eval()
        fus = self.generator(ir, vi)
        return fus

    def criterion_dis_t(self, ir: Tensor, vi: Tensor, mk: Tensor) -> Tensor:
        """
        criterion on target discriminator 'ir * m <- pixel distribution -> fus * m'
        """

        logging.debug('criterion on target discriminator')

        # switch to train mode
        self.dis_t.train()

        # sample real & fake
        real_s = ir * mk
        fake_s = self.eval(ir, vi) * mk
        fake_s.detach_()

        # judge value towards real & fake
        real_v = torch.squeeze(self.dis_t(real_s))
        fake_v = torch.squeeze(self.dis_t(fake_s))

        # loss calculate
        real_l, fake_l = -real_v.mean(), fake_v.mean()
        div = div_loss(self.dis_t, real_s, fake_s, self.wp)
        loss = real_l + fake_l + self.wk * div

        return loss

    def criterion_dis_d(self, ir: Tensor, vi: Tensor, mk: Tensor) -> Tensor:
        """
        criterion on detail discriminator 'vi * m <- grad distribution -> fus * (1-m)'
        mask: optional
        """

        logging.debug('criterion on detail discriminator')

        # switch to train mode
        self.dis_d.train()

        # sample real & fake
        mk = mk if self.config.loss.fuse.d_mask else 0  # use mask or not
        real_s = self.gradient(vi) * (1 - mk)
        fake_s = self.gradient(self.eval(ir, vi)) * (1 - mk)
        fake_s.detach_()

        # judge value towards real & fake
        real_v = torch.squeeze(self.dis_d(real_s))
        fake_v = torch.squeeze(self.dis_d(fake_s))

        # loss calculate
        real_l, fake_l = -real_v.mean(), fake_v.mean()
        div = div_loss(self.dis_d, real_s, fake_s, self.wp)
        loss = real_l + fake_l + self.wk * div

        return loss

    def criterion_generator(self, ir: Tensor, vi: Tensor, mk: Tensor, w1: Tensor, w2: Tensor, d_warming: bool = True):
        """
        criterion on generator 'ir, vi <- loss -> fus'
        return: Tuple[Tensor, List[number]] (only fuse), Tuple[Tensor, Tensor, List[number]] (joint mode)
        """

        logging.debug('criterion on generator')

        # forward (train mode for calculate loss)
        fus = self.forward(ir, vi)

        # calculate src and adv loss
        f_loss = self.config.loss.fuse
        src_w, adv_w = f_loss.src, f_loss.adv
        adv_w = 0 if d_warming else adv_w
        src_l = w1 * self.src_loss(fus, ir) + w2 * self.src_loss(fus, vi)
        adv_l, tar_l, det_l = self.adv_loss(fus, mk)
        loss = src_w * src_l.mean() + adv_w * adv_l.mean()

        # only fuse
        return loss, [src_l.mean().item(), adv_l.mean().item(), tar_l, det_l]

    @staticmethod
    def gradient(x: Tensor, eps: float = 1e-8) -> Tensor:
        s = spatial_gradient(x, 'sobel')
        dx, dy = s[:, :, 0, :, :], s[:, :, 1, :, :]
        u = torch.sqrt(torch.pow(dx, 2) + torch.pow(dy, 2) + eps)  # sqrt backwork x range: (0, n]
        return u

    def src_loss(self, x: Tensor, y: Tensor) -> Tensor:
        src_fn = self.config.loss.fuse.src_fn
        match src_fn:
            case 'v0':
                "fus <- 0.01*ssim + 0.99*l1 -> src"
                return 0.01 * ssim_loss(x, y, window_size=11) + 0.99 * l1_loss(x, y)
            case 'v1':
                "fus <- ms-ssim -> src"
                return self.ms_ssim_loss(x, y)
            case _:
                assert NotImplemented, f'unsupported src function: {src_fn}'

    def adv_loss(self, fus: Tensor, mk: Tensor) -> Tuple[Tensor, number, number]:
        # weights
        f_loss = self.config.loss.fuse
        tar_w, det_w = f_loss.t_adv, f_loss.d_adv
        # target loss
        self.dis_t.eval()
        tar_l = -self.dis_t(fus * mk)  # fus * m -> target pixel distribution (max -> -min)
        # detail loss
        self.dis_d.eval()
        mk = mk if self.config.loss.fuse.d_mask else 0  # use mask or not
        det_l = -self.dis_d(self.gradient(fus) * (1 - mk))  # grad(fus) * (1-m) -> grad distribution (max -> -min)
        return tar_w * tar_l + det_w * det_l, tar_l.mean().item(), det_l.mean().item()

    def param_groups(self, key: Optional[Literal['g', 'd']] = None) -> tuple[List, List, List]:
        match key:
            case 'g':
                return self.g_params()
            case 'd':
                return self.d_params()
            case _:
                g_params, d_params = self.g_params(), self.d_params()
                group = [], [], []
                for idx in range(3):
                    group[idx].extend(g_params[idx])
                    group[idx].extend(d_params[idx])
                return group

    def g_params(self) -> tuple[List, List, List]:
        return get_param_groups(self.generator)

    def d_params(self) -> tuple[List, List, List]:
        group = [], [], []
        for module in [self.dis_t, self.dis_d]:
            tmp = get_param_groups(module)
            for idx in range(3):
                group[idx].extend(tmp[idx])
        return group
