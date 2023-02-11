import logging
from functools import reduce
from pathlib import Path

import torch
import wandb
from kornia.filters import SpatialGradient
from kornia.losses import SSIMLoss
from kornia.metrics import AverageMeter
from torch import nn, Tensor
from torch.optim import RMSprop
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from functions.div_loss import div_loss
from modules.discriminator import Discriminator
from modules.generator import Generator
from utils.environment_probe import EnvironmentProbe
from utils.fusion_data import FusionData


class Train:
    """
    The train process for TarDAL.
    """

    def __init__(self, environment_probe: EnvironmentProbe, config: dict):
        logging.info(f'TarDAL Training | mask: {config.mask} | weight: {config.weight} | adv: {config.adv_weight}')
        self.config = config
        self.environment_probe = environment_probe

        # modules
        logging.info(f'generator | dim: {config.dim} | depth: {config.depth}')
        self.generator = Generator(config.dim, config.depth)
        logging.info(f'discriminator | dim: {config.dim} | size: {config.size}')
        self.dis_target = Discriminator(config.dim, (config.size, config.size))
        self.dis_detail = Discriminator(config.dim, (config.size, config.size))

        # WGAN adam optim
        logging.info(f'RMSprop | learning rate: {config.learning_rate}')
        self.opt_generator = RMSprop(self.generator.parameters(), lr=config.learning_rate)
        self.opt_dis_target = RMSprop(self.dis_target.parameters(), lr=config.learning_rate)
        self.opt_dis_detail = RMSprop(self.dis_detail.parameters(), lr=config.learning_rate)

        # move to device
        logging.info(f'module device: {environment_probe.device}')
        self.generator.to(environment_probe.device)
        self.dis_target.to(environment_probe.device)
        self.dis_detail.to(environment_probe.device)

        # loss
        self.l1 = nn.L1Loss(reduction='none')
        self.ssim = SSIMLoss(window_size=11, reduction='none')
        self.l1.cuda()
        self.ssim.cuda()

        # functions
        self.spatial = SpatialGradient('diff')

        # WGAN div hyper parameters
        self.wk, self.wp = 2, 6

        # datasets
        folder = Path(config.folder)
        resize = transforms.Resize((config.size, config.size))
        dataset = FusionData(folder, config.mask, 'train', transforms=resize)
        self.dataloader = DataLoader(dataset, config.batch_size, True, num_workers=config.num_workers, pin_memory=True)
        logging.info(f'dataset | folder: {str(folder)} | size: {len(self.dataloader) * config.batch_size}')

    def train_dis_target(self, ir: Tensor, vi: Tensor, mk: Tensor) -> Tensor:
        """
        Train target discriminator for 'ir * m <- pixel -> fus * m'
        """

        logging.debug('train target discriminator')
        # switch to train mode
        self.dis_target.train()

        # sample real & fake
        real_s = ir * mk
        self.generator.eval()
        fake_s = self.generator(ir, vi).detach() * mk

        # judge value towards real & fake
        real_v = torch.squeeze(self.dis_target(real_s))
        fake_v = torch.squeeze(self.dis_target(fake_s))

        # loss calculate
        real_l, fake_l = -real_v.mean(), fake_v.mean()
        div = div_loss(self.dis_target, real_s, fake_s, self.wp)
        loss = real_l + fake_l + self.wk * div

        # backward
        self.opt_dis_target.zero_grad()
        loss.backward()
        self.opt_dis_target.step()

        return loss.item()

    def train_dis_detail(self, ir: Tensor, vi: Tensor, mk: Tensor) -> Tensor:
        """
        Train detail discriminator for 'vi * (1-m) <- Grad -> fus * (1-m)'
        """

        logging.debug('train detail discriminator')
        # switch to train mode
        self.dis_detail.train()

        # sample real & fake
        real_s = self.gradient(vi * (1 - mk))
        self.generator.eval()
        fake_s = self.gradient(self.generator(ir, vi).detach() * (1 - mk))

        # judge value towards real & fake
        real_v = torch.squeeze(self.dis_detail(real_s))
        fake_v = torch.squeeze(self.dis_detail(fake_s))

        # loss calculate
        real_l, fake_l = -real_v.mean(), fake_v.mean()
        div = div_loss(self.dis_detail, real_s, fake_s, self.wp)
        loss = real_l + fake_l + self.wk * div

        # backward
        self.opt_dis_detail.zero_grad()
        loss.backward()
        self.opt_dis_detail.step()

        return loss.item()

    def gradient(self, x: Tensor, eps: float = 1e-6) -> Tensor:
        s = self.spatial(x)
        dx, dy = s[:, :, 0, :, :], s[:, :, 1, :, :]
        u = torch.sqrt(torch.pow(dx, 2) + torch.pow(dy, 2) + eps)
        return u

    def train_generator(self, ir: Tensor, vi: Tensor, mk: Tensor, s1: Tensor, s2: Tensor) -> dict:
        """
        Train generator 'ir + vi -> fus'
        """

        logging.debug('train generator')
        self.generator.train()
        fus = self.generator(ir, vi)

        # calculate loss towards criterion
        b1, b2, b3 = self.config.weight  # b1 * ssim + b2 * l1 + b3 * adv

        l_ir = b1 * self.ssim(fus, ir) + b2 * self.l1(fus, ir)
        l_vi = b1 * self.ssim(fus, vi) + b2 * self.l1(fus, vi)

        w1, w2 = 0.5 + 0.5 * (s1 - s2), 0.5 + 0.5 * (s2 - s1)  # data driven loss weights
        l_src = w1 * l_ir + w2 * l_vi  # fus <- ssim + l1 -> (ir, vi)
        l_src = l_src.mean()

        self.dis_target.eval()
        l_target = -self.dis_target(fus * mk).mean()  # judge target: fus * m
        self.dis_detail.eval()
        l_detail = -self.dis_detail(self.gradient(fus * (1 - mk))).mean()  # judge detail: Grad(fus * (1-mk))

        c1, c2 = self.config.adv_weight  # c1 * l_target + c2 * l_detail
        l_adv = c1 * l_target + c2 * l_detail

        loss = l_src + b3 * l_adv

        # backward
        self.opt_generator.zero_grad()
        loss.backward()
        self.opt_generator.step()

        # loss state
        state = {
            'g_loss': loss.item(),
            'g_src_ir': l_ir.mean().item(),
            'g_src_vi': l_vi.mean().item(),
            'g_adv_target': l_target.item(),
            'g_adv_detail': l_detail.item(),
        }

        return state

    def run(self):
        for epoch in range(1, self.config.epochs + 1):
            process = tqdm(enumerate(self.dataloader), disable=not self.config.debug)
            meter = AverageMeter()
            for idx, sample in process:
                ir, vi, mk = sample['ir'], sample['vi'], sample['mk']
                s1, s2 = sample['vsm']['ir'], sample['vsm']['vi']
                im = torch.cat([ir, vi, mk, s1, s2], dim=1)
                im = im.to(self.environment_probe.device)
                ir, vi, mk, s1, s2 = torch.chunk(im, 5, dim=1)

                g_loss = self.train_generator(ir, vi, mk, s1, s2)
                d_target_loss = self.train_dis_target(ir, vi, mk)
                d_detail_loss = self.train_dis_detail(ir, vi, mk)

                process.set_description(f'g: {g_loss["g_loss"]:03f} | d: {d_target_loss:03f}, {d_detail_loss:03f}')
                meter.update(Tensor(list(g_loss.values()) + [d_target_loss] + [d_detail_loss]))

            keys = ['g_loss', 'g_src_ir', 'g_src_vi', 'g_adv_t', 'g_adv_d', 'd_t', 'd_d']
            state = reduce(lambda x, y: x | y, [{k: v} for k, v in zip(keys, meter.avg)])
            print(state)
            wandb.log(state)
            if epoch % 5 == 0:
                self.save(epoch)

    def save(self, epoch: int):
        path = Path(self.config.cache) / self.config.id
        path.mkdir(parents=True, exist_ok=True)
        cache = path / f'{epoch:03d}.pth'
        logging.info(f'save checkpoint to {str(cache)}')
        state = {
            'g': self.generator.state_dict(),
            'd': {
                't': self.dis_target.state_dict(),
                'd': self.dis_target.state_dict(),
            },
            'opt': {
                'g': self.opt_generator.state_dict(),
                't': self.opt_dis_target.state_dict(),
                'd': self.opt_dis_detail.state_dict(),
            },
        }
        torch.save(state, cache)
