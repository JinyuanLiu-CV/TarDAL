import argparse
import logging
from functools import reduce
from pathlib import Path

import torch
import wandb
import yaml
from kornia.metrics import AverageMeter
from torch import Tensor
from torch.optim import AdamW, Adam, SGD
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm

import loader
from config import from_dict, ConfigDict
from pipeline.fuse import Fuse
from tools.dict_to_device import dict_to_device


class TrainF:
    def __init__(self, config: str | Path | ConfigDict, wandb_key: str):
        # init logger
        log_f = '%(asctime)s | %(filename)s[line:%(lineno)d] | %(levelname)s | %(message)s'
        logging.basicConfig(level='INFO', format=log_f)
        logging.info(f'TarDAL-v1 Training Script')

        # init config
        if isinstance(config, str) or isinstance(config, Path):
            config = yaml.safe_load(Path(config).open('r'))
            config = from_dict(config)  # convert dict to object
        else:
            config = config
        self.config = config

        # debug mode
        if config.debug.fast_run:
            logging.warning('fast run mode is on, only for debug!')

        # wandb run
        wandb.login(key=wandb_key)  # wandb api key
        runs = wandb.init(project='TarDAL-v1', config=config, mode=config.debug.wandb_mode)
        self.runs = runs

        # init save folder
        save_dir = Path(self.config.save_dir) / self.runs.id
        save_dir.mkdir(parents=True, exist_ok=True)
        self.save_dir = save_dir
        logging.info(f'model weights will be saved to {str(save_dir)}')

        # init pipeline
        fuse = Fuse(config, mode='train')
        self.fuse = fuse

        # freeze & grad
        for k, v in fuse.generator.named_parameters():
            v.requires_grad = True  # train all layers
            if any(x in k for x in config.train.freeze):
                logging.info(f'freezing {k}')
                v.requires_grad = False

        # init optimizer
        o_cfg = config.optimizer
        fuse_pg = fuse.param_groups()  # [weight(with decay), weight(no decay), bias]
        groups = [
            {'params': fuse_pg[0], 'lr': o_cfg.lr_i, 'weight_decay': o_cfg.weight_decay},
            {'params': fuse_pg[1], 'lr': o_cfg.lr_i, 'weight_decay': 0},
        ]
        match o_cfg.name:
            case 'sgd':
                optimizer = SGD(fuse_pg[2], lr=o_cfg.lr_i, momentum=o_cfg.momentum, nesterov=True)
            case 'adam':
                optimizer = Adam(fuse_pg[2], lr=o_cfg.lr_i, betas=(o_cfg.momentum, 0.999))
            case 'adamw':
                optimizer = AdamW(fuse_pg[2], lr=o_cfg.lr_i, betas=(o_cfg.momentum, 0.999), weight_decay=0)
            case _:
                optimizer = None
                assert NotImplemented, f'unsupported optimizer: {o_cfg.name}'
        self.optimizer = optimizer
        self.optimizer.add_param_group(groups[0])
        self.optimizer.add_param_group(groups[1])

        # init scheduler
        lr_fn = lambda x: (1 - x / config.train.epochs) * (1 - o_cfg.lr_f) + o_cfg.lr_f
        self.scheduler = LambdaLR(self.optimizer, lr_lambda=lr_fn)

        # init dataset & dataloader
        data_t = getattr(loader, config.dataset.name)  # dataset type
        t_dataset = data_t(root=config.dataset.root, mode='train', config=config)
        v_dataset = data_t(root=config.dataset.root, mode='val', config=config)
        self.t_loader = DataLoader(
            t_dataset, batch_size=config.train.batch_size, shuffle=True,
            collate_fn=data_t.collate_fn, pin_memory=True, num_workers=config.train.num_workers,
        )
        self.v_loader = DataLoader(
            v_dataset, batch_size=config.train.batch_size,
            collate_fn=data_t.collate_fn, pin_memory=True, num_workers=config.train.num_workers,
        )

    def run(self):
        # epochs & eval interval & save interval
        epochs = self.config.train.epochs
        e_interval = self.config.train.eval_interval
        s_interval = self.config.train.save_interval
        # start training process
        for epoch in range(1, epochs + 1):
            # train
            t_l = tqdm(self.t_loader, disable=False, total=len(self.t_loader) if not self.config.debug.fast_run else 3, ncols=120)
            g_history = [AverageMeter() for _ in range(5)]  # tot, src, adv, tar, det
            disc_history = AverageMeter(), AverageMeter()  # target, detail
            log_dict = {}
            for sample in t_l:
                sample = dict_to_device(sample, self.fuse.device)
                # train generator
                g_loss, [src_l, adv_l, tar_l, det_l] = self.fuse.criterion_generator(
                    ir=sample['ir'], vi=sample['vi'],
                    mk=sample['mask'],
                    w1=sample['ir_w'], w2=sample['vi_w'],
                    d_warming=epoch <= self.config.loss.fuse.d_warm,
                )
                g_history[0].update(g_loss.item())
                _ = [g_history[idx + 1].update(v) for idx, v in enumerate([src_l, adv_l, tar_l, det_l])]
                self.optim(g_loss)
                # train target discriminator
                d_t_loss = self.fuse.criterion_dis_t(
                    ir=sample['ir'], vi=sample['vi'],
                    mk=sample['mask'],
                )
                disc_history[0].update(d_t_loss.item())
                self.optim(d_t_loss)
                # train detail discriminator
                d_d_loss = self.fuse.criterion_dis_d(
                    ir=sample['ir'], vi=sample['vi'],
                    mk=sample['mask'],
                )
                disc_history[1].update(d_d_loss.item())
                self.optim(d_d_loss)
                # fast run (jump out)
                if self.config.debug.fast_run and t_l.n > 2:
                    logging.info('fast mode: jump')
                    break
            # train logs
            g_l, src_l, adv_l, tar_l, det_l = [g_history[i].avg for i in range(5)]
            d_t_l, d_d_l = disc_history[0].avg, disc_history[1].avg
            log_dict |= {'g/tot': g_l, 'g/src': src_l, 'g/adv': adv_l, 'g/tar': d_t_l, 'g/det': d_d_l, 'disc/tar': tar_l, 'disc/det': det_l}
            logging.info(f'Epoch {epoch}/{epochs} | Generator Loss: {g_l:.4f} | Source Loss: {src_l:.4f} | Adversarial Loss: {adv_l:.4f}')

            # eval (fuse: show in wandb)
            if epoch % e_interval == 0 or self.config.debug.fast_run:
                e_l = tqdm(self.v_loader, disable=True)
                for sample in e_l:
                    sample = dict_to_device(sample, self.fuse.device)
                    fus = self.fuse.eval(ir=sample['ir'], vi=sample['vi'])
                    log_dict |= {'fuse': wandb.Image(fus), 'mask': wandb.Image(sample['mask'])}
                    break
            # update scheduler and show lr
            log_dict |= reduce(lambda x, y: x | y, [{f'lr_{i}': v['lr']} for i, v in enumerate(self.optimizer.param_groups)])
            self.scheduler.step()

            # update wandb
            self.runs.log(log_dict)
            # save model
            if epoch % s_interval == 0 or self.config.debug.fast_run:
                ckpt = self.fuse.save_ckpt()
                torch.save(ckpt, self.save_dir / f'{str(epoch).zfill(5)}.pth')
                logging.info(f'Epoch {epoch}/{epochs} | Model Saved')

    def optim(self, loss: Tensor):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', default='config/default.yaml', help='config file path')
    parser.add_argument('--auth', help='wandb auth api key')
    args = parser.parse_args()
    train = TrainF(args.cfg, args.auth)
    train.run()
