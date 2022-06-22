import argparse
import logging
import sys
from functools import reduce
from itertools import chain
from pathlib import Path

import numpy
import torch
import wandb
import yaml
from kornia.color import ycbcr_to_rgb
from kornia.metrics import AverageMeter
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm

import loader
from config import from_dict, ConfigDict
from module.detect.utils.metrics import ap_per_class
from pipeline.detect import Detect
from pipeline.fuse import Fuse
from scripts.utils.smart_optimizer import smart_optimizer
from tools.dict_to_device import dict_to_device


class TrainFD:
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
        save_dir = Path(config.save_dir) / runs.id
        save_dir.mkdir(parents=True, exist_ok=True)
        self.save_dir = save_dir
        logging.info(f'model weights will be saved to {str(save_dir)}')

        # load dataset
        data_t = getattr(loader, config.dataset.name)
        self.data_t = data_t
        t_dataset = data_t(root=config.dataset.root, mode='train', config=config)
        v_dataset = data_t(root=config.dataset.root, mode='val', config=config)
        if 'detect' not in t_dataset.type:
            logging.fatal(f'dataset {config.dataset.name} not support detect')
            sys.exit(1)
        self.t_loader = DataLoader(
            t_dataset, batch_size=config.train.batch_size, shuffle=True,
            collate_fn=data_t.collate_fn, pin_memory=True, num_workers=config.train.num_workers,
        )
        self.v_loader = DataLoader(
            v_dataset, batch_size=config.train.batch_size,
            collate_fn=data_t.collate_fn, pin_memory=True, num_workers=config.train.num_workers,
        )

        # init pipeline
        fuse = Fuse(config, mode='train')
        self.fuse = fuse
        detect = Detect(config, mode='train', nc=len(t_dataset.classes), classes=t_dataset.classes, labels=t_dataset.labels)
        self.detect = detect

        # freeze & grad
        for k, v in chain(fuse.generator.named_parameters(), detect.net.named_parameters()):
            v.requires_grad = True  # train all layers
            if any(x in k for x in config.train.freeze):
                logging.info(f'freezing {k}')
                v.requires_grad = False

        # init fuse optimizer
        o_cfg = config.optimizer
        f_p, d_p = fuse.param_groups('g'), detect.param_groups()
        self.fd_opt = smart_optimizer(o_cfg, tuple(f_p[i] + d_p[i] for i in range(3)))
        self.disc_opt = smart_optimizer(o_cfg, fuse.param_groups('d'), lr=o_cfg.lr_d)

        # init scheduler
        lr_fn = lambda x: (1 - x / config.train.epochs) * (1 - o_cfg.lr_f) + o_cfg.lr_f
        self.lr_fn = lr_fn
        self.scheduler = LambdaLR(self.fd_opt, lr_lambda=lr_fn)

        # hyperparameters check
        # bridge warm & scheduler warm phase.0
        if config.loss.bridge.warm != config.scheduler.warmup_epochs[0]:
            logging.warning(f'overwriting bridge warm {config.loss.bridge.warm} with {config.scheduler.warmup_epochs[0]}')
            config.loss.bridge.warm = config.scheduler.warmup_epochs[0]
        # discriminator warm & bridge warm
        if config.loss.fuse.d_warm >= config.loss.bridge.warm / 2:
            logging.warning(f'overwriting discriminator warm {config.loss.fuse.d_warm} with {round(config.loss.bridge.warm / 2)}')
            config.loss.fuse.d_warm = round(config.loss.bridge.warm / 2)

    def run(self):
        # epochs & eval interval & save interval
        epochs = self.config.train.epochs
        e_interval = self.config.train.eval_interval
        s_interval = self.config.train.save_interval
        # history switch
        best_map = -1

        # start training process
        l_opt_shot = -1
        n_batch_size = 64
        accumulate = max(round(n_batch_size / self.config.train.batch_size), 1)
        for epoch in range(1, epochs + 1):
            # train
            t_l = tqdm(self.t_loader, disable=False, total=len(self.t_loader) if not self.config.debug.fast_run else 3, ncols=120)
            # recorder
            g_history = AverageMeter()  # generator total loss
            f_history = [AverageMeter() for _ in range(5)]  # fuse loss: tot, src, adv, tar, det
            d_history = [AverageMeter() for _ in range(4)]  # detect loss: tot, box, obj, cls
            disc_history = AverageMeter(), AverageMeter()  # discriminator loss: target, detail
            log_dict = {}
            # warm up shots, max(warmup_epochs, 100 shots)
            w_config = self.config.scheduler
            w_shots_0 = max(round(w_config.warmup_epochs[0] * len(self.t_loader)), 100)  # bridge warm
            w_shots_1 = max(round(w_config.warmup_epochs[1] * len(self.t_loader)), 100)  # normal warm
            w_shots = (w_shots_0, w_shots_1)
            # process
            self.fd_opt.zero_grad()
            for idx, sample in enumerate(t_l):
                # warm up
                c_shots = idx + len(self.t_loader) * (epoch - 1)
                if c_shots < w_shots[0]:
                    for jdx, x in enumerate(self.fd_opt.param_groups):
                        x['lr'] = w_config.warmup_bias_lr if jdx == 0 else 0
                        if 'momentum' in x:
                            x['momentum'] = w_config.warmup_momentum
                if w_shots[0] <= c_shots < w_shots[1]:
                    x_shot = [c_shots, w_shots[1] + c_shots]
                    # accumulate = max(1, numpy.interp(c_shots, x_shot, [1, n_batch_size / self.config.train.batch_size]).round())
                    for jdx, x in enumerate(self.fd_opt.param_groups):
                        o_config = self.config.optimizer
                        # bias lr falls from 0.1 to lr_i, all other lrs rise from 0.0 to lr_i
                        w_range = [w_config.warmup_bias_lr if jdx == 0 else 0, x['initial_lr'] * self.lr_fn(epoch - 1)]
                        x['lr'] = numpy.interp(c_shots, x_shot, w_range)
                        if 'momentum' in x:
                            x['momentum'] = numpy.interp(c_shots, x_shot, [w_config.warmup_momentum, o_config.momentum])
                lr_s = [x['lr'] for x in self.fd_opt.param_groups]
                logging.debug(f'adjust lr {lr_s[0]:.6f} {lr_s[1]:.6f} {lr_s[2]:.6f}')

                # forward
                sample = dict_to_device(sample, self.fuse.device)

                # train generator
                # ir & vi -> f_net -> fus -> d_net -> obj
                # loss: fus -> src + adv, obj -> ground truth
                # f_net forward and cal loss
                f_loss, [src_l, adv_l, tar_l, det_l] = self.fuse.criterion_generator(
                    ir=sample['ir'], vi=sample['vi'],
                    mk=sample['mask'],
                    w1=sample['ir_w'], w2=sample['vi_w'],
                    d_warming=epoch <= self.config.loss.fuse.d_warm,
                )
                fus = self.fuse.eval(ir=sample['ir'], vi=sample['vi'])
                if epoch <= self.config.loss.bridge.warm:
                    fus.detach_()  # (det -> det, fuse -> fuse, det no-> fuse)
                # recolor
                if self.data_t.color:
                    fus = torch.cat([fus, sample['cbcr']], dim=1)
                    fus = ycbcr_to_rgb(fus)
                # d_net forward and cal loss
                d_loss, [box_l, obj_l, cls_l] = self.detect.criterion(
                    imgs=fus,
                    targets=sample['labels'],
                )
                # merge loss
                b_c = self.config.loss.bridge
                g_loss = b_c['fuse'] * f_loss + b_c['detect'] * d_loss  # generator total loss
                g_history.update(g_loss.item())
                _ = [f_history[idx].update(v) for idx, v in enumerate([f_loss.item(), src_l, adv_l, tar_l, det_l])]
                _ = [d_history[idx].update(v) for idx, v in enumerate([d_loss.item(), box_l, obj_l, cls_l])]
                # optimize
                g_loss.backward()
                if c_shots - l_opt_shot >= accumulate:
                    clip_grad_norm_(chain(self.fuse.generator.parameters(), self.detect.net.parameters()), max_norm=10.0)
                    self.fd_opt.step()
                    self.fd_opt.zero_grad()
                    l_opt_shot = c_shots
                    logging.debug(f'optimize f+d | shots: {c_shots} | accumulate: {accumulate} | last: {l_opt_shot}')

                # train target discriminator
                d_t_loss = self.fuse.criterion_dis_t(
                    ir=sample['ir'], vi=sample['vi'],
                    mk=sample['mask'],
                )
                disc_history[0].update(d_t_loss.item())
                self.disc_opt.zero_grad()
                d_t_loss.backward()
                self.disc_opt.step()

                # train detail discriminator
                d_d_loss = self.fuse.criterion_dis_d(
                    ir=sample['ir'], vi=sample['vi'],
                    mk=sample['mask'],
                )
                disc_history[1].update(d_d_loss.item())
                self.disc_opt.zero_grad()
                d_d_loss.backward()
                self.disc_opt.step()

                # update description
                t_l.set_description(f'{epoch}/{epochs} | g: {g_history.avg:.4f} | f: {f_history[0].avg:.4f} | d: {d_history[0].avg:.4f}')

                # fast run (jump out)
                if self.config.debug.fast_run and t_l.n > 2:
                    logging.info('fast mode: jump')
                    break

            # train logs
            # fuse loss
            f_l, src_l, adv_l, tar_l, det_l = [f_history[idx].avg for idx in range(5)]
            log_dict |= {'fus/tot': f_l, 'fus/src': src_l, 'fus/adv': adv_l, 'fus/tar': tar_l, 'fus/det': det_l}
            # detect loss
            d_l, box_l, obj_l, cls_l = [d_history[idx].avg for idx in range(4)]
            log_dict |= {'det/tot': d_l, 'det/box': box_l, 'det/obj': obj_l, 'det/cls': cls_l}
            # generator loss
            g_l = g_history.avg
            log_dict |= {'gen/tot': g_l, 'gen/fus': f_l, 'gen/det': d_l}
            # discriminator loss
            d_t_l, d_d_l = [disc_history[idx].avg for idx in range(2)]
            log_dict |= {'disc/tar': d_t_l, 'disc/det': d_d_l}
            # learning rate
            lrs = [x['lr'] for x in self.fd_opt.param_groups]
            log_dict |= {'lr/0': lrs[0], 'lr/1': lrs[1], 'lr/2': lrs[2]}
            # log to console
            logging.info(f'Epoch {epoch}/{epochs} | Generator Loss: {g_l:.4f} | Fuse loss: {f_l:.4f} | Detect loss: {d_l:.4f}')

            # update scheduler
            self.scheduler.step()

            # eval (fuse & detect: print result in wandb)
            if epoch % e_interval == 0 or self.config.debug.fast_run:
                e_l = tqdm(self.v_loader, disable=False, total=len(self.v_loader) if not self.config.debug.fast_run else 3, ncols=120)

                # matrix
                seen = 0
                dt, p, r, f1, mp, mr, map50, map_all = [0.0, 0.0, 0.0], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
                j_dict, stats, ap50, ap, ap_class = [], [], [], [], []

                # process
                for sample in e_l:
                    sample = dict_to_device(sample, self.fuse.device)
                    # f_net
                    fus = self.fuse.eval(ir=sample['ir'], vi=sample['vi'])
                    # recolor
                    if self.data_t.color:
                        fus = torch.cat([fus, sample['cbcr']], dim=1)
                        fus = ycbcr_to_rgb(fus)
                    # d_net
                    seen_x, preview = self.detect.eval(imgs=fus, targets=sample['labels'], stats=stats, preview='detect' not in log_dict)
                    seen += seen_x
                    if preview is not None and 'detect' not in log_dict:
                        log_dict |= {'detect': wandb.Image(preview)}
                    # fast run (jump out)
                    if self.config.debug.fast_run and t_l.n > 2:
                        logging.info('fast mode: jump')
                        break

                # compute statistics
                stats = [torch.cat(x, 0).cpu().numpy() for x in zip(*stats)]
                names = reduce(lambda x, y: x | y, [{idx: name} for idx, name in enumerate(self.data_t.classes)])
                if len(stats) and stats[0].any():
                    tp, fp, p, r, f1, ap, ap_class = ap_per_class(*stats, names=names)
                    ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
                    mp, mr, map50, map_all = p.mean(), r.mean(), ap50.mean(), ap.mean()
                num_t = numpy.bincount(stats[3].astype(int), minlength=len(self.data_t.classes))  # number of targets per class
                if num_t.sum() == 0:
                    logging.warning(f'no labels found, can not compute metrics without labels.')

                # eval logs
                log_dict |= {'eval/precision': mp, 'eval/recall': mr, 'eval/map50': map50, 'eval/map': map_all}
                # log to console (per class)
                logging.info(f'Epoch {epoch}/{epochs} | Precision: {mp:.4f} | Recall: {mr:.4f} | mAP50: {map50:.4f} | mAP: {map_all:.4f}')
                if len(stats) and len(self.data_t.classes) > 1:
                    for i, c in enumerate(ap_class):
                        logging.info(
                            f'{names[c]} | tot: {num_t[c]} | p: {p[i]:.4f} | r: {r[i]:.4f} | ap50: {ap50[i]:.4f} | ap: {ap[i]:.4f}'
                        )

                # mark best
                if map_all > best_map:
                    best_map = map_all
                    Path(self.save_dir / 'meta.txt').write_text(f'best_map: {best_map:.4f} | epoch: {epoch}')
                    ckpt = self.fuse.save_ckpt() | self.detect.save_ckpt()
                    torch.save(ckpt, self.save_dir / f'{str(epoch).zfill(5)}-{best_map:.4f}.pth')

            # update wandb
            self.runs.log(log_dict)
            # save model
            if epoch % s_interval == 0 or self.config.debug.fast_run:
                ckpt = self.fuse.save_ckpt() | self.detect.save_ckpt()
                torch.save(ckpt, self.save_dir / f'{str(epoch).zfill(5)}.pth')
                logging.info(f'Epoch {epoch}/{epochs} | Model Saved')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', default='config/default.yaml', help='config file path')
    parser.add_argument('--auth', help='wandb auth api key')
    args = parser.parse_args()
    train = TrainFD(args.cfg, args.auth)
    train.run()
