import logging
from pathlib import Path

import torch
import yaml
from kornia.color import ycbcr_to_rgb
from torch.utils.data import DataLoader
from tqdm import tqdm

import loader
from config import ConfigDict, from_dict
from pipeline.fuse import Fuse
from tools.dict_to_device import dict_to_device


class InferF:
    def __init__(self, config: str | Path | ConfigDict, save_dir: str | Path):
        # init logger
        log_f = '%(asctime)s | %(filename)s[line:%(lineno)d] | %(levelname)s | %(message)s'
        logging.basicConfig(level='INFO', format=log_f)
        logging.info(f'TarDAL-v1 Inference Script')

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

        # create save(output) folder
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        logging.info(f'create save folder {str(save_dir)}')
        self.save_dir = save_dir

        # init dataset & dataloader
        data_t = getattr(loader, config.dataset.name)  # dataset type
        self.data_t = data_t
        p_dataset = data_t(root=config.dataset.root, mode='pred', config=config)
        self.p_loader = DataLoader(
            p_dataset, batch_size=config.inference.batch_size, shuffle=False,
            collate_fn=data_t.collate_fn, pin_memory=True, num_workers=config.inference.num_workers,
        )

        # init pipeline
        fuse = Fuse(config, mode='inference')
        self.fuse = fuse

    @torch.inference_mode()
    def run(self):
        p_l = tqdm(self.p_loader, total=len(self.p_loader), ncols=120)
        for sample in p_l:
            sample = dict_to_device(sample, self.fuse.device)
            # f_net forward
            fus = self.fuse.inference(ir=sample['ir'], vi=sample['vi'])
            # recolor
            if self.data_t.color and self.config.inference.grayscale is False:
                fus = torch.cat([fus, sample['cbcr']], dim=1)
                fus = ycbcr_to_rgb(fus)
            # save images
            self.data_t.pred_save(
                fus, [self.save_dir / name for name in sample['name']],
                shape=sample['shape']
            )
