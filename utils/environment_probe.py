import logging
import sys

import torch


class EnvironmentProbe:
    """
    Detects the configuration of the environment and returns devices status.
    """

    def __init__(self):
        python_v = sys.version.split()[0]
        pytorch_v = torch.__version__
        cuda_s = torch.cuda.is_available()
        device = torch.cuda.current_device() if cuda_s else 'cpu'
        device_n = torch.cuda.get_device_name(device)
        logging.info(f'python: {python_v} | pytorch: {pytorch_v} | gpu: {device_n if cuda_s else False}')
        self.device = device

    def memory_status(self):
        if not torch.cuda.is_available():
            return {'current': 'unavailable', 'max': 'unavailable'}
        memory_a = torch.cuda.memory_allocated(self.device) / 1024 ** 3
        memory_ma = torch.cuda.max_memory_allocated(self.device) / 1024 ** 3
        logging.debug(f'memory: {memory_a:.2f}GB (history max: {memory_ma:.2f}GB)')
        return {'current': memory_a, 'max': memory_ma}


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    probe = EnvironmentProbe()
    probe.memory_status()
