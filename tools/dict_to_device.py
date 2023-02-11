from typing import Dict

from torch import Tensor
from torch.types import Device


def dict_to_device(d: Dict, device: Device) -> Dict | None:
    if d is None:
        return None
    for k, v in d.items():
        if isinstance(v, Tensor):
            d[k] = d[k].to(device)
    return d
