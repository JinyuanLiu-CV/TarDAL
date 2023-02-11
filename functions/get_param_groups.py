from typing import List

from torch import nn


def get_param_groups(module) -> tuple[List, List, List]:
    group = [], [], []
    bn = tuple(v for k, v in nn.__dict__.items() if 'Norm' in k)  # normalization layers
    for v in module.modules():
        if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):
            "bias"
            group[2].append(v.bias)
        if isinstance(v, bn):
            "weight (no decay)"
            group[1].append(v.weight)
        elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):
            "weight (with decay)"
            group[0].append(v.weight)
    return group
