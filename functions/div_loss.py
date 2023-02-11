import logging

import torch
import torch.autograd as autograd


def div_loss(disc, real_x, fake_x, wp: int = 6, eps: float = 1e-6):
    logging.debug(f'calculating div: real {real_x.mean():.2f}, fake {fake_x.mean():.2f}')
    alpha = torch.rand((real_x.shape[0], 1, 1, 1)).cuda()
    tmp_x = (alpha * real_x + (1 - alpha) * fake_x).requires_grad_(True)
    tmp_y = disc(tmp_x)
    grad = autograd.grad(
        outputs=tmp_y,
        inputs=tmp_x,
        grad_outputs=torch.ones_like(tmp_y),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    grad = grad.view(tmp_x.shape[0], -1) + eps
    div = (grad.norm(2, dim=1) ** wp).mean()
    return div
