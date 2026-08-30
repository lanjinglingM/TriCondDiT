import typing as tp
import torch

from torch.nn import functional as F
from torch import nn

class LossModule(nn.Module):
    def __init__(self, name: str, weight: float = 1.0, decay = 1.0):
        super().__init__()

        self.name = name
        self.decay = float(decay)
        self.master_weight = weight
        weight = torch.tensor(float(weight))
        self.register_buffer('weight', weight)

    def decay_weight(self):
        if self.decay != 1.0:
            self.weight *= self.decay
        elif self.decay == 1.0 and self.weight != self.master_weight:
            self.weight = torch.tensor(self.master_weight, dtype=self.weight.dtype, device=self.weight.device)
    def forward(self, info, *args, **kwargs):
        raise NotImplementedError

class MSELoss(LossModule):
    def __init__(self, key_a: str, key_b: str, weight: float = 1.0, mask_key: str = None, name: str = 'mse_loss',decay = 1.0):
        super().__init__(name=name, weight=weight, decay=decay)

        self.key_a = key_a
        self.key_b = key_b

        self.mask_key = mask_key

    def forward(self, info):
        mse_loss = F.mse_loss(info[self.key_a], info[self.key_b], reduction='none')

        if self.mask_key is not None and self.mask_key in info and info[self.mask_key] is not None:
            mask = info[self.mask_key]

            if mask.ndim == 2 and mse_loss.ndim == 3:
                mask = mask.unsqueeze(1)

            if mask.shape[1] != mse_loss.shape[1]:
                mask = mask.repeat(1, mse_loss.shape[1], 1)

            mse_loss = mse_loss[mask]

        mse_loss = mse_loss.mean()
        self.decay_weight()
        return self.weight * mse_loss

class MultiLoss(nn.Module):
    def __init__(self, losses: tp.List[LossModule]):
        super().__init__()

        self.losses = nn.ModuleList(losses)

    def forward(self, info):
        total_loss = 0

        losses = {}

        for loss_module in self.losses:
            module_loss = loss_module(info)
            total_loss += module_loss
            losses[loss_module.name] = module_loss

        return total_loss, losses
