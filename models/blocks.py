import math
import torch
from torch import nn
from torch.nn import functional as F


class FourierFeatures(nn.Module):
    def __init__(self, in_features, out_features, std=1.):
        super().__init__()
        assert out_features % 2 == 0
        self.weight = nn.Parameter(torch.randn(
            [out_features // 2, in_features]) * std)

    def forward(self, input):
        f = 2 * math.pi * input @ self.weight.T
        return torch.cat([f.cos(), f.sin()], dim=-1)


class SPADEZero1dGate(nn.Module):

    def __init__(self, mask_dim, embedding_dim, kernel_size=3):
        super().__init__()
        mid = embedding_dim // 2

        self.mlp_shared = nn.Sequential(
            nn.Conv1d(mask_dim, mid, kernel_size, padding=kernel_size//2),
            nn.GELU()
        )

        self.mlp_gamma = nn.Conv1d(mid, embedding_dim, kernel_size, padding=kernel_size//2)
        self.mlp_beta  = nn.Conv1d(mid, embedding_dim, kernel_size, padding=kernel_size//2)
        self.mlp_gate  = nn.Conv1d(mid, embedding_dim, kernel_size, padding=kernel_size//2)

        nn.init.zeros_(self.mlp_gamma.weight)
        nn.init.zeros_(self.mlp_gamma.bias)
        nn.init.zeros_(self.mlp_beta.weight)
        nn.init.zeros_(self.mlp_beta.bias)

        nn.init.zeros_(self.mlp_gate.weight)
        nn.init.constant_(self.mlp_gate.bias, -10.0)

    def forward(self, x, mask_features):
        if mask_features.shape[-1] != x.shape[-1]:
            mask_features = F.interpolate(mask_features, size=x.shape[-1], mode='nearest')

        act = self.mlp_shared(mask_features)
        gamma = self.mlp_gamma(act)
        beta  = self.mlp_beta(act)
        gate  = torch.sigmoid(self.mlp_gate(act))

        delta = x * gamma + beta
        return x + gate * delta
