import torch
from torch import nn

class IdentityPretransform(nn.Module):
    def __init__(self):
        super().__init__()
        self.downsampling_ratio = 1
        self.fake_param = nn.Parameter(torch.zeros(1), requires_grad=False)

    def encode(self, x, **kwargs):
        return x

    def decode(self, z, **kwargs):
        return z


class Pretransform(nn.Module):
    def __init__(self, enable_grad, io_channels, is_discrete):
        super().__init__()

        self.is_discrete = is_discrete
        self.io_channels = io_channels
        self.encoded_channels = None
        self.downsampling_ratio = None

        self.enable_grad = enable_grad

    def encode(self, x):
        raise NotImplementedError

    def decode(self, z):
        raise NotImplementedError

    def tokenize(self, x):
        raise NotImplementedError

    def decode_tokens(self, tokens):
        raise NotImplementedError
