import os
import torch
from safetensors.torch import load_file


def load_ckpt_state_dict(ckpt_path):
    if ckpt_path.endswith(".safetensors"):
        state_dict = load_file(ckpt_path)
    else:
        state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)["state_dict"]

    return state_dict

try:
    torch._dynamo.config.cache_size_limit = max(64, torch._dynamo.config.cache_size_limit)
    torch._dynamo.config.suppress_errors = True
except Exception as e:
    pass

enable_torch_compile = os.environ.get("ENABLE_TORCH_COMPILE", "0") == "1"

def compile(function, *args, **kwargs):

    if enable_torch_compile:
        try:
            return torch.compile(function, *args, **kwargs)
        except RuntimeError:
            return function

    return function
