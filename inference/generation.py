import numpy as np
import torch
import typing as tp

from .utils import prepare_audio
from .sampling import sample_k, sample_rf


def generate_diffusion_cond(
        model,
        steps: int = 250,
        cfg_scale=6,
        conditioning: dict = None,
        conditioning_tensors: tp.Optional[dict] = None,
        negative_conditioning: dict = None,
        negative_conditioning_tensors: tp.Optional[dict] = None,
        batch_size: int = 1,
        sample_size: int = 2097152,
        sample_rate: int = 48000,
        seed: int = -1,
        device: str = "cuda",
        init_audio: tp.Optional[tp.Tuple[int, torch.Tensor]] = None,
        init_noise_level: float = 1.0,
        return_latents = False,
        noise=None,
        **sampler_kwargs
        ) -> torch.Tensor:

    audio_sample_size = sample_size

    if model.pretransform is not None:
        sample_size = sample_size // model.pretransform.downsampling_ratio

    seed = seed if seed != -1 else np.random.randint(0, 2**32 - 1)
    print(seed)
    torch.manual_seed(seed)
    if noise is None:
        noise = torch.randn([batch_size, model.io_channels, sample_size], device=device)

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cudnn.benchmark = False

    assert conditioning is not None or conditioning_tensors is not None, "Must provide either conditioning or conditioning_tensors"
    if conditioning_tensors is None:
        conditioning_tensors = model.conditioner(conditioning, device)
    conditioning_inputs = model.get_conditioning_inputs(conditioning_tensors)

    if negative_conditioning is not None or negative_conditioning_tensors is not None:

        if negative_conditioning_tensors is None:
            negative_conditioning_tensors = model.conditioner(negative_conditioning, device)

        negative_conditioning_tensors = model.get_conditioning_inputs(negative_conditioning_tensors, negative=True)
    else:
        negative_conditioning_tensors = {}

    if init_audio is not None:
        in_sr, init_audio = init_audio

        io_channels = model.io_channels

        if model.pretransform is not None:
            io_channels = model.pretransform.io_channels

        init_audio = prepare_audio(init_audio, in_sr=in_sr, target_sr=model.sample_rate, target_length=audio_sample_size, target_channels=io_channels, device=device)

        if model.pretransform is not None:
            init_audio = model.pretransform.encode(init_audio)

        init_audio = init_audio.repeat(batch_size, 1, 1)

        sampler_kwargs["sigma_max"] = init_noise_level

    model_dtype = next(model.model.parameters()).dtype
    noise = noise.type(model_dtype)
    conditioning_inputs = {k: v.type(model_dtype) if v is not None else v for k, v in conditioning_inputs.items()}

    diff_objective = model.diffusion_objective

    if diff_objective == "v":
        sampled = sample_k(model.model, noise, init_audio, steps, **sampler_kwargs, **conditioning_inputs, **negative_conditioning_tensors, cfg_scale=cfg_scale, batch_cfg=True, rescale_cfg=True, device=device)
    elif diff_objective in ["rectified_flow", "rf_denoiser"]:

        if "sigma_min" in sampler_kwargs:
            del sampler_kwargs["sigma_min"]

        if "rho" in sampler_kwargs:
            del sampler_kwargs["rho"]

        sampled = sample_rf(model.model, noise, init_data=init_audio, steps=steps, **sampler_kwargs, **conditioning_inputs, **negative_conditioning_tensors, dist_shift=model.dist_shift, cfg_scale=cfg_scale, batch_cfg=True, rescale_cfg=True, device=device)

    del noise
    del conditioning_tensors
    del conditioning_inputs
    torch.cuda.empty_cache()
    if model.pretransform is not None and not return_latents:
        sampled = sampled.to(next(model.pretransform.parameters()).dtype)
        sampled = model.pretransform.decode(sampled)

    return sampled
