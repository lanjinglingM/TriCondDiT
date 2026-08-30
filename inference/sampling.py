import torch
import math
from tqdm import trange, tqdm
import torch.distributions as dist

import k_diffusion as K

def get_alphas_sigmas(t):

    return torch.cos(t * math.pi / 2), torch.sin(t * math.pi / 2)

def alpha_sigma_to_t(alpha, sigma):

    return torch.atan2(sigma, alpha) / math.pi * 2

def t_to_alpha_sigma(t):

    return torch.cos(t * math.pi / 2), torch.sin(t * math.pi / 2)

class DistributionShift:
    def __init__(self, base_shift=0.5, max_shift=1.15, max_length=4096, min_length=256, use_sine=False):
        self.base_shift = base_shift
        self.max_shift = max_shift
        self.max_length = max_length
        self.min_length = min_length
        self.use_sine = use_sine

    def time_shift(self, t: torch.Tensor, seq_len: int):
        sigma = 1.0
        mu = - (self.base_shift + (self.max_shift - self.base_shift) * (seq_len - self.min_length) / (self.max_length - self.min_length))
        t_out = 1 - math.exp(mu) / (math.exp(mu) + (1 / (1 - t) - 1) ** sigma)

        if self.use_sine:
            t_out = torch.sin(t_out * math.pi / 2)

        return t_out

def sample_timesteps_logsnr(batch_size, mean_logsnr=-1.2, std_logsnr=2.0):

    logsnr = torch.randn(batch_size) * std_logsnr + mean_logsnr

    t = torch.sigmoid(-logsnr)

    t = t.clamp(1e-4, 1 - 1e-4)

    return t

def truncated_logistic_normal_rescaled(shape, left_trunc=0.075, right_trunc=1):

    logits = torch.randn(shape)

    normal_dist = dist.Normal(0, 1)
    cdf_values = normal_dist.cdf(logits)

    lower_bound = normal_dist.cdf(torch.logit(torch.tensor(left_trunc)))
    upper_bound = normal_dist.cdf(torch.logit(torch.tensor(right_trunc)))

    truncated_cdf_values = lower_bound + (upper_bound - lower_bound) * cdf_values

    truncated_samples = torch.sigmoid(normal_dist.icdf(truncated_cdf_values))

    rescaled_samples = (truncated_samples - left_trunc) / (right_trunc - left_trunc)

    return rescaled_samples

@torch.no_grad()
def sample_discrete_euler(model, x, steps=None, sigma_max=1, sigmas=None, callback=None, dist_shift=None, disable_tqdm=False, **extra_args):

    assert steps is not None or sigmas is not None, "Either steps or sigmas must be provided"

    ts = x.new_ones([x.shape[0]])

    if sigmas is None:

        t = torch.linspace(sigma_max, 0, steps + 1)

        if dist_shift is not None:
            t = dist_shift.time_shift(t, x.shape[-1])

    else:
        t = sigmas

    for i, (t_curr, t_prev) in enumerate(tqdm(zip(t[:-1], t[1:]), disable=disable_tqdm)):
        t_curr_tensor = t_curr * torch.ones(
            (x.shape[0],), dtype=x.dtype, device=x.device
        )

        dt = t_prev - t_curr

        v = model(x, t_curr_tensor, **extra_args)
        x = x + dt * v

        if callback is not None:
            denoised = x - t_prev * v
            callback({'x': x, 't': t_curr, 'sigma': t_curr, 'i': i+1, 'denoised': denoised })

    return x

@torch.no_grad()
def sample_rk4(model, x, steps=None, sigma_max=1, sigmas=None, callback=None, dist_shift=None, **extra_args):

    assert steps is not None or sigmas is not None, "Either steps or sigmas must be provided"

    ts = x.new_ones([x.shape[0]])

    if sigmas is None:

        t = torch.linspace(sigma_max, 0, steps + 1)

        if dist_shift is not None:
            t = dist_shift.time_shift(t, x.shape[-1])

    else:
        t = sigmas

    for i, (t_curr, t_prev) in enumerate(tqdm(zip(t[:-1], t[1:]))):
        t_curr_tensor = t_curr * ts
        dt = t_prev - t_curr

        k1 = model(x, t_curr_tensor, **extra_args)
        k2 = model(x + dt / 2 * k1, (t_curr + dt / 2) * ts, **extra_args)
        k3 = model(x + dt / 2 * k2, (t_curr + dt / 2) * ts, **extra_args)
        k4 = model(x + dt * k3, t_prev * ts, **extra_args)

        x = x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

        if callback is not None:
            denoised = x - t_prev * k4
            callback({'x': x, 't': t_curr, 'sigma': t_curr, 'i': i+1, 'denoised': denoised })

    return x

@torch.no_grad()
def sample_flow_dpmpp(model, x, steps=None, sigma_max=1, sigmas=None, callback=None, dist_shift=None, disable_tqdm=False, **extra_args):

    assert steps is not None or sigmas is not None, "Either steps or sigmas must be provided"

    ts = x.new_ones([x.shape[0]])

    if sigmas is None:

        t = torch.linspace(sigma_max, 0, steps + 1)

        if dist_shift is not None:
            t = dist_shift.time_shift(t, x.shape[-1])

    else:
        t = sigmas

    old_denoised = None

    log_snr = lambda t: ((1-t) / t).log()

    for i in trange(len(t) - 1, disable=disable_tqdm):

        t_curr, t_next = t[i], t[i + 1]

        denoised = x - t_curr * model(x, t_curr * ts, **extra_args)
        if callback is not None:
            callback({'x': x, 'i': i, 't': t_curr, 'sigma': t_curr, 'denoised': denoised})
        alpha_t = 1-t_next
        h = log_snr(t_next) - log_snr(t_curr)
        if old_denoised is None or t_next == 0:
            x = (t_next / t_curr) * x - alpha_t * (-h).expm1() * denoised
        else:
            h_last = log_snr(t_curr) - log_snr(t[i - 1])
            r = h_last / h
            denoised_d = (1 + 1 / (2 * r)) * denoised - (1 / (2 * r)) * old_denoised
            x = (t_next / t_curr) * x - alpha_t * (-h).expm1() * denoised_d
        old_denoised = denoised
    return x

@torch.no_grad()
def sample_flow_pingpong(model, x, steps=None, sigma_max=1, sigmas=None, callback=None, dist_shift=None, **extra_args):

    assert steps is not None or sigmas is not None, "Either steps or sigmas must be provided"

    ts = x.new_ones([x.shape[0]])

    if sigmas is None:

        t = torch.linspace(sigma_max, 0, steps + 1)

        if dist_shift is not None:
            t = dist_shift.time_shift(t, x.shape[-1])

    else:
        t = sigmas

    for i in trange(len(t) - 1, disable=False):

        denoised = x - t[i] * model(x, t[i] * ts, **extra_args)
        if callback is not None:
            callback({'x': x, 'i': i, 't': t[i], 'sigma': t[i], 'sigma_hat': t[i], 'denoised': denoised})

        t_next = t[i + 1]
        x = (1-t_next) * denoised + t_next * torch.randn_like(x)

    return x

@torch.no_grad()
def sample(model, x, steps, eta, callback=None, sigma_max=1.0, dist_shift=None, cfg_pp=False, **extra_args):
    ts = x.new_ones([x.shape[0]])

    t = torch.linspace(sigma_max, 0, steps + 1)[:-1]

    if dist_shift is not None:
        t = dist_shift.time_shift(t, x.shape[-1])

    alphas, sigmas = get_alphas_sigmas(t)

    for i in trange(steps):

        if cfg_pp:
            v, info = model(x, ts * t[i], return_info=True, **extra_args)

            if "uncond_output" in info:
                v_eps = info["uncond_output"]
            else:
                v_eps = v
        else:
            v = model(x, ts * t[i], **extra_args)
            v_eps = v

        pred = x * alphas[i] - v * sigmas[i]
        eps = x * sigmas[i] + v_eps * alphas[i]

        if i < steps - 1:
            ddim_sigma = eta * (sigmas[i + 1]**2 / sigmas[i]**2).sqrt() * \
                (1 - alphas[i]**2 / alphas[i + 1]**2).sqrt()
            adjusted_sigma = (sigmas[i + 1]**2 - ddim_sigma**2).sqrt()

            x = pred * alphas[i + 1] + eps * adjusted_sigma

            if eta:
                x += torch.randn_like(x) * ddim_sigma

        if callback is not None:
            denoised = pred
            callback({'x': x, 't': t[i], 'sigma': sigmas[i], 'i': i, 'denoised': denoised })

    return pred

def get_bmask(i, steps, mask):
    strength = (i+1)/(steps)
    bmask = torch.where(mask<=strength,1,0)
    return bmask

def make_cond_model_fn(model, cond_fn):
    def cond_model_fn(x, sigma, **kwargs):
        with torch.enable_grad():
            x = x.detach().requires_grad_()
            denoised = model(x, sigma, **kwargs)
            cond_grad = cond_fn(x, sigma, denoised=denoised, **kwargs).detach()
            cond_denoised = denoised.detach() + cond_grad * K.utils.append_dims(sigma**2, x.ndim)
        return cond_denoised
    return cond_model_fn

def sample_k(
        model_fn,
        noise,
        init_data=None,
        steps=100,
        sampler_type="dpmpp-2m-sde",
        sigma_min=0.01,
        sigma_max=100,
        rho=1.0,
        device="cuda",
        callback=None,
        cond_fn=None,
        **extra_args
    ):

    is_k_diff = sampler_type in ["k-heun", "k-lms", "k-dpmpp-2s-ancestral", "k-dpm-2", "k-dpm-fast", "k-dpm-adaptive", "dpmpp-2m-sde", "dpmpp-3m-sde","dpmpp-2m"]
    is_v_diff = sampler_type in ["v-ddim", "v-ddim-cfgpp"]

    if is_k_diff:

        denoiser = K.external.VDenoiser(model_fn)

        if cond_fn is not None:
            denoiser = make_cond_model_fn(denoiser, cond_fn)

        sigmas = K.sampling.get_sigmas_polyexponential(steps, sigma_min, sigma_max, rho, device=device)
        noise = noise * sigmas[0]

        if init_data is not None:
            x = init_data + noise
        else:
            x = noise

        if sampler_type == "k-heun":
            return K.sampling.sample_heun(denoiser, x, sigmas, disable=False, callback=callback, extra_args=extra_args)
        elif sampler_type == "k-lms":
            return K.sampling.sample_lms(denoiser, x, sigmas, disable=False, callback=callback, extra_args=extra_args)
        elif sampler_type == "k-dpmpp-2s-ancestral":
            return K.sampling.sample_dpmpp_2s_ancestral(denoiser, x, sigmas, disable=False, callback=callback, extra_args=extra_args)
        elif sampler_type == "k-dpm-2":
            return K.sampling.sample_dpm_2(denoiser, x, sigmas, disable=False, callback=callback, extra_args=extra_args)
        elif sampler_type == "k-dpm-fast":
            return K.sampling.sample_dpm_fast(denoiser, x, sigma_min, sigma_max, steps, disable=False, callback=callback, extra_args=extra_args)
        elif sampler_type == "k-dpm-adaptive":
            return K.sampling.sample_dpm_adaptive(denoiser, x, sigma_min, sigma_max, rtol=0.01, atol=0.01, disable=False, callback=callback, extra_args=extra_args)
        elif sampler_type == "dpmpp-2m":
            return K.sampling.sample_dpmpp_2m(denoiser, x, sigmas, disable=False, callback=callback, extra_args=extra_args)
        elif sampler_type == "dpmpp-2m-sde":
            return K.sampling.sample_dpmpp_2m_sde(denoiser, x, sigmas, disable=False, callback=callback, extra_args=extra_args)
        elif sampler_type == "dpmpp-3m-sde":
            return K.sampling.sample_dpmpp_3m_sde(denoiser, x, sigmas, disable=False, callback=callback, extra_args=extra_args)
        else:
            raise ValueError(f"Unknown sampler_type: {sampler_type}")
    elif is_v_diff:

        if sigma_max > 1:
            sigma_max = 1

        if cond_fn is not None:
            model_fn = make_cond_model_fn(model_fn, cond_fn)

        alpha, sigma = t_to_alpha_sigma(torch.tensor(sigma_max))

        if init_data is not None:
            x = init_data * alpha + noise * sigma
        else:
            x = noise

        if sampler_type == "v-ddim" or sampler_type == "v-ddim-cfgpp":
            use_cfg_pp = sampler_type == "v-ddim-cfgpp"
            return sample(model_fn, x, steps, eta=0.0, sigma_max=sigma_max, cfg_pp=use_cfg_pp, callback=callback, **extra_args)
    else:
        raise ValueError(f"Unknown sampler type {sampler_type}")

def sample_rf(
        model_fn,
        noise,
        init_data=None,
        steps=100,
        sampler_type="euler",
        sigma_max=1,
        device="cuda",
        callback=None,
        cond_fn=None,
        **extra_args
    ):

    if sigma_max > 1:
        sigma_max = 1

    if cond_fn is not None:
        denoiser = make_cond_model_fn(denoiser, cond_fn)

    if init_data is not None:
        x = init_data * (1 - sigma_max) + noise * sigma_max

    else:
        x = noise

    logsnr_max = math.log(((1-sigma_max)/sigma_max) + 1e-6) if sigma_max < 1 else -6

    logsnr = torch.linspace(logsnr_max, 2, steps + 1)

    t = torch.sigmoid(-logsnr)

    t[0] = sigma_max
    t[-1] = 0

    if sampler_type == "euler":
        return sample_discrete_euler(model_fn, x, sigmas=t, sigma_max=sigma_max, callback=callback, **extra_args)
    elif sampler_type == "rk4":
        return sample_rk4(model_fn, x, steps, sigma_max, callback=callback, **extra_args)
    elif sampler_type == "dpmpp":
        return sample_flow_dpmpp(model_fn, x, sigmas=t, sigma_max=sigma_max, callback=callback, **extra_args)
    elif sampler_type == "pingpong":
        return sample_flow_pingpong(model_fn, x, sigmas=t, sigma_max=sigma_max, callback=callback, **extra_args)
    else:
        raise ValueError(f"Unknown sampler_type: {sampler_type}")
