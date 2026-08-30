import pytorch_lightning as pl
import gc
import random
import torch
import torchaudio
import typing as tp

from ema_pytorch import EMA
from einops import rearrange
from safetensors.torch import save_file
from torch.nn import functional as F
from pytorch_lightning.utilities.rank_zero import rank_zero_only

from inference.sampling import get_alphas_sigmas, sample, sample_discrete_euler, sample_flow_pingpong, truncated_logistic_normal_rescaled, DistributionShift, sample_timesteps_logsnr
from models.diffusion import ConditionedDiffusionModelWrapper
from models.losses import MSELoss, MultiLoss
from training.utils import create_optimizer_from_config, create_scheduler_from_config

def log_metric(label, value, step): pass
def log_audio(label, audio, step, sr): pass
def log_image(label, image, step): pass

from time import time

class Profiler:

    def __init__(self):
        self.ticks = [[time(), None]]

    def tick(self, msg):
        self.ticks.append([time(), msg])

    def __repr__(self):
        rep = 80 * "=" + "\n"
        for i in range(1, len(self.ticks)):
            msg = self.ticks[i][1]
            ellapsed = self.ticks[i][0] - self.ticks[i - 1][0]
            rep += msg + f": {ellapsed*1000:.2f}ms\n"
        rep += 80 * "=" + "\n\n\n"
        return rep

class DiffusionCondTrainingWrapper(pl.LightningModule):
    def __init__(
            self,
            model: ConditionedDiffusionModelWrapper,
            lr: float = None,
            mask_padding: bool = True,
            mask_padding_dropout: float = 0.0,
            use_ema: bool = True,
            log_loss_info: bool = False,
            optimizer_configs: dict = None,
            pre_encoded: bool = False,
            cfg_dropout_prob = 0.1,
            timestep_sampler: tp.Literal["uniform", "logit_normal", "trunc_logit_normal", "log_snr"] = "uniform",
            timestep_sampler_options: tp.Optional[tp.Dict[str, tp.Any]] = None,
            validation_timesteps = [0.1, 0.3, 0.5, 0.7, 0.9],
            p_one_shot: float = 0.0,
            inpainting_config: dict = None,
            loss_configs: dict = None
    ):
        super().__init__()

        self.diffusion = model

        if use_ema:
            self.diffusion_ema = EMA(
                self.diffusion.model,
                beta=0.9999,
                power=3/4,
                update_every=1,
                update_after_step=1,
                include_online_model=False
            )
        else:
            self.diffusion_ema = None

        self.mask_padding = mask_padding
        self.mask_padding_dropout = mask_padding_dropout

        self.loss_modules = []

        mse_config = {}
        if loss_configs is not None and "mse" in loss_configs:
            mse_config = loss_configs["mse"]

        self.loss_modules.append(
            MSELoss(
                "output",
                "targets",
                weight=mse_config.get("weight", 1.0),
                mask_key="padding_mask" if self.mask_padding else None,
                name=mse_config.get("name", "mse_loss")
            )
        )

        self.losses = MultiLoss(self.loss_modules)

        self.cfg_dropout_prob = cfg_dropout_prob

        self.rng = torch.quasirandom.SobolEngine(1, scramble=True)

        self.timestep_sampler = timestep_sampler

        self.timestep_sampler_options = {} if timestep_sampler_options is None else timestep_sampler_options

        if self.timestep_sampler == "log_snr":
            self.mean_logsnr = self.timestep_sampler_options.get("mean_logsnr", -1.2)
            self.std_logsnr = self.timestep_sampler_options.get("std_logsnr", 2.0)

        self.p_one_shot = p_one_shot

        self.diffusion_objective = model.diffusion_objective

        self.log_loss_info = log_loss_info

        assert lr is not None or optimizer_configs is not None, "Must specify either lr or optimizer_configs in training config"

        if optimizer_configs is None:
            optimizer_configs = {
                "diffusion": {
                    "optimizer": {
                        "type": "Adam",
                        "config": {
                            "lr": lr
                        }
                    }
                }
            }
        else:
            if lr is not None:
                print(f"WARNING: learning_rate and optimizer_configs both specified in config. Ignoring learning_rate and using optimizer_configs.")

        self.optimizer_configs = optimizer_configs

        self.pre_encoded = pre_encoded

        self.inpainting_config = inpainting_config

        if self.inpainting_config is not None:
            self.inpaint_mask_kwargs = self.inpainting_config.get("mask_kwargs", {})

        self.validation_timesteps = validation_timesteps

        self.validation_step_outputs = {}

        for validation_timestep in self.validation_timesteps:
            self.validation_step_outputs[f'val/loss_{validation_timestep:.1f}'] = []

    def configure_optimizers(self):
        diffusion_opt_config = self.optimizer_configs['diffusion']
        opt_diff = create_optimizer_from_config(diffusion_opt_config['optimizer'], self.diffusion.parameters())

        if "scheduler" in diffusion_opt_config:
            sched_diff = create_scheduler_from_config(diffusion_opt_config['scheduler'], opt_diff)
            sched_diff_config = {
                "scheduler": sched_diff,
                "interval": "step"
            }
            return [opt_diff], [sched_diff_config]

        return [opt_diff]


    def training_step(self, batch, batch_idx):
        reals, metadata = batch

        p = Profiler()

        if reals.ndim == 4 and reals.shape[0] == 1:
            reals = reals[0]

        loss_info = {}
        diffusion_input = reals

        if not self.pre_encoded:
            loss_info["audio_reals"] = diffusion_input

        p.tick("setup")

        conditioning = self.diffusion.conditioner(metadata, self.device)

        first_padding_mask = metadata[0]["padding_mask"]
        if isinstance(first_padding_mask, list) and len(first_padding_mask) == 1:
            padding_masks = torch.stack([md["padding_mask"][0] for md in metadata], dim=0).to(self.device)
        else:
            padding_masks = torch.stack([md["padding_mask"] for md in metadata], dim=0).to(self.device)

        p.tick("conditioning")

        if self.diffusion.pretransform is not None:
            self.diffusion.pretransform.to(self.device)

            if not self.pre_encoded:
                with torch.cuda.amp.autocast() and torch.set_grad_enabled(self.diffusion.pretransform.enable_grad):
                    self.diffusion.pretransform.train(self.diffusion.pretransform.enable_grad)
                    diffusion_input = self.diffusion.pretransform.encode(diffusion_input)

                    p.tick("pretransform")

                    padding_masks = F.interpolate(
                        padding_masks.unsqueeze(1).float(),
                        size=diffusion_input.shape[2],
                        mode="nearest"
                    ).squeeze(1).bool()
            else:
                if hasattr(self.diffusion.pretransform, "scale") and self.diffusion.pretransform.scale != 1.0:
                    diffusion_input = diffusion_input / self.diffusion.pretransform.scale

        if self.timestep_sampler == "uniform":
            t = self.rng.draw(reals.shape[0])[:, 0].to(self.device)
        elif self.timestep_sampler == "logit_normal":
            t = torch.sigmoid(torch.randn(reals.shape[0], device=self.device))
        elif self.timestep_sampler == "trunc_logit_normal":
            t = 1 - truncated_logistic_normal_rescaled(reals.shape[0]).to(self.device)
        elif self.timestep_sampler == "log_snr":
            t = sample_timesteps_logsnr(reals.shape[0], mean_logsnr=self.mean_logsnr, std_logsnr=self.std_logsnr).to(self.device)
        else:
            raise ValueError(f"Invalid timestep_sampler: {self.timestep_sampler}")

        if self.diffusion.dist_shift is not None:
            t = self.diffusion.dist_shift.time_shift(t, reals.shape[2])

        if self.diffusion_objective == "v":
            alphas, sigmas = get_alphas_sigmas(t)
        elif self.diffusion_objective in ["rectified_flow", "rf_denoiser"]:
            alphas, sigmas = 1-t, t

        alphas = alphas[:, None, None]
        sigmas = sigmas[:, None, None]
        noise = torch.randn_like(diffusion_input)

        noised_inputs = diffusion_input * alphas + noise * sigmas

        if self.diffusion_objective == "v":
            targets = noise * alphas - diffusion_input * sigmas
        elif self.diffusion_objective in ["rectified_flow", "rf_denoiser"]:
            targets = noise - diffusion_input

        p.tick("noise")

        use_padding_mask = self.mask_padding and random.random() > self.mask_padding_dropout
        extra_args = {}
        if use_padding_mask:
            extra_args["mask"] = padding_masks

        output = self.diffusion(
            noised_inputs, t,
            cond=conditioning,
            cfg_dropout_prob=self.cfg_dropout_prob,
            **extra_args
        )
        p.tick("diffusion")

        if self.diffusion_objective == "v":
            x0_pred = alphas * noised_inputs - sigmas * output
        elif self.diffusion_objective in ["rectified_flow", "rf_denoiser"]:
            x0_pred = noised_inputs - sigmas * output
        else:
            raise ValueError(f"Unsupported diffusion_objective: {self.diffusion_objective}")

        loss_info.update({
            "output": output,
            "targets": targets,
            "padding_mask": padding_masks if use_padding_mask else None,
            "x0_pred": x0_pred,
        })

        loss, losses = self.losses(loss_info)

        p.tick("loss")

        log_dict = {
            'train/loss': loss.detach(),
            'train/std_data': diffusion_input.std(),
            'train/lr': self.trainer.optimizers[0].param_groups[0]['lr']
        }

        for loss_name, loss_value in losses.items():
            log_dict[f"train/{loss_name}"] = loss_value.detach()

        self.log_dict(log_dict, prog_bar=True, on_step=True)
        p.tick("log")

        return loss

    def on_before_zero_grad(self, *args, **kwargs):
        if self.diffusion_ema is not None:
            self.diffusion_ema.update()

    def validation_step(self, batch, batch_idx):

        reals, metadata = batch

        if reals.ndim == 4 and reals.shape[0] == 1:
            reals = reals[0]

        loss_info = {}

        diffusion_input = reals

        with torch.cuda.amp.autocast() and torch.no_grad():
            conditioning = self.diffusion.conditioner(metadata, self.device)

        if self.diffusion.pretransform is not None:
            self.diffusion.pretransform.to(self.device)

            if not self.pre_encoded:
                with torch.cuda.amp.autocast() and torch.no_grad():
                    self.diffusion.pretransform.train(self.diffusion.pretransform.enable_grad)

                    diffusion_input = self.diffusion.pretransform.encode(diffusion_input)
            else:
                if hasattr(self.diffusion.pretransform, "scale") and self.diffusion.pretransform.scale != 1.0:
                    diffusion_input = diffusion_input / self.diffusion.pretransform.scale

        for validation_timestep in self.validation_timesteps:

            t = torch.full((reals.shape[0],), validation_timestep, device=self.device)

            if self.diffusion_objective in ["v"]:
                alphas, sigmas = get_alphas_sigmas(t)
            elif self.diffusion_objective in ["rectified_flow", "rf_denoiser"]:
                alphas, sigmas = 1-t, t

            alphas = alphas[:, None, None]
            sigmas = sigmas[:, None, None]
            noise = torch.randn_like(diffusion_input)
            noised_inputs = diffusion_input * alphas + noise * sigmas

            if self.diffusion_objective == "v":
                targets = noise * alphas - diffusion_input * sigmas
            elif self.diffusion_objective in ["rectified_flow", "rf_denoiser"]:
                targets = noise - diffusion_input

            extra_args = {}

            with torch.cuda.amp.autocast() and torch.no_grad():
                output = self.diffusion(noised_inputs, t, cond=conditioning, cfg_dropout_prob = 0, **extra_args)

                val_loss = F.mse_loss(output, targets)

                self.validation_step_outputs[f'val/loss_{validation_timestep:.1f}'].append(val_loss.item())

    def on_validation_epoch_end(self):
        log_dict = {}
        for validation_timestep in self.validation_timesteps:
            outputs_key = f'val/loss_{validation_timestep:.1f}'
            val_loss = sum(self.validation_step_outputs[outputs_key]) / len(self.validation_step_outputs[outputs_key])

            val_loss = self.all_gather(val_loss).mean().item()

            log_metric(self.logger, outputs_key, val_loss, step=self.global_step)

        val_loss = torch.tensor([val for val in self.validation_step_outputs.values()]).mean()

        val_loss = self.all_gather(val_loss).mean().item()

        log_metric(self.logger, 'val/avg_loss', val_loss, step=self.global_step)

        for validation_timestep in self.validation_timesteps:
            self.validation_step_outputs[f'val/loss_{validation_timestep:.1f}'] = []


    def export_model(self, path, use_safetensors=False):
        if self.diffusion_ema is not None:
            self.diffusion.model = self.diffusion_ema.ema_model

        if use_safetensors:
            save_file(self.diffusion.state_dict(), path)
        else:
            torch.save({"state_dict": self.diffusion.state_dict()}, path)

class DiffusionCondDemoCallback(pl.Callback):
    def __init__(self,
                 demo_every=2000,
                 num_demos=8,
                 sample_size=65536,
                 demo_steps=250,
                 sample_rate=48000,
                 demo_conditioning: tp.Optional[tp.Dict[str, tp.Any]] = {},
                 demo_cfg_scales: tp.Optional[tp.List[int]] = [3, 5, 7],
                 demo_cond_from_batch: bool = False
    ):
        super().__init__()

        self.demo_every = demo_every
        self.num_demos = num_demos
        self.demo_samples = sample_size
        self.demo_steps = demo_steps
        self.sample_rate = sample_rate
        self.last_demo_step = -1
        self.demo_conditioning = demo_conditioning
        self.demo_cfg_scales = demo_cfg_scales

        self.demo_cond_from_batch = demo_cond_from_batch

    @rank_zero_only
    @torch.no_grad()
    def on_train_batch_end(self, trainer, module: DiffusionCondTrainingWrapper, outputs, batch, batch_idx):
        if (trainer.global_step - 1) % self.demo_every != 0 or self.last_demo_step == trainer.global_step:
            return

        module.eval()

        print(f"Generating demo")
        self.last_demo_step = trainer.global_step

        demo_samples = self.demo_samples

        demo_cond = self.demo_conditioning

        if self.demo_cond_from_batch:
            demo_cond = batch[1][:self.num_demos]

        if module.diffusion.pretransform is not None:
            demo_samples = demo_samples // module.diffusion.pretransform.downsampling_ratio

        noise = torch.randn([self.num_demos, module.diffusion.io_channels, demo_samples]).to(module.device)

        try:
            print("Getting conditioning")
            with torch.cuda.amp.autocast():
                conditioning = module.diffusion.conditioner(demo_cond, module.device)

            cond_inputs = module.diffusion.get_conditioning_inputs(conditioning)

            for cfg_scale in self.demo_cfg_scales:

                print(f"Generating demo for cfg scale {cfg_scale}")

                with torch.cuda.amp.autocast():
                    model = module.diffusion_ema.ema_model if module.diffusion_ema is not None else module.diffusion.model

                    if module.diffusion_objective == "v":
                        fakes = sample(model, noise, self.demo_steps, 0, **cond_inputs, cfg_scale=cfg_scale, dist_shift=module.diffusion.dist_shift, batch_cfg=True)
                    elif module.diffusion_objective == "rectified_flow":
                        fakes = sample_discrete_euler(model, noise, self.demo_steps, **cond_inputs, cfg_scale=cfg_scale, dist_shift=module.diffusion.dist_shift, batch_cfg=True)
                    elif module.diffusion_objective == "rf_denoiser":
                        logsnr = torch.linspace(-6, 2, self.demo_steps+1).to(module.device)
                        sigmas = torch.sigmoid(-logsnr)

                        sigmas[0] = 1.0
                        sigmas[-1] = 0.0

                        fakes = sample_flow_pingpong(model, noise, sigmas=sigmas, **cond_inputs, cfg_scale=cfg_scale, dist_shift=module.diffusion.dist_shift, batch_cfg=True)

                    if module.diffusion.pretransform is not None:
                        fakes = module.diffusion.pretransform.decode(fakes)

                fakes = rearrange(fakes, 'b d n -> d (b n)')

                filename = f'demo_cfg_{cfg_scale}_{trainer.global_step:08}.wav'
                fakes_out = fakes.to(torch.float32).div(torch.max(torch.abs(fakes))).mul(32767).to(torch.int16).cpu()
                torchaudio.save(filename, fakes_out, self.sample_rate)
                log_audio(trainer.logger, f'demo_cfg_{cfg_scale}', filename, self.sample_rate)

            del fakes

        except Exception as e:
            raise e
        finally:
            gc.collect()
            torch.cuda.empty_cache()
            module.train()
