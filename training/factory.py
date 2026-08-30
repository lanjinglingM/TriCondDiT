from models.factory import create_model_from_config

def create_training_wrapper_from_config(model_config, model):
    model_type = model_config.get('model_type', None)
    assert model_type is not None, 'model_type must be specified in model config'

    training_config = model_config.get('training', None)
    assert training_config is not None, 'training config must be specified in model config'

    if model_type in ['diffusion_cond', 'diffusion_cond_inpaint']:

        from .diffusion import DiffusionCondTrainingWrapper
        return DiffusionCondTrainingWrapper(
            model,
            lr=training_config.get("learning_rate", None),
            mask_padding=training_config.get("mask_padding", False),
            mask_padding_dropout=training_config.get("mask_padding_dropout", 0.0),
            use_ema = training_config.get("use_ema", True),
            log_loss_info=training_config.get("log_loss_info", False),
            optimizer_configs=training_config.get("optimizer_configs", None),
            pre_encoded=training_config.get("pre_encoded", False),
            cfg_dropout_prob = training_config.get("cfg_dropout_prob", 0.1),
            timestep_sampler = training_config.get("timestep_sampler", "uniform"),
            timestep_sampler_options = training_config.get("timestep_sampler_options", {}),
            p_one_shot=training_config.get("p_one_shot", 0.0),
            inpainting_config = training_config.get("inpainting", None),
            loss_configs = training_config.get("losses", None)
        )

    else:
        raise NotImplementedError(f'Unknown model type: {model_type}')

def create_demo_callback_from_config(model_config, **kwargs):
    model_type = model_config.get('model_type', None)
    assert model_type is not None, 'model_type must be specified in model config'

    training_config = model_config.get('training', None)
    assert training_config is not None, 'training config must be specified in model config'

    demo_config = training_config.get("demo", {})

    if model_type in ["diffusion_cond", "diffusion_cond_inpaint"]:
        from .diffusion import DiffusionCondDemoCallback

        return DiffusionCondDemoCallback(
            demo_every=demo_config.get("demo_every", 2000),
            sample_size=model_config["sample_size"],
            sample_rate=model_config["sample_rate"],
            demo_steps=demo_config.get("demo_steps", 250),
            num_demos=demo_config["num_demos"],
            demo_cfg_scales=demo_config["demo_cfg_scales"],
            demo_conditioning=demo_config.get("demo_cond", {}),
            demo_cond_from_batch=demo_config.get("demo_cond_from_batch", False)
        )

    else:
        raise NotImplementedError(f'Unknown model type: {model_type}')
