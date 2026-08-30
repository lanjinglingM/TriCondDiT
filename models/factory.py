import json

def create_model_from_config(model_config):
    model_type = model_config.get('model_type', None)

    assert model_type is not None, 'model_type must be specified in model config'

    if model_type == 'diffusion_cond' or model_type == 'diffusion_cond_inpaint':
        from .diffusion import create_diffusion_cond_from_config
        return create_diffusion_cond_from_config(model_config)
    else:
        raise NotImplementedError(f'Unknown model type: {model_type}')

def create_model_from_config_path(model_config_path):
    with open(model_config_path) as f:
        model_config = json.load(f)

    return create_model_from_config(model_config)

def create_pretransform_from_config(pretransform_config, sample_rate):
    pretransform_type = pretransform_config.get('type', None)

    assert pretransform_type is not None, 'type must be specified in pretransform config'

    if pretransform_type == "identity":
        from .pretransforms import IdentityPretransform
        pretransform = IdentityPretransform()
    else:
        raise NotImplementedError(f'Unknown pretransform type: {pretransform_type}')

    enable_grad = pretransform_config.get('enable_grad', False)
    if not hasattr(pretransform, 'enable_grad'):
        pretransform.enable_grad = enable_grad

    if any(p.requires_grad for p in pretransform.parameters()):
        pretransform.eval().requires_grad_(pretransform.enable_grad)
    else:
        pretransform.eval()

    return pretransform
