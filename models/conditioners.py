
import torch
import typing as tp

from torch import nn
import torch.nn.functional as F

class Conditioner(nn.Module):
    def __init__(
            self,
            dim: int,
            output_dim: int,
            project_out: bool = False
            ):

        super().__init__()

        self.dim = dim
        self.output_dim = output_dim
        self.proj_out = nn.Linear(dim, output_dim) if (dim != output_dim or project_out) else nn.Identity()

    def forward(self, x: tp.Any) -> tp.Any:
        raise NotImplementedError()

class ECGLabelConditioner(Conditioner):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 256):
        super().__init__(output_dim, output_dim)
        self.label_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, labels: tp.List[torch.Tensor], device=None) -> tp.Any:
        if isinstance(labels, list):
            labels = torch.stack(labels, dim=0)

        labels = labels.to(device).float()
        label_embeds = self.label_proj(labels).unsqueeze(1)

        return [label_embeds, torch.ones(label_embeds.shape[0], 1).to(device)]

class ECGPatientInfoConditioner(Conditioner):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__(output_dim, output_dim)
        self.info_proj = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Linear(output_dim, output_dim)
        )

    def forward(self, patient_info: tp.List[torch.Tensor], device=None) -> tp.Any:
        if isinstance(patient_info, list):
            patient_info = torch.stack(patient_info, dim=0)

        patient_info = patient_info.to(device).float()
        info_embeds = self.info_proj(patient_info).unsqueeze(1)

        return [info_embeds, torch.ones(info_embeds.shape[0], 1).to(device)]

class ECGMaskConditioner(nn.Module):
    def __init__(self,
                 input_dim: int = 4,
                 output_dim: int = 512,
                 sample_size: int = 512,
                 audio_channels: int = 1,
                 downsampling_ratio: int = 1
                 ):
        super().__init__()
        self.input_dim = input_dim
        self.sample_size = sample_size
        self.audio_channels = audio_channels
        self.downsampling_ratio = downsampling_ratio

        self.effective_leads = self.audio_channels

        first_layer_in_channels = self.effective_leads * input_dim

        if self.downsampling_ratio == 1:
            strides = [1, 1, 1, 1]
        elif self.downsampling_ratio == 16:
            strides = [2, 2, 2, 2]
        else:
            raise ValueError(f"Unsupported downsampling_ratio: {self.downsampling_ratio}. Use 1 for Raw or 16 for AE.")

        def conv_block(in_f, out_f, stride):
            return nn.Sequential(
                nn.Conv1d(in_f, out_f, kernel_size=7, padding=3, stride=stride),
                nn.GroupNorm(8, out_f),
                nn.GELU()
            )

        self.model = nn.Sequential(
            conv_block(first_layer_in_channels, 128, stride=strides[0]),
            conv_block(128, 256, stride=strides[1]),
            conv_block(256, 512, stride=strides[2]),
            nn.Conv1d(512, output_dim, kernel_size=3, padding=1, stride=strides[3])
        )

        nn.init.kaiming_normal_(self.model[-1].weight, a=0.2)
        nn.init.zeros_(self.model[-1].bias)

    def forward(self, masks: torch.Tensor, device=None) -> list:
        if isinstance(masks, list):
            masks = torch.stack(masks, dim=0)

        masks = masks.to(device).long()

        curr_len = masks.shape[-1]
        if curr_len < self.sample_size:
            masks = F.pad(masks, (0, self.sample_size - curr_len), value=0)
        else:
            masks = masks[:, :, :self.sample_size]

        masks_onehot = F.one_hot(masks, num_classes=self.input_dim).float()
        b, l, s, c = masks_onehot.shape
        embedded = masks_onehot.permute(0, 1, 3, 2).reshape(b, l * c, s)

        mask_features = self.model(embedded)

        return [mask_features, torch.ones(b, mask_features.shape[2], device=device)]


class MultiConditioner(nn.Module):
    def __init__(self, conditioners: tp.Dict[str, Conditioner], default_keys: tp.Dict[str, str] = {}, pre_encoded_keys: tp.List[str] = []):
        super().__init__()

        self.conditioners = nn.ModuleDict(conditioners)
        self.default_keys = default_keys
        self.pre_encoded_keys = pre_encoded_keys

    def forward(self, batch_metadata: tp.List[tp.Dict[str, tp.Any]], device: tp.Union[torch.device, str]) -> tp.Dict[str, tp.Any]:
        output = {}

        for key, conditioner in self.conditioners.items():
            condition_key = key

            conditioner_inputs = []

            for x in batch_metadata:

                if condition_key not in x:
                    if condition_key in self.default_keys:
                        condition_key = self.default_keys[condition_key]
                    else:
                        raise ValueError(f"Conditioner key {condition_key} not found in batch metadata")

                if isinstance(x[condition_key], list) or isinstance(x[condition_key], tuple) and len(x[condition_key]) == 1:
                    conditioner_input = x[condition_key][0]

                else:
                    conditioner_input = x[condition_key]

                conditioner_inputs.append(conditioner_input)

            if key in self.pre_encoded_keys:
                output[key] = [torch.stack(conditioner_inputs, dim=0).to(device), None]
            else:
                output[key] = conditioner(conditioner_inputs, device)

        return output

def create_multi_conditioner_from_conditioning_config(config: tp.Dict[str, tp.Any], pretransform=None) -> MultiConditioner:
    conditioners = {}
    cond_dim = config["cond_dim"]

    default_keys = config.get("default_keys", {})

    pre_encoded_keys = config.get("pre_encoded_keys", [])

    for conditioner_info in config["configs"]:
        id = conditioner_info["id"]

        conditioner_type = conditioner_info["type"]

        conditioner_config = {"output_dim": cond_dim}

        conditioner_config.update(conditioner_info["config"])

        if conditioner_type == "ecg_labels":
            conditioners[id] = ECGLabelConditioner(**conditioner_config)
        elif conditioner_type == "ecg_patient_info":
            conditioners[id] = ECGPatientInfoConditioner(**conditioner_config)
        elif conditioner_type == "ecg_mask":
            conditioners[id] = ECGMaskConditioner(**conditioner_config)
        else:
            raise ValueError(f"Unknown conditioner type: {conditioner_type}")

    return MultiConditioner(conditioners, default_keys=default_keys, pre_encoded_keys=pre_encoded_keys)
