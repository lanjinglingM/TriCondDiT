import torch


class InverseLR(torch.optim.lr_scheduler._LRScheduler):

    def __init__(self, optimizer, inv_gamma=1., power=1., warmup=0., final_lr=0.,
                 last_epoch=-1):
        self.inv_gamma = inv_gamma
        self.power = power
        if not 0. <= warmup < 1:
            raise ValueError('Invalid value for warmup')
        self.warmup = warmup
        self.final_lr = final_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            import warnings
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.")

        return self._get_closed_form_lr()

    def _get_closed_form_lr(self):
        warmup = 1 - self.warmup ** (self.last_epoch + 1)
        lr_mult = (1 + self.last_epoch / self.inv_gamma) ** -self.power
        return [warmup * max(self.final_lr, base_lr * lr_mult)
                for base_lr in self.base_lrs]

def create_optimizer_from_config(optimizer_config, parameters):

    optimizer_type = optimizer_config["type"]

    if optimizer_type == "FusedAdam":
        from deepspeed.ops.adam import FusedAdam
        optimizer = FusedAdam(parameters, **optimizer_config["config"])
    else:
        optimizer_fn = getattr(torch.optim, optimizer_type)
        optimizer = optimizer_fn(parameters, **optimizer_config["config"])
    return optimizer

def create_scheduler_from_config(scheduler_config, optimizer):
    if scheduler_config["type"] == "InverseLR":
        scheduler_fn = InverseLR
    else:
        scheduler_fn = getattr(torch.optim.lr_scheduler, scheduler_config["type"])
    scheduler = scheduler_fn(optimizer, **scheduler_config["config"])
    return scheduler
