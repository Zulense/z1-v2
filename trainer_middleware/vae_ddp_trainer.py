import torch 
from torch import nn 
from typing import Iterable
import math, sys


from .utils import MetricLogger, SmoothedValue



def train_one_epoch(
        model: nn.Module,
        model_dtype: str,
        data_loader: Iterable,
        optimizer: torch.optim.Optimizer,
        optimizer_disc: torch.optim.Optimizer,
        device: torch.device,
        epoch:int,
        loss_scaler,
        loss_scaler_disc,
        clip_grad: float = 0,
        log_writer=None,
        lr_scheduler=None,
        start_steps=None,
        lr_schedule_values=None,
        lr_schedule_values_disc=None,
        args=None,
        print_freq=20,
        iters_per_epoch=2000
):

    # The trainer for causal video vae 
    model.train()
    metric_logger = MetricLogger(delimiter=" ")

    if optimizer is not None:
        metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt='{value:.6f}'))
        metric_logger.add_meter("min_lr", SmoothedValue(window_size=1, fmt='{value:.6f}'))

    if optimizer_disc is not None:
        metric_logger.add_meter('disc_lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
        metric_logger.add_meter('disc_min_lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))

    header = f'Epoch: [{epoch}]'

    if model_dtype == 'bf16':
        _dtype = torch.bfloat16
    else:
        _dtype = torch.float16

    print(f"Start training epoch {epoch}, {iters_per_epoch} iters per inner epoch.")

    for step in metric_logger.log_every(range(iters_per_epoch), print_freq, header):
        if step >= iters_per_epoch:
            break

        it = start_steps + step  # global training iteration.
        if lr_schedule_values is not None:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[it] * param_group.get("lr_scale", 1.0)

        if optimizer_disc is not None:
            for i, param_group in enumerate(optimizer_disc.param_groups):
                if lr_schedule_values_disc is not None:
                    param_group["lr"] = lr_schedule_values_disc[it] * param_group.get("lr_scale", 1.0)

        samples = next(data_loader)

        samples['video'] = samples['video'].to(device, non_blocking=True)

        with torch.amp.autocast(device_type="cuda",
                                dtype=_dtype,
                                enabled=True):
            
            rec_loss, gan_loss, log_loss = model(samples['video'],
                                                 args.global_step,
                                                 identifier=samples['identifier'])

            print(f"<--------------------> rec_loss: {rec_loss}, Gan_loss: {gan_loss}, Log_loss: {log_loss} <-------------------->")

        ###################################################################################
        

    
        args.global_step = args.global_step + 1


