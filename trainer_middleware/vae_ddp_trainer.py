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
  

    if model_dtype == 'bf16':
        _dtype = torch.bfloat16
    else:
        _dtype = torch.float16

    print(f"Start training epoch {epoch}, {iters_per_epoch} iters per inner epoch.")

    for step in range(iters_per_epoch):
        if step >= iters_per_epoch:
            break

       
        

        samples = next(data_loader)

        samples['video'] = samples['video'].to(device, non_blocking=True)

        with torch.amp.autocast(device_type="cuda",
                                dtype=_dtype,
                                enabled=True):
            
            rec_loss, rec_log = model(x=samples['video'],
                                      step=args.global_step,
                                      identifier=samples['identifier'])
            print(f"<---------------> rec_loss={rec_loss}, rec_log={rec_log} <------------------->")
        ###################################################################################

        # The update of rec_loss 
        if rec_loss is not None:
            loss_value = rec_loss.item()

            if not math.isfinite(loss_value):
                print(f"Loss is {loss_value}, stopping training", force=True)
                sys.exit(1)

            optimizer.zero_grad()
            is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
            grad_norm = loss_scaler(rec_loss,
                                    optimizer,
                                    clip_grad=clip_grad,
                                    parameters=model.modules.vae.parameters(),
                                    create_graph=is_second_order)

            if "scale" in loss_scaler.state_dict():
                loss_scaler_value = loss_scaler.state_dict()["scale"]
            else:
                loss_scaler_value = 1 

            
            

        


    
        args.global_step = args.global_step + 1


