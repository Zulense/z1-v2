import torch 
from torch import nn 
from .vae import CausalVideoVae

import sys 
from pathlib import Path
# Add the parent directory (z1-v2)
sys.path.append(str(Path(__file__).resolve().parent.parent))
from context_parallel import (
        is_context_parallel_initialized, 
        get_context_parallel_world_size, 
        get_context_parallel_group_rank, 
        get_context_parallel_group,
        conv_scatter_to_context_parallel_region
        )




class VAELossWrapper(nn.Module):

    def __init__(self,
                 model_dtype='fp32',
                 disc_start=0,
                 logvar_init=0.0,
                 kl_weight=1.0,
                 pixelloss_weight=1.0,
                 perceptual_weight=1.0,
                 disc_weight=0.5,
                 interpolate=True,  # i don't understand why does have initialized.
                 add_discriminator=True,
                 freeze_encoder=False,
                 load_loss_module=False,
                 lpips_ckpt=None,
                 **kwargs
                 ):
        super().__init__()

        if model_dtype == 'bf16':
            torch_dtype = torch.bfloat16
        elif model_dtype == 'fp16':
            torch_dtype = torch.float16

        else:
            torch_dtype = torch.float32


        self.vae = CausalVideoVae()
        self.vae_scale_factor = self.vae.config.scaling_factor


    def forward(self, x, step, identifier=['video']):


        xdim = x.ndim
        if 'video' in identifier:
            print("video are found.")

        if is_context_parallel_initialized():

            assert self.training, "Only supports during training now."
            cp_world_size = get_context_parallel_world_size()
            global_src_rank = get_context_parallel_group_rank() * cp_world_size

            # sync the input and split 
            torch.distributed.broadcast(x, 
                                        src=global_src_rank,
                                         group=get_context_parallel_group())
            batch_x = conv_scatter_to_context_parallel_region(x, dim=2, kernel_size=1)


        posterior, reconstruct = self.vae(batch_x,
                                          is_init_image=True,
                                          temporal_chunk=False)

        print(posterior, reconstruct)


        
            




