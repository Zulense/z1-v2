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

from .loss import LPIPSWithDiscriminator




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
                 load_loss_module=True,
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

        # Enable Memory saving 
        self.vae.encoder.gradient_checkpointing = True
        self.vae.decoder.gradient_checkpointing = True

        if load_loss_module:
            self.loss = LPIPSWithDiscriminator(disc_start=disc_start,
                                               logvar_init=logvar_init,
                                               kl_weight=kl_weight,
                                               pixelloss_weight=pixelloss_weight,
                                               perceptual_weight=perceptual_weight,
                                               disc_weight=disc_weight,
                                               add_discriminator=add_discriminator,
                                               using_3d_discriminator=False,
                                               disc_num_layers=4,
                                               lpips_ckpt=lpips_ckpt)
            

        


    def forward(self, x, step, identifier=['video']):

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
        else:
            batch_x = x 


        posterior, reconstruct = self.vae(batch_x,
                                          is_init_image=True,
                                          temporal_chunk=False)

        print(f"<--------------> [vae_wrapper.py] Let's know the posterior: {posterior} and reconstruct: {reconstruct} <------------------>")

        # The reconstruct loss 
        reconstruct_loss, rec_log = self.loss(
            batch_x, reconstruct, posterior, optimizer_idx=0, global_step=step, last_layer=self.vae.get_last_layer()
        )

        print(f"<--------------> [vae_wrapper.py] Let's know the reconstruct_loss: {reconstruct_loss} and rec_log: {rec_log} <------------------>")




        return reconstruct_loss, rec_log
    

        
