import torch 
from torch import nn 

from vae import CausalVideoVae



class VAELossWrapper(nn.Module):

    def __init__(self,
                 model_path,
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


    def forward(self, x):

        xdim = x.ndim 

        if 'video' in identifier:
            




