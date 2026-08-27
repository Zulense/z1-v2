import torch 
from torch import nn 
from typing import Tuple, Optional
from diffusers.utils import is_torch_version, BaseOutput
from torch.utils.checkpoint import checkpoint
from diffusers.utils.torch_utils import randn_tensor
import numpy as np 
from dataclasses import dataclass


from .causal_conv import CausalConv3d, CausalGroupNorm
from .block import DownEncoderBlockCausal3D, MidBlockCausal3D, UpDecoderBlockCausal3D


class CausalVaeEncoder(nn.Module):

    def __init__(self,
                 in_channels: int = 3,
                 out_channels: int = 3,
                 down_black_types: Tuple[str, ...] = ("DownEncoderBlockCausal3D",),
                 spatial_down_sample: Tuple[bool, ...] = (True,),
                 temporal_down_sample: Tuple[bool, ...] = (False,),
                 block_out_channels: Tuple[int, ...] = (64,),
                 layers_per_block: Tuple[int, ...] = (2,),
                 norm_num_groups: int = 32,
                 act_fn: str = "silu",
                 double_z: bool = True,
                 block_dropout: Tuple[int, ...] = (0.0,),
                 mid_block_add_attention = True,
                 ):

        super().__init__()
        self.layers_per_block = layers_per_block

        self.conv_in = CausalConv3d(in_channels,
                                    output_chaannels=block_out_channels[0],  # (64, 128, 512, 512)
                                    kernel_size=3,
                                    stride=1
                                    )

        # DOWN BLOCK
        self.mid_block = None 
        self.down_blocks = nn.ModuleList([])
        output_channel = block_out_channels[0]
        for i, down_black_type in enumerate(down_black_types):
            input_channel = output_channel
            output_channel = block_out_channels[i]

            down_block = DownEncoderBlockCausal3D(in_channels=input_channel,
                                                  out_channels=output_channel,
                                                  dropout=block_dropout[i],
                                                  num_layers=self.layers_per_block[i],
                                                  resnet_eps=1e-6,
                                                  resnet_time_scale_shift="default",
                                                  resnet_act_fn=act_fn,
                                                  resnet_groups=norm_num_groups,
                                                  add_spatial_downsample=spatial_down_sample[i],
                                                  add_temporal_downsample=temporal_down_sample[i],
                                                  )
            self.down_blocks.append(down_block)

        # MID BLOCK 
        self.mid_block = MidBlockCausal3D(in_channels=block_out_channels[-1],
                                          temb_channels=None,
                                          dropout=block_dropout[-1],
                                          num_layers=1,
                                          resnet_eps=1e-6,
                                          resnet_time_scale_shift="default",
                                          resnet_act_fn=act_fn,
                                          resnet_groups=norm_num_groups,
                                          add_attention=mid_block_add_attention,
                                          attention_head_dim=block_out_channels[-1],
                                          output_scale_factor=1)


        # OUT LAYER 
        self.conv_norm_out = CausalGroupNorm(num_groups=norm_num_groups,
                                             num_channels=block_out_channels[-1],
                                             eps=1e-6)
        self.conv_act = nn.SiLU()

        conv_out_channels = 2 * out_channels if double_z else out_channels
        self.conv_out = CausalConv3d(input_channels=block_out_channels[-1],
                                     output_chaannels=conv_out_channels,
                                     kernel_size=3,
                                     stride=1)
        self.gradient_checkpointing = False 


    def forward(self,
                sample: torch.FloatTensor,
                is_init_image=True,
                temporal_chunk=False) -> torch.FloatTensor:

        sample = self.conv_in(sample,
                              is_init_image=is_init_image,
                              temporal_chunk=temporal_chunk)

        if self.training and self.gradient_checkpointing:

            def create_custom_forward(module):
                def custom_forward(*inputs):
                    return module(*inputs)
                return custom_forward


            # Down Block 
            if is_torch_version(">=", "1.11.0"):
                for down_block in self.down_blocks:
                    sample = checkpoint(create_custom_forward(down_block),
                                        sample,
                                        is_init_image,
                                        temporal_chunk,
                                        use_reentrant=False)

                # middle 
                sample = checkpoint(create_custom_forward(self.mid_block),
                                    sample,
                                    is_init_image,
                                    temporal_chunk,
                                    use_reentrant=False)


        else:
            for down_block in self.down_blocks:
                sample = down_block(sample, is_init_image, temporal_chunk)

            sample = self.mid_block(sample, is_init_image, temporal_chunk)


        ## post-process 
        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample, 
                               is_init_image=is_init_image, 
                               temporal_chunk=temporal_chunk)

        return sample 



class CausalVaeDecoder(nn.Module):

    def __init__(self,
                 in_channels: int = 3,
                 out_channels: int = 3,
                 up_block_types: Tuple[str, ...] = ("UpDecoderBlockCausal3D",),
                 spatial_up_sample: Tuple[bool, ...] = (True,),
                 temporal_up_sample: Tuple[bool, ...] = (False,),
                 block_out_channels: Tuple[int, ...] = (64,),
                 layers_per_block: Tuple[int, ...] = (2,),
                 norm_num_groups: int = 32,
                 act_fn: str = "silu",
                 mid_block_add_attention=True,
                 block_dropout: Tuple[int, ...] = (0.0,)
                 ):

        super().__init__()
        self.layers_per_block = layers_per_block

        self.conv_in = CausalConv3d(input_channels=in_channels,
                                    output_chaannels=block_out_channels[-1],
                                    kernel_size=3,
                                    stride=1)

        self.mid_block = None
        self.up_blocks = nn.ModuleList([])

        ## MID-BLOCK
        self.mid_block = MidBlockCausal3D(in_channels=block_out_channels[-1],
                                          temb_channels=None,
                                          dropout=block_dropout[-1],
                                          num_layers=1,
                                          resnet_eps=1e-6,
                                          resnet_time_scale_shift="default",
                                          resnet_act_fn=act_fn,
                                          resnet_groups=norm_num_groups,
                                          resnet_pre_norm=True,
                                          add_attention=mid_block_add_attention,
                                          attention_head_dim=block_out_channels[-1],
                                          output_scale_factor=1)

        # UPPER-BLOCK
        reversed_block_out_channels = list(reversed(block_out_channels))
        output_channel = reversed_block_out_channels[0]

        for i, up_block_type in enumerate(up_block_types):
            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]

            
            up_block = UpDecoderBlockCausal3D(in_channels=prev_output_channel,
                                              out_channels=output_channel,
                                              resolution_idx=None,
                                              dropout=block_dropout[i],
                                              num_layers=self.layers_per_block[i],
                                              resnet_eps=1e-6,
                                              resnet_time_scale_shift="default",
                                              resnet_act_fn=act_fn,
                                              resnet_groups=norm_num_groups,
                                              resnet_pre_norm=True,
                                              output_scale_factor=1,
                                              add_spatial_upsample=spatial_up_sample[i],
                                              add_temporal_upsample=temporal_up_sample[i],
                                              temb_channels=None,
                                              )

            self.up_blocks.append(up_block)
            prev_output_channel = output_channel


        ## OUTPUT 
        self.conv_norm_out = CausalGroupNorm(num_groups=norm_num_groups,
                                             num_channels=block_out_channels[0],
                                             eps=1e-6)
        self.conv_act = nn.SiLU()
        self.conv_out = CausalConv3d(input_channels=block_out_channels[0],
                                     output_chaannels=out_channels,
                                     kernel_size=3,
                                     stride=1)
        self.gradient_checkpointing = False 


    def forward(self,
                sample: torch.FloatTensor,
                is_init_image=True,
                temporal_chunk=False,
                ) -> torch.FloatTensor:

        sample = self.conv_in(sample,
                              is_init_image,
                              temporal_chunk)


        upscale_dtype = next(iter(self.up_blocks.parameters())).dtype 
        if self.training and self.gradient_checkpointing:

            def create_custom_function(module):
                def custom_forward(*inputs):
                    return module(*inputs)
                return custom_forward


            if is_torch_version(">=", "1.11.0"):
                sample = checkpoint(create_custom_function(self.mid_block),
                                    sample,
                                    is_init_image=is_init_image,
                                    temporal_chunk=temporal_chunk,
                                    use_reentrant=False)
                sample = sample.to(upscale_dtype)

                for up_block in self.up_blocks:
                    sample = checkpoint(
                        create_custom_function(up_block),
                        sample,
                        is_init_image=is_init_image,
                        temporal_chunk=temporal_chunk,
                        use_reentrant=False
                    )

        else:
            sample = self.mid_block(sample, 
                                    is_init_image=is_init_image,
                                    temporal_chunk=temporal_chunk)
            sample = sample.to(upscale_dtype)

            for up_block in self.up_blocks:
                sample = self.up_block(sample, 
                                       is_init_image=is_init_image,
                                       temporal_chunk=temporal_chunk)


        # POST-PROCESS 
        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample, 
                               is_init_image=is_init_image,
                               temporal_chunk=temporal_chunk)

        return sample 



class DiagonalGaussianDistribution(object):

    def __init__(self,
                 parameters: torch.Tensor,
                 deterministic: bool = False):

        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic

        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(
                self.mean, device=self.parameters.device, dtype=self.parameters.dtype
            )

    def sample(self,
               generator: Optional[torch.Generator] = None) -> torch.FloatTensor:

        # make sure sample is on the same device as the parameters and has same dtype 
        sample = randn_tensor(
            self.mean.shape,
            generator=generator,
            device=self.parameters.device,
            dtype=self.parameters.dtype
        )
        x = self.mean + self.std * sample 
        return x 


    def kl(self,
           other: "DiagonalGaussianDistribution" = None) -> torch.Tensor:

        if self.deterministic:
            return torch.Tensor([0.0])

        else:
            if other is None:
                return 0.5 * torch.sum(
                    torch.pow(self.mean, 2) + self.var - 1.0 - self.logvar,
                    dim=[2, 3, 4]
                )

            else:
                return 0.5 * torch.sum(
                    torch.pow(self.mean - other.mean, 2) / other.var 
                    + self.var / other.var 
                    - 1.0 
                    - self.logvar
                    + other.logvar,
                    dim=[2, 3, 4]
                )

    def nll(self,
            sample: torch.Tensor,
            dims: Tuple[int, ...] = [1, 2, 3]) -> torch.Tensor:

        if self.deterministic:
            return torch.Tensor([0.0])

        logtwopi = np.log(2.0 * np.pi)

        return 0.5 * torch.sum(
            logtwopi + self.logvar + torch.pow(sample - self.mean, 2) / self.var,
            dim=dims
        )


    def mode(self) -> torch.Tensor:
        return self.mean 

    

@dataclass
class DecoderOutput(BaseOutput):

    """
    Output of decoding method.

    Args:
        sample (`torch.FloatTensor` or shape `(batch_size, num_channels, height, width)`):
            The decoded output sample from the last layer of the model.
    """


    sample: torch.FloatTensor