from torch import nn 
import torch 
from typing import Tuple, Optional
import numpy as np 
from dataclasses import dataclass


from .conv import Causal3d
from .block import get_down_block3d, UNetMidBlockCausal3D, get_up_block3d

from diffusers.models.attention_processor import SpatialNorm
from diffusers.utils import is_torch_version, BaseOutput
from diffusers.utils.torch_utils import randn_tensor



class EncoderCausal3D(nn.Module):

    def __init__(self,
                 in_channels: int = 3,
                 out_channels: int = 3,
                 down_block_type: Tuple[str, ...] = ("DownEncoderBlockCausal3D",),
                 block_out_channels: Tuple[int, ...] = (64,),
                 layers_per_block: int = 2,
                 norm_num_groups: int = 32,
                 act_fn: str = "silu",
                 double_z: bool = True,
                 mid_block_add_attention: bool = True,
                 time_compression_ratio: int = 4,
                 spatial_compression_ratio: int = 8
                 ): 

        super().__init__()
        self.layers_per_block = layers_per_block

        self.conv_in = Causal3d(in_channels=in_channels,
                                out_channels=block_out_channels[0],
                                kernel_size=3,
                                stride=1)
        self.mid_block = None
        self.down_blocks = nn.ModuleList([])


        # Down 
        output_channels = block_out_channels[0] # [128]
        for i, down_block_type in enumerate(down_block_type):
            input_channels = output_channels
            output_channels = block_out_channels[i]

            is_final_block = i == len(block_out_channels) -1  # [0 3]->False, [1 3]->False, [2 3]->False, [3, 3]->True
            num_spatial_downsample_layers = int(np.log2(spatial_compression_ratio)) # 3, 3, 3, 3
            num_time_downsample_layers = int(np.log2(time_compression_ratio)) # 2, 2, 2, 2

            if time_compression_ratio == 4:
                add_spatial_downsample = bool(i < num_spatial_downsample_layers)    # [0 3]->True, [1 3]->True, [2 3]->True, [3 3]->False
                add_time_downsample = bool(
                    i >= (len(block_out_channels) - 1 - num_time_downsample_layers)  # [0>=4-1-2], [1>=4-1-2], [2>=4-1-2], [3>=4-1-2]
                    and not is_final_block
                )
            else:
                raise ValueError(f"Unsupported time_compression_ratio: {time_compression_ratio}")

            downsample_stride_HW = (2, 2) if add_spatial_downsample else (1, 1)
            downsample_stride_T = (2,) if add_time_downsample else (1,)
            downsample_stride = tuple(downsample_stride_T + downsample_stride_HW)
            down_block = get_down_block3d(
                down_block_type=down_block_type,
                num_layers=self.layers_per_block,
                in_channels=input_channels,
                out_channels=output_channels,
                add_downsample=bool(add_spatial_downsample or add_time_downsample),
                downsample_stride=downsample_stride,
                resnet_eps=1e-6,
                downsample_padding=0,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                attention_head_dim=output_channels
            )
            self.down_blocks.append(down_block)

        # mid 
        self.mid_block = UNetMidBlockCausal3D(
            in_channels=block_out_channels[-1],
            temb_channels=None,
            resnet_eps=1e-6,
            resnet_time_scale_shift="default",
            resnet_act_fn=act_fn,
            resnet_groups=norm_num_groups,
            add_attention=mid_block_add_attention,
            attention_head_dim=block_out_channels[-1],
            output_scale_factor=1
        )

        # out 
        self.conv_norm_out = nn.GroupNorm(num_groups=norm_num_groups,
                                          num_channels=block_out_channels[-1],
                                          eps=1e-6)
        self.conv_act = nn.SiLU()

        conv_out_channels = 2 * out_channels if double_z else out_channels
        self.conv_out = Causal3d(in_channels=block_out_channels[-1],
                                 out_channels=conv_out_channels,
                                 kernel_size=3)


    






    def forward(self,
                sample: torch.FloatTensor) -> torch.FloatTensor:

        # sample shape = ([32, 3, 4, 256, 256])

        assert len(sample.shape) == 5, "The input tensor should have 5 dim."

        # ([32, 3, 4, 256, 256]) -> ([32, 64, 4, 256, 256])
        sample = self.conv_in(sample)

        # down 
        for down_block in self.down_blocks:
            sample = down_block(sample)

        # middle 
        sample = self.mid_block(sample)

        # post-process 
        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        return sample
    

class DecoderCausal3D(nn.Module):

    def __init__(
            self,
            in_channels: int = 3,
            out_channels: int = 3,
            up_block_types: Tuple[str, ...] = ("UpDecoderBlockCausal3D",),
            block_out_channels: Tuple[int, ...] = (64,),
            layers_per_block: int = 2,
            norm_num_groups:int = 32,
            act_fn: str = "silu",
            norm_type: str = "group",  # group, spatial
            mid_block_add_attention=True,
            time_compression_ratio: int = 4,
            spatial_compression_ratio: int = 8
    ):

        super.__init__()
        self.layer_per_block = self.layer_per_block

        self.conv_in = Causal3d(in_channels=in_channels,
                                out_channels=block_out_channels[-1],
                                kernel_size=3,
                                stride=1)
        self.mid_block = None
        self.up_blocks = nn.ModuleList([])

        temb_channels = in_channels if norm_type == "spatial" else None

        # mid 
        self.mid_block = UNetMidBlockCausal3D(
                    in_channels=block_out_channels[-1],
                    temb_channels=temb_channels,
                    resnet_eps=1e-6,
                    resnet_time_scale_shift="default" if norm_type == "group" else norm_type,
                    resnet_act_fn=act_fn,
                    resnet_groups=norm_num_groups,
                    add_attention=mid_block_add_attention,
                    attention_head_dim=block_out_channels[-1],
                    output_scale_factor=1
                )

        # up 
        reversed_block_out_channels = list(reversed(block_out_channels))
        output_channel = reversed_block_out_channels[0]

        for i, up_block_types in enumerate(up_block_types):
            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]

            is_final_block = i == len(block_out_channels) -1
            num_spatial_upsample_layers = int(np.log2(spatial_compression_ratio))
            num_time_upsample_layers = int(np.log2(time_compression_ratio))

            if time_compression_ratio == 4:
                add_spatial_upsample = bool(i < num_spatial_upsample_layers)
                add_time_upsample = bool(
                    i >= len(block_out_channels) - 1 - num_time_upsample_layers and not is_final_block
                )
            else:
                raise ValueError(f"Unsupported time_compression_ratio: {time_compression_ratio}")

            upsample_scale_factor_HW = (2, 2) if add_spatial_upsample else (1, 1)
            upsample_scale_factor_T = (2,) if add_time_upsample else (1,)
            upsample_scale_factor = tuple(upsample_scale_factor_T + upsample_scale_factor_HW)

            up_block = get_up_block3d(
                up_block_type=up_block_types,
                num_layers=self.layer_per_block + 1,
                in_channels=prev_output_channel,
                out_channels=output_channel,
                temb_channels=temb_channels,
                add_upsample=bool(add_spatial_upsample or add_time_upsample),
                upsample_scale_factor=upsample_scale_factor,
                resnet_eps=1e-6,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                attention_head_dim=output_channel
            )

            self.up_blocks.append(up_block)
            prev_output_channel = output_channel


        # out 
        if norm_type == "spatial":
            self.conv_norm_out = SpatialNorm(f_channels=block_out_channels[0],
                                             zq_channels=temb_channels)
        else:
            self.conv_norm_out = nn.GroupNorm(num_groups=norm_num_groups,
                                              num_channels=block_out_channels[0],
                                              eps=1e-6)
        self.conv_act = nn.SiLU()
        self.conv_out = Causal3d(in_channels=block_out_channels[0],
                                 out_channels=out_channels,
                                 kernel_size=3)

        self.gradient_checkpointing = False

    def forward(self,
                sample: torch.FloatTensor,
                latent_embeds: Optional[torch.FloatTensor] = None) -> torch.FloatTensor:


        assert len(sample.shape) == 5, "The input tensor should have 5 dimensions."

        sample = self.conv_in(sample)

        upscale_dtype = next(iter(self.up_blocks.parameters())).dtype
        if self.training and self.gradient_checkpointing:

            def create_custom_forward(module):
                def custom_forward(*inputs):
                    return module(*inputs)

                return custom_forward

            if is_torch_version(">=", "1.11.0"):
                # middle 
                sample = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(self.mid_block),
                    sample,
                    latent_embeds,
                    use_reentrant=False
                )
                sample = sample.to(upscale_dtype)

                # up 
                for up_block in self.up_blocks:
                    sample = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(up_block),
                        sample,
                        latent_embeds,
                        use_reentrant=False,
                    )

        # post-process 
        if latent_embeds is None:
            sample = self.conv_norm_out(sample)
        else:
            sample = self.conv_norm_out(sample, latent_embeds)

        sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        return sample


class DiagonalGaussianDistribution(object):

    def __init__(self,
                 parameters: torch.Tensor,
                 deterministic: bool = False):

        if parameters.ndim == 3:
            dim = 2  # (B, L, C)
        elif parameters.ndim == 5 or parameters.ndim == 4:
            dim = 1  # (B, C, T, H, W) / (B, C, H, W)

        else:
            raise NotImplementedError


        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=dim)

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

    def mode(self) -> torch.Tensor:
        return self.mean




@dataclass
class DecoderOutput(BaseOutput):
    sample: torch.FloatTensor

@dataclass
class DecoderOutput2(BaseOutput):
    sample: torch.FloatTensor
    posterior: Optional[DiagonalGaussianDistribution] = None




if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EncoderCausal3D(in_channels=3,
                            out_channels=3,
                            down_block_type=("DownEncoderBlockCausal3D",
                                             "DownEncoderBlockCausal3D",
                                             "DownEncoderBlockCausal3D",
                                             "DownEncoderBlockCausal3D",),
                            block_out_channels=(128, 256, 512, 512,),
                            )

    x = torch.randn(32, 3, 4, 256, 256)
    out = model(x)

