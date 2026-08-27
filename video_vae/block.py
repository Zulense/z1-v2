import torch 
from typing import Optional
from torch import nn 
from diffusers.utils import logging
from diffusers.models.attention_processor import Attention
from einops import rearrange

from .resnet import CausalResnetBlock3D, CausalDownsample2x, CausalTemporalDownSample2x, CausalTemporalUpsample2x, CausalUpsample2x
logger = logging.get_logger(__name__)



def get_input_layer(
        in_channels: int,
        out_channels: int,
        norm_num_groups: int,
        layer_type: str,
        norm_type: str = "group",
        affine: bool = True
):

    if layer_type == 'conv':
        input_layer = nn.Conv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1
        )

    elif layer_type == "pixel_shuffle":
        input_layer = nn.Sequential(
            nn.PixelUnshuffle(2),
            nn.Conv2d(in_channels=in_channels * 4,
                      out_channels=out_channels,
                      kernel_size=1)
        )

    else:
        raise NotImplementedError(f"Not support input layer {layer_type}")

    return input_layer


def get_output_layer(in_channels: int,
                     out_channels: int,
                     norm_num_groups: int,
                     layer_type: str,
                     affine: bool = True):

    if layer_type == 'norm_act_conv':
        output_layer = nn.Sequential(
            nn.GroupNorm(num_groups=norm_num_groups,
                         num_channels=in_channels,
                         eps=1e-6,
                         affine=affine),
            nn.SiLU(),
            nn.Conv3d(in_channels=in_channels,
                      out_channels=out_channels,
                      kernel_size=3,
                      stride=1,
                      padding=1)
        )

    elif layer_type == "pixel_shuffle":
        output_layer = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,
                      out_channels=out_channels * 4,
                      kernel_size=1),
            nn.PixelShuffle(2)
        )

    else:
        raise NotImplementedError(F"Not Support output layer {layer_type}")

    return output_layer





class DownEncoderBlockCausal3D(nn.Module):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 dropout: float = 0.0,
                 num_layers: int = 1,
                 resnet_eps: float = 1e-6,
                 resnet_time_scale_shift: str = "default",
                 resnet_act_fn: str = "swish",
                 resnet_groups: int = 32,
                 resnet_pre_norm: bool = True,
                 output_scale_factor: float = 1.0,
                 add_spatial_downsample: bool = True,
                 add_temporal_downsample: bool = False,
                 ):

        super().__init__()

        resnets = []
        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            resnets.append(
                CausalResnetBlock3D(in_channels=in_channels,
                                    out_channels=out_channels,
                                    dropout=dropout,
                                    temb_channels=None,
                                    groups=resnet_groups,
                                    time_embedding_norm=resnet_time_scale_shift,
                                    eps=resnet_eps,
                                    non_linearity=resnet_act_fn,
                                    output_scale_factor=output_scale_factor,
                                    pre_norm=resnet_pre_norm
                                    )
            )

        self.resnets = nn.ModuleList(resnets)

        if add_spatial_downsample:
            self.downsamplers = nn.ModuleList([
                CausalDownsample2x(channels=out_channels,
                                   use_conv=True,
                                   )
            ])

        if add_temporal_downsample:
            self.temporal_downsamplers = nn.ModuleList([
                CausalTemporalDownSample2x(
                    channels=out_channels,
                    use_conv=True,
                )
            ])

    def forward(self,
                hidden_states: torch.FloatTensor,
                is_init_image=True,
                temporal_chunk = False) -> torch.FloatTensor:

        for resnet in self.resnets:
            hidden_states = resnet(hidden_states,
                                   temb=None,
                                   is_init_image=is_init_image,
                                   temporal_chunk=temporal_chunk)

        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states,
                                            is_init_image=is_init_image,
                                            temporal_chunk=temporal_chunk)

        if self.temporal_downsamplers is not None:
            for temporal_downsampler in self.temporal_downsamplers:
                hidden_states = temporal_downsampler(hidden_states,
                                                     is_init_image=is_init_image,
                                                     temporal_chunk=temporal_chunk)

        return hidden_states


class MidBlockCausal3D(nn.Module):

    def __init__(self,
                in_channels: int,
                temb_channels: int,
                dropout: float = 0.0,
                num_layers: int = 1,
                resnet_eps: float = 1e-6,
                resnet_time_scale_shift: str = "default",  # default, spatial
                resnet_act_fn: str = "swish",
                resnet_groups: int = 32,
                attn_groups: Optional[int] = None,
                resnet_pre_norm: bool = True,
                add_attention: bool = True,
                attention_head_dim: int = 1,
                output_scale_factor: float = 1.0
                 ):
        super().__init__()
        resnet_groups = resnet_groups if resnet_groups is not None else min(in_channels // 4, 32)
        self.add_attention = add_attention

        if attn_groups is None:
            attn_groups = resnet_groups if resnet_time_scale_shift == "default" else None 

        # there is always at least one resnet 
        resnets = [
            CausalResnetBlock3D(
                in_channels=in_channels,
                out_channels=in_channels,
                temb_channels=temb_channels,
                eps=resnet_eps,
                groups=resnet_groups,
                dropout=dropout,
                time_embedding_norm=resnet_time_scale_shift,
                non_linearity=resnet_act_fn,
                output_scale_factor=output_scale_factor,
                pre_norm=resnet_pre_norm
            )
        ]

        attentions = []
        if attention_head_dim is None:
            logger.warn(f"It is not recommend to pass `attention_head_dim=None`. Default `attention_head_dim` to `in_channels`: {in_channels}")
            attention_head_dim = in_channels

        for _ in range(num_layers):
            if self.add_attention:
                # Spatial attention
                attentions.append(
                    Attention(
                        query_dim=in_channels,
                        heads=in_channels // attention_head_dim,
                        dim_head=attention_head_dim,
                        rescale_output_factor=output_scale_factor,
                        eps=resnet_eps,
                        norm_num_groups=attn_groups,
                        spatial_norm_dim=temb_channels if resnet_time_scale_shift == "spatial" else None,
                        residual_connection=True,
                        bias=True,
                        upcast_softmax=True,
                        _from_deprecated_attn_block=True
                    )
                )
            else:
                attentions.append(None)

            resnets.append(
                CausalResnetBlock3D(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm
                )
            )

        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)


    def forward(self,
                hidden_states: torch.FloatTensor,
                temb: Optional[torch.FloatTensor] = None,
                is_init_image = True,
                temporal_chunk = False) -> torch.FloatTensor:

        hidden_states = self.resnets[0](hidden_states, temb, is_init_image=is_init_image, temporal_chunk=temporal_chunk)
        t = hidden_states.shape[2]

        for attn, resnet in zip(self.attentions, self.resnets[1:]):
            if attn is not None:
                hidden_states = rearrange(hidden_states, 'b c t h w -> b t c h w')
                hidden_states = rearrange(hidden_states, 'b t c h w -> (b t) c h w')
                hidden_states = attn(hidden_states, temb)
                hidden_states = rearrange(hidden_states, '(b t) c h w -> b t c h w', t=t)
                hidden_states = rearrange(hidden_states, 'b t c h w -> b c t h w')

            hidden_states = resnet(hidden_states, temb, is_init_image, temporal_chunk)

        return hidden_states

    

class UpDecoderBlockCausal3D(nn.Module):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 resolution_idx: Optional[int] = None,
                 dropout: float = 0.0,
                 num_layers: int = 1,
                 resnet_eps: float = 1e-6,
                 resnet_time_scale_shift: str = "default",  # default, spatial
                 resnet_act_fn: str = "swish",
                 resnet_groups: int = 32,
                 resnet_pre_norm: bool = True,
                 output_scale_factor: float = 1.0,
                 add_spatial_upsample: bool = True,
                 add_temporal_upsample: bool = False,
                 temb_channels: Optional[int] = None,
                 ): 

        super().__init__()
        resnets = []

        for i in range(num_layers):
            input_channels = in_channels if i == 0 else out_channels

            resnets.append(
                CausalResnetBlock3D(
                    in_channels=input_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm
                )
            )
        self.resnets = nn.ModuleList(resnets)

        if add_spatial_upsample:
            self.upsamplers = nn.ModuleList([
                CausalUpsample2x(channels=out_channels,
                                 use_conv=True,
                                 )
            ])

        if add_temporal_upsample:
            self.temporal_upsamplers = nn.ModuleList([
                CausalTemporalUpsample2x(channels=out_channels,
                                         use_conv=True,
                                         )
            ])

        self.resolution_idx = resolution_idx


    def forward(self,
                hidden_states: torch.FloatTensor,
                temb: Optional[torch.FloatTensor] = None,
                is_init_image = True,
                temporal_chunk = False) -> torch.FloatTensor:

        for resnet in self.resnets:
            hidden_states = resnet(hidden_states,
                                   temb=temb,
                                   is_init_image=is_init_image,
                                   temporal_chunk=temporal_chunk)

        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states,
                                          is_init_image=is_init_image,
                                          temporal_chunk=temporal_chunk)

        if self.temporal_upsamplers is not None:
            for temporal_upsampler in self.temporal_upsamplers:
                hidden_states = temporal_upsampler(hidden_states,
                                                         is_init_image=is_init_image,
                                                         temporal_chunk=temporal_chunk)

        return hidden_states


