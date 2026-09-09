import torch 
from torch import nn 
from typing import Optional, Tuple
from diffusers.utils import logging
logger = logging.get_logger(__name__)
from diffusers.models.attention_processor import Attention
from einops import rearrange

from .resnet import ResnetBlockCausal3D, DownsampleCausal3D, UpsampleCausal3D



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
                 add_downsample: bool = True,
                 downsample_stride: int = 2,
                 downsample_padding: int = 1):

        super().__init__()
        resnets = []

        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            resnets.append(
                ResnetBlockCausal3D(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    dropout=dropout,
                    temb_channels=None,
                    groups=resnet_groups,
                    pre_norm=resnet_pre_norm,
                    eps=resnet_eps,
                    non_linearity=resnet_act_fn,
                    time_embedding_norm=resnet_time_scale_shift,
                    output_scale_factor=output_scale_factor,
                )
            )
        self.resnets = nn.ModuleList(resnets)

        if add_downsample:
            self.downsamplers = nn.ModuleList(
                [
                    DownsampleCausal3D(
                        channels=out_channels,
                        use_conv=True,
                        out_channels=out_channels,
                        padding=downsample_padding,
                        name="op",
                        stride=downsample_stride
                    )
                ]
            ) 
        else:
            self.downsamplers = None


    def forward(self,
                hidden_states: torch.FloatTensor,
                scale: float = 1.0) -> torch.FloatTensor:

        for resnet in self.resnets:
            hidden_states = resnet(hidden_states, temb=None, scale=scale)

        if self.downsamplers is None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states, scale)

        return hidden_states


def prepare_causal_attention_mask(n_frame: int,
                                  n_hw: int,
                                  dtype, 
                                  device,
                                  batch_size: int = None):


    seq_len = n_frame * n_hw
    mask = torch.full((seq_len, seq_len), float("-inf"), dtype=dtype, device=device)

    for i in range(seq_len):
        i_frame = i // n_hw
        mask[i, : (i_frame + 1) * n_hw] = 0 

    if batch_size is not None:
        mask = mask.unsqueeze(0).expand(batch_size, -1, -1)

    return mask



class UNetMidBlockCausal3D(nn.Module):

    def __init__(self,
                 in_channels: int,
                 temb_channels: int,
                 dropout: float = 0.0,
                 num_layers: int = 1,
                 resnet_eps: float = 1e-6,
                 resnet_time_scale_shift: str = "difault",
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
            ResnetBlockCausal3D(
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
            logger.warning(f"It is not recommend to pass `attention_head_dim=None`. Default `attention_head_dim` to `in_channels`: {in_channels}")
            attention_head_dim = in_channels


        for _ in range(num_layers):
            if self.add_attention:
                attentions.append(
                    Attention(
                        in_channels,
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
                ResnetBlockCausal3D(
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
                temb: Optional[torch.FloatTensor] = None
                ) -> torch.FloatTensor:

        hidden_states = self.resnets[0](hidden_states, temb)

        for attn, resnet in zip(self.attentions, self.resnets[1:]):
            if attn is not None:
                B, C, T, H, W = hidden_states.shape
                hidden_states = rearrange(hidden_states, "b c t h w -> b (t h w) c")
                attention_mask = prepare_causal_attention_mask(
                    T, H * W, hidden_states.dtype, hidden_states.device, batch_size=B
                )
                hidden_states = attn(hidden_states, temb=temb, attention_mask=attention_mask)
                hidden_states = rearrange(hidden_states,
                                          "b (t h w) c -> b c t h w", t=T, h=H, w=W)

            hidden_states = resnet(hidden_states, temb)

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
                 add_upsample: bool = True,
                 upsample_scale_factor=(2, 2, 2),
                 temb_channels: Optional[int] = None
                 ):

        super().__init__()

        resnets = []
        for i in range(num_layers):
            input_channels = in_channels if i == 0 else out_channels

            resnets.append(
                ResnetBlockCausal3D(
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

        if add_upsample:
            self.upsamplers = nn.ModuleList([
                UpsampleCausal3D(
                    channels=out_channels,
                    use_conv=True,
                    out_channels=out_channels,
                    upsample_factor=upsample_scale_factor
                )
            ])
        else:
            self.upsamplers = None

        self.resolution_idx = resolution_idx


    def forward(self, 
                hidden_states: torch.FloatTensor,
                temb: Optional[torch.FloatTensor] = None,
                scale: float = 1.0) -> torch.FloatTensor:

        for resnet in self.resnets:
            hidden_states = resnet(hidden_states, temb=temb, scale=scale)

        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states)

        return hidden_states



def get_down_block3d(
        down_block_type: str,
        num_layers: int,
        in_channels: int,
        out_channels: int,
        add_downsample: bool,
        downsample_stride: int,
        resnet_eps: float,
        resnet_act_fn: str,
        num_attention_heads: Optional[int] = None,
        resnet_groups: Optional[int] = None,
        downsample_padding: Optional[int] = None,
        resnet_time_scale_shift: str = "default",
        attention_head_dim: Optional[int] = None,
        dropout: float = 0.0

):

    # If attention head dim is not defined, we default it to the number of heads
    if attention_head_dim is None:
        logger.warning(
            f"It is recommended to provide `attention_head_dim` when calling `get_down_block`. Default `attention_head_dim` = {num_attention_heads}"
        )
        attention_head_dim = num_attention_heads

    down_block_type = down_block_type[7:] if down_block_type.startswith("UNetRes") else down_block_type
    if down_block_type == "DownEncoderBlockCausal3D":
        return DownEncoderBlockCausal3D(
            num_layers=num_layers,
            in_channels=in_channels,
            out_channels=out_channels,
            dropout=dropout,
            add_downsample=add_downsample,
            downsample_stride=downsample_stride,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            resnet_groups=resnet_groups,
            downsample_padding=downsample_padding,
            resnet_time_scale_shift=resnet_time_scale_shift
        )
    raise ValueError(f"{down_block_type} does not exist.")




def get_up_block3d(
        up_block_type: str,
        num_layers: int,
        in_channels: int,
        out_channels: int,
        temb_channels: int,
        add_upsample: bool,
        upsample_scale_factor: Tuple,
        resnet_eps: float,
        resnet_act_fn: str,
        resolution_idx: Optional[int] = None,
        num_attention_heads: Optional[int] = None,
        resnet_groups: Optional[int] = None,
        resnet_time_scale_shift: str = "default",
        attention_head_dim: Optional[int] = None,
        dropout: float = 0.0
) -> nn.Module:

    # if attention head dim is not defined, we default it to the number of heads.
    if attention_head_dim is None:
        logger.warning(
            f"It is recommend to provide `attention_head_dim` when calling `get_up_block`. Default `attention_head_dim` = {num_attention_heads}"
        )
        attention_head_dim = num_attention_heads

    up_block_type = up_block_type[7:] if up_block_type.startswith("UNetRes") else up_block_type
    if up_block_type == "UpDecoderBlockCausal3D":
        return UpDecoderBlockCausal3D(
            num_layers=num_layers,
            in_channels=in_channels,
            out_channels=out_channels,
            resolution_idx=resolution_idx,
            dropout=dropout,
            add_upsample=add_upsample,
            upsample_scale_factor=upsample_scale_factor,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            resnet_groups=resnet_groups,
            resnet_time_scale_shift=resnet_time_scale_shift,
            temb_channels=temb_channels
        )
    raise ValueError(f"{up_block_type} does not exist.")








