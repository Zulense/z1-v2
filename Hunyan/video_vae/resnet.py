import torch 
from torch import nn 
from typing import Optional
from diffusers.models.normalization import RMSNorm, AdaGroupNorm
from diffusers.models.attention_processor import SpatialNorm
from diffusers.models.activations import get_activation
from einops import rearrange
from torch.nn import functional as F

from .conv import Causal3d



class DownsampleCausal3D(nn.Module):

    def __init__(
            self,
            channels: int,
            use_conv: bool = False,
            out_channels: Optional[int] = None,
            padding: int = 1,
            name: str = "conv",
            kernel_size=3,
            norm_type=None,
            eps=None,
            elementwise_affine=None,
            bias=True,
            stride=2
    ):

        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.padding = padding
        stride = stride
        self.name = name 

        if norm_type == "ln_norm":
            self.norm = nn.LayerNorm(channels, eps, elementwise_affine)
        elif norm_type == "rms_norm":
            self.norm = RMSNorm(channels, eps, elementwise_affine)
        elif norm_type is None:
            self.norm = None
        else:
            raise ValueError(f"unknown norm_type: {norm_type}")

        if use_conv:
            conv = Causal3d(
                self.channels,
                self.out_channels,
                kernel_size=kernel_size,
                stride=stride,
                bias=bias
            )
        else:
            raise NotImplementedError

        if name == "conv":
            self.conv2d_0 = conv 
            self.conv = conv 
        elif name == "Conv2d_0":
            self.conv = conv 
        else:
            self.conv = conv 

    def forward(self, hidden_states: torch.FloatTensor, scale: float = 1.0) -> torch.FloatTensor:

        assert hidden_states.shape[1] == self.channels

        # For video
        # t = hidden_states.shape[2]
        # hidden_states = rearrange(hidden_states, 
        #                           'b c t h w -> (b t) c h w')

        if self.norm is not None:
            # hidden_states = self.norm(hidden_states.permute(0, 2, 3, 1))
            hidden_states = hidden_states.permute(0, 2, 3, 1)
            hidden_states = self.norm(hidden_states)
            hidden_states = hidden_states.permute(0, 3, 1, 2)
            print(hidden_states.shape)
         

        assert hidden_states.shape[1] == self.channels

        # for video
        # hidden_states = rearrange(hidden_states,
        #                           '(b t) c h w -> b c t h w', t=t)
        hidden_states = self.conv(hidden_states)

        return hidden_states


class UpsampleCausal3D(nn.Module):

    def __init__(self,
                 channels: int,
                 use_conv: bool = False,
                 use_conv_transpose: bool = False,
                 out_channels: Optional[int] = None,
                 name: str = "conv",
                 kernel_size: Optional[int] = None,
                 padding = 1,
                 norm_type = None,
                 eps=None,
                 elementwise_affine=None,
                 bias=True,
                 interpolate=True,
                 upsample_factor=(2, 2, 2)):


        super().__init__()
        self.channels = channels
        self.out_channels = out_channels
        self.use_conv = use_conv
        self.use_conv_transpose = use_conv_transpose
        self.name = name 
        self.interpolate = interpolate
        self.upsample_factor = upsample_factor

        if norm_type == "ln_norm":
            self.norm = nn.LayerNorm(channels, eps, elementwise_affine)
        elif norm_type == "rms_norm":
            self.norm = RMSNorm(channels, eps, elementwise_affine)
        elif norm_type is None:
            self.norm = None
        else:
            raise ValueError(f"unknown norm_type: {norm_type}")

        conv = None 
        if use_conv_transpose:
            raise NotImplementedError
        elif use_conv:
            if kernel_size is None:
                kernel_size = 3 
            conv = Causal3d(self.channels,
                            self.out_channels,
                            kernel_size=kernel_size,
                            bias=bias)

        if name == "conv":
            self.conv = conv 
        else:
            self.conv2d_0 = conv 


    def forward(self,
                hidden_states: torch.FloatTensor,
                output_size: Optional[int] = None,
                scale: float = 1.0) -> torch.FloatTensor:

        assert hidden_states.shape[1] == self.channels

        if self.norm is not None:
            raise NotImplementedError

        if self.use_conv_transpose:
            raise NotImplementedError

        # Cast to float32 to as 'upsample_nearest2d_out_frame' op does not support bfloat16
        dtype = hidden_states.dtype 
        if dtype == torch.bfloat16:
            hidden_states = hidden_states.to(torch.float32)

        # upsample_nearest_nhwc fails with large batch size.
        if hidden_states.shape[0] >= 64:
            hidden_states = hidden_states.contiguous()

        # if 'output_size' is passed we force the interpolation output
        # size and do not make use of `scale_factor=2`
        if self.interpolate:
            B, C, T, H, W = hidden_states.shape 
            first_h, other_h = hidden_states.split((1, T - 1), dim=2)
            
            if output_size is None:
                if T > 1:
                    other_h = F.interpolate(other_h, 
                                            scale_factor=self.upsample_factor,
                                            mode="nearest")

                first_h = first_h.squeeze(2)
                first_h = F.interpolate(first_h, scale_factor=self.upsample_factor[1:], mode="nearest")
                first_h = first_h.unsqueeze(2)

            else:
                raise NotImplementedError

            if T > 1:
                hidden_states = torch.cat((first_h, other_h), dim=2)
            else:
                hidden_states = first_h

        # if the input if bf16, we cast back to bf16
        if dtype == torch.bfloat16:
            hidden_states = hidden_states.to(torch.bfloat16)

        if self.use_conv:
            if self.name == "conv":
                hidden_states = self.conv(hidden_states)
            else:
                hidden_states = self.conv2d_0(hidden_states)

        return hidden_states



class ResnetBlockCausal3D(nn.Module):

    def __init__(self,
                 *,
                 in_channels: int,
                 out_channels: int = None,
                 conv_shortcut: bool = False,
                 dropout: float = 0.0,
                 temb_channels: int = 512,
                 groups: int = 32,
                 groups_out: Optional[int] = None,
                 pre_norm: bool = True,
                 eps: float = 1e-6,
                 non_linearity: str = "swish",
                 skip_time_act: bool = False,
                 time_embedding_norm: str = "default",  # # default, scale_shift, ada_group, spatial
                 kernel: Optional[torch.FloatTensor] = None,
                 output_scale_factor: Optional[bool] = True,
                 use_in_shortcut: Optional[bool] = None,
                 up: bool = False,
                 down: bool = False,
                 conv_shortcut_bias: bool = True,
                 conv_3d_out_channels: Optional[int] = None
                 ):

        super().__init__()
        self.pre_norm = pre_norm
        self.pre_norm = True 
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut
        self.up = up 
        self.down = down 
        self.output_scale_factor = output_scale_factor
        self.time_embedding_norm = time_embedding_norm
        self.skip_time_act = skip_time_act

        linear_cls = nn.Linear

        if groups_out is None:
            groups_out = groups

        if self.time_embedding_norm == "ada_group":   # for Image
            self.norm1 = AdaGroupNorm(temb_channels, out_channels, groups_out, eps=eps)
        elif self.time_embedding_norm == "spatial":     # for Image
            self.norm1 = SpatialNorm(in_channels, temb_channels)
        else:
            self.norm1 = nn.GroupNorm(groups, in_channels, eps=eps, affine=True)

        self.conv1 = Causal3d(in_channels,
                              out_channels,
                              kernel_size=3,
                              stride=1)

        if temb_channels is not None:
            if self.time_embedding_norm == "default":
                self.time_emb_proj = linear_cls(temb_channels, out_channels)
            elif self.time_embedding_norm == "scale_shift":
                self.time_emb_proj = linear_cls(temb_channels, 2 * out_channels)
            elif self.time_embedding_norm == "ada_group" or self.time_embedding_norm == "spatial":
                self.time_emb_proj = None 
            else:
                raise ValueError(f"Unknown time_embedding_norm: {self.time_embedding_norm}")

        else:
            self.time_emb_proj = None 


        if self.time_embedding_norm == "ada_group":
            self.norm2 = AdaGroupNorm(temb_channels, out_channels, groups_out, eps=eps)
        elif self.time_embedding_norm == "spatial":
            self.norm2 = SpatialNorm(out_channels, temb_channels)
        else:
            self.norm2 = nn.GroupNorm(num_groups=groups_out,
                                      num_channels=out_channels,
                                      eps=eps,
                                      affine=True)

        self.dropout = nn.Dropout(dropout)
        conv_3d_out_channels = conv_3d_out_channels or out_channels
        self.conv2 = Causal3d(out_channels,
                              conv_3d_out_channels,
                              kernel_size=3,
                              stride=1)

        self.nonlinearity = get_activation(non_linearity)
        self.upsample = self.downsample = None
        if self.up:
            self.upsample = UpsampleCausal3D(in_channels, use_conv=False)
        elif self.down:
            self.downsample = DownsampleCausal3D(in_channels, use_conv=False, name="op")

        self.use_in_shortcut = self.in_channels != conv_3d_out_channels if use_in_shortcut is None else self.use_in_shortcut

        self.conv_shortcut = None
        if self.use_in_shortcut:
            self.conv_shortcut = Causal3d(
                in_channels,
                conv_3d_out_channels,
                kernel_size=1,
                stride=1,
                bias=conv_shortcut_bias
            )

    def forward(self, 
                input_tensor: torch.FloatTensor,
                temb: torch.FloatTensor,
                scale: float = 1.0,
                ) -> torch.FloatTensor:

        hidden_states = input_tensor
        

        if self.time_embedding_norm == "ada_group" or self.time_embedding_norm == "spatial":
            hidden_states = self.norm1(hidden_states, temb)
        else:
            hidden_states = self.norm1(hidden_states)
        

        hidden_states = self.nonlinearity(hidden_states)

        if self.upsample is not None:
            if hidden_states.shape[0] >= 64:
                input_tensor = input_tensor.contiguous()
                hidden_states = hidden_states.contiguous()

            input_tensor = (
                self.upsample(input_tensor, scale=scale)
            )
            hidden_states = (
                self.upsample(hidden_states, scale=scale)
            )

        elif self.downsample is not None:
            input_tensor = (
                self.downsample(input_tensor, scale=scale)
            )
            hidden_states = (
                self.downsample(hidden_states, scale=scale)
            )

        hidden_states = self.conv1(hidden_states)

        if self.time_emb_proj is not None:
            if not self.skip_time_act:
                temb = self.nonlinearity(temb)

            temb = (
                self.time_emb_proj(temb)[:, :, None, None]
            )

        ## [video] Testing
        # temb = temb[:, :, :, :, None]

        
        if temb is not None and self.time_embedding_norm == "default":
            hidden_states = hidden_states + temb  # ([32, 128, 2, 32, 32]) + ([32, 128, 1, 1])

        if self.time_embedding_norm == "ada_group" or self.time_embedding_norm == "spatial":
            hidden_states = self.norm2(hidden_states, temb)
        else:
            hidden_states = self.norm2(hidden_states)

        if temb is not None and self.time_embedding_norm == "scale_shift":
            scale, shift = torch.chunk(temb, 2, dim=1)
            hidden_states = hidden_states * (1 + scale) + shift

        hidden_states = self.nonlinearity(hidden_states)

        hidden_states = self.dropout(hidden_states)
        hidden_states = self.conv2(hidden_states)

        if self.conv_shortcut is not None:
            input_tensor = (
                self.conv_shortcut(input_tensor)
            )

        output_tensor = (input_tensor + hidden_states) / self.output_scale_factor
        return output_tensor


        

        

            


            




        
    


        

    








if __name__ == "__main__":

    x = torch.randn(32, 128, 2, 32, 32)
    temb = torch.randn(32, 512)
    # model = DownsampleCausal3D(channels=3,
    #                          use_conv=True,
    #                          out_channels=3,
    #                          norm_type="rms_norm",
    #                          eps=1e-6)

    # out = model(x)
    # print(out.shape)
    # ------------------------------------------------------
    # model = UpsampleCausal3D(channels=3, 
    #                          use_conv=True,
    #                         #  out_channels=3,
    #                          kernel_size=3,
    #                          eps=1e-6,
    #                          )

    # out = model(x)
    # print(out.shape)
    # ----------------------------------------------------------

    model = ResnetBlockCausal3D(in_channels=128,
                                out_channels=128,
                                kernel=3,
                                time_embedding_norm="default"
                                )

    out = model(x, temb)
    # print(out)

    
    
