import torch 
from torch import nn 
from typing import Optional
from diffusers.models.normalization import AdaGroupNorm
from diffusers.models.attention_processor import SpatialNorm
from diffusers.models.activations import get_activation
from einops import rearrange

from causal_conv import CausalGroupNorm, CausalConv3d


class CausalResnetBlock3D(nn.Module):

    def __init__(self,
                 in_channels: int,
                 out_channels: Optional[int] = None,
                 conv_shortcut: bool = False,
                 dropout: float = 0.0,
                 temb_channels: int = 512,
                 groups: int = 32,
                 pre_norm: bool = True,
                 eps: float = 1e-6,
                 non_linearity: str = "swish",
                 time_embedding_norm: str = "default",  # default, scale_shift, ada_group, spatial
                 output_scale_factor: float = 1.0,
                 use_in_shortcut: Optional[bool] = None,
                  ):

        super().__init__()
        self.pre_norm = pre_norm
        self.pre_norm = True

        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.out_conv_shortcut = conv_shortcut
        self.time_embedding_norm = time_embedding_norm
        self.output_scale_factor = output_scale_factor

        ## <--- Normalization --> ##
        if self.time_embedding_norm == "ada_group":
            self.norm1 = AdaGroupNorm(embedding_dim=temb_channels,
                                      out_dim=in_channels,
                                      num_groups=groups,
                                      eps=eps)

        elif self.time_embedding_norm == "spatial":
            self.norm1 = SpatialNorm(f_channels=in_channels,
                                     zq_channels=temb_channels)

        else:
            self.norm1 = CausalGroupNorm(num_groups=groups,
                                         num_channels=in_channels,
                                         eps=eps,
                                         affine=True)


        ## <--- conv layer --> ##
        self.conv1 = CausalConv3d(in_channels,
                                  output_chaannels=out_channels,
                                  kernel_size=3,
                                  stride=1)

        ## <--- Normalization --> ##
        if self.time_embedding_norm == "ada_group":
            self.norm2 = AdaGroupNorm(embedding_dim=temb_channels,
                                      out_dim=out_channels,
                                      num_groups=groups,
                                      eps=eps)

        elif self.time_embedding_norm == "spatial":
            self.norm2 = SpatialNorm(f_channels=out_channels,
                                     zq_channels=temb_channels)

        else:
            self.norm2 = CausalGroupNorm(num_groups=groups,
                                         num_channels=out_channels,
                                         eps=eps,
                                         affine=True)

        ## <--- conv layer --> ##
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = CausalConv3d(out_channels,
                                  out_channels,
                                  kernel_size=3,
                                  stride=1)

        self.nonlinearity = get_activation(non_linearity)


        self.use_in_shortcut = self.in_channels != out_channels if use_in_shortcut is None else use_in_shortcut
        self.conv_shortcut = None
        if self.use_in_shortcut:
            self.conv_shortcut = CausalConv3d(
                input_channels=in_channels,
                output_chaannels=out_channels,
                kernel_size=1,
                stride=1,
            )


    def forward(self,
                input_tensor: torch.FloatTensor,
                temb: torch.FloatTensor = None,
                is_init_image=True,
                temporal_chunk=False) -> torch.FloatTensor:

        hidden_states = input_tensor

        if self.time_embedding_norm == "ada_group" or self.time_embedding_norm == "spatial":
            hidden_states = self.norm1(hidden_states, temb)

        else:
            hidden_states = self.norm1(hidden_states)

        hidden_states = self.nonlinearity(hidden_states)

        hidden_states = self.conv1(hidden_states, 
                                   is_init_image=is_init_image,
                                   temporal_chunk=temporal_chunk)

        if temb is not None and self.time_embedding_norm == "default":
            hidden_states = hidden_states + temb 

        if self.time_embedding_norm == "ada_group" or self.time_embedding_norm == "spatial":
            hidden_states = self.norm2(hidden_states, temb)

        else:
            hidden_states = self.norm2(hidden_states)

        hidden_states = self.nonlinearity(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.conv2(hidden_states, 
                                   is_init_image=is_init_image, 
                                   temporal_chunk=temporal_chunk)

        if self.conv_shortcut is not None:
            input_tensor = self.conv_shortcut(input_tensor, 
                                              is_init_image=is_init_image,
                                              temporal_chunk=temporal_chunk)

        output_tensor = (input_tensor + hidden_states) / self.output_scale_factor

        return output_tensor



class CausalDownsample2x(nn.Module):

    def __init__(self,
                 channels: int,
                 use_conv: bool = True,
                 name: str = "conv",
                 kernel_size=3,
                 bias=True):

        super().__init__()
        self.channels = channels
        self.use_conv = use_conv
        stride = (1, 2, 2)
        self.name = name 

        if use_conv:
            conv = CausalConv3d(self.channels,
                                self.channels,
                                kernel_size=kernel_size,
                                stride=stride,
                                bias=bias)

        self.conv = conv

    def forward(self, 
                hidden_states: torch.FloatTensor,
                is_init_image=True,
                temporal_chunk=False) -> torch.FloatTensor:

        assert hidden_states.shape[1] == self.channels, "make sure channels match `hidden_state`"
        hidden_states = self.conv(hidden_states,
                                  is_init_image=is_init_image,
                                  temporal_chunk=temporal_chunk)

        return hidden_states



class CausalTemporalDownSample2x(nn.Module):

    def __init__(self,
                 channels: int,
                 use_conv: bool = False,
                 kernel_size=3,
                 bias=True):

        super().__init__()
        self.channels = channels
        self.use_conv = use_conv
        stride = (2, 1, 1)

        if use_conv:
            conv = CausalConv3d(input_channels=self.channels,
                                output_chaannels=self.channels,
                                kernel_size=kernel_size,
                                stride=stride,
                                bias=bias)
        self.conv = conv 

    def forward(self, 
                hidden_states: torch.FloatTensor,
                is_init_image = True,
                temporal_chunk = False) -> torch.FloatTensor:

        assert hidden_states.shape[1] == self.channels
        hidden_states = self.conv(hidden_states, is_init_image=is_init_image, temporal_chunk=temporal_chunk)

        return hidden_states

    

class CausalUpsample2x(nn.Module):

    def __init__(self,
                 channels: int,
                 use_conv: bool = False,
                 name: str = "conv",
                 kernel_size: Optional[int] = 3,
                 bias = True,
                 interpolate = False):

        super().__init__()
        self.channels = channels
        self.use_conv = use_conv
        self.name = name 
        self.interpolate = interpolate

        
        self.conv = CausalConv3d(self.channels,
                                self.channels * 4,
                                kernel_size=kernel_size,
                                stride=1,
                                bias=bias)

        


    def forward(self, 
                hidden_states: torch.FloatTensor,
                is_init_image=True,
                temporal_chunk=False) -> torch.FloatTensor:

        assert hidden_states.shape[1] == self.channels
        hidden_states = self.conv(hidden_states,
                                  is_init_image=is_init_image,
                                  temporal_chunk=temporal_chunk)
        hidden_states = rearrange(hidden_states, 'b (c p1 p2) t h w -> b c t (h p1) (w p2)', p1=2, p2=2)
        return hidden_states 


class CausalTemporalUpsample2x(nn.Module):

    def __init__(self,
                 channels: int,
                 use_conv: bool = True,
                 kernel_size: Optional[int] = 3,
                 bias = True,
                 ):

        super().__init__()
        self.channels = channels
        self.use_conv = use_conv
        

        self.conv = CausalConv3d(self.channels, 
                                 self.channels  * 2, 
                                 kernel_size=kernel_size,
                                 stride=1,
                                 bias=bias)


    def forward(self,
                hidden_states: torch.FloatTensor,
                is_init_image=True,
                temporal_chunk=False) -> torch.FloatTensor:

        assert hidden_states.shape[1] == self.channels
        t = hidden_states.shape[2]
        hidden_states = self.conv(hidden_states,
                                  is_init_image=is_init_image,
                                  temporal_chunk=temporal_chunk)
        hidden_states = rearrange(hidden_states,
                                  'b (c p) t h w -> b c (t p) h w', p=2)

        if is_init_image:
            hidden_states = hidden_states[:, :, 1:]

        return hidden_states
    




            






