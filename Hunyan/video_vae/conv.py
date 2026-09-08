import torch 
from torch import nn 
from typing import Union, Tuple
from torch.nn import functional as F

class Causal3d(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size: Union[int, Tuple[int, int, int]],
                 stride: Union[int, Tuple[int, int, int]] = 1,
                 dilation: Union[int, Tuple[int, int, int]] = 1,
                 pad_mode='replicate',
                 **kwargs
                 ):

        super().__init__()

        
        self.pad_mode = pad_mode
        padding = (kernel_size // 2, kernel_size // 2, kernel_size // 2, kernel_size // 2, kernel_size - 1, 0)  # W, H, T
        self.time_causal_padding = padding

        self.conv = nn.Conv3d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size,
                              stride=stride,
                              dilation=dilation,
                              **kwargs)

    def forward(self, x):

        x = F.pad(x, self.time_causal_padding, mode=self.pad_mode)
        return self.conv(x)



