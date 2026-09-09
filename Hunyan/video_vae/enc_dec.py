from torch import nn 
import torch 
from typing import Tuple

from .conv import Causal3d
import numpy as np 



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
        self.down_block = nn.ModuleList([])


        # Down 
        output_channels = block_out_channels[0]
        for i, down_block_type in enumerate(down_block_type):
            input_channels = output_channels
            output_channels = block_out_channels[i]

            is_final_block = i == len(block_out_channels) -1
            num_spatial_downsample_layers = int(np.log2(spatial_compression_ratio))
            num_time_downsample_layers = int(np.log2(time_compression_ratio))

            if time_compression_ratio == 4:
                add_spatial_downsample = bool()


    def forward(self,
                sample: torch.FloatTensor) -> torch.FloatTensor:

        assert len(sample.shape) == 5, "The input tensor should have 5 dim."

        sample = self.conv_in(sample)
        print(sample.shape)



if __name__ == "__main__":

    

