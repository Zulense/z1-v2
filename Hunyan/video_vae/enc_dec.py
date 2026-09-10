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
                print(f"{len(block_out_channels) -1 - num_time_downsample_layers}")
            else:
                raise ValueError(f"Unsupported time_compression_ratio: {time_compression_ratio}")



    def forward(self,
                sample: torch.FloatTensor) -> torch.FloatTensor:

        # sample shape = ([32, 3, 4, 256, 256])

        assert len(sample.shape) == 5, "The input tensor should have 5 dim."

        # ([32, 3, 4, 256, 256]) -> ([32, 64, 4, 256, 256])
        sample = self.conv_in(sample)

        # down 
        for down_block in self.down_blocks:
            sample = down_block(sample)
            # print(sample.shape)



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

