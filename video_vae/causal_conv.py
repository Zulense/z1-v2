import torch 
from torch import nn 
from typing import Union, Tuple


def is_odd(num):
    return num % 2 != 0 
 

class CausalConv3d(nn.Module):

    def __init__(self,
                 input_channels,
                 output_chaannels,
                kernel_size: Union[int, Tuple[int, int, int]],
                stride: Union[int, Tuple[int, int, int]],
                pad_mode: str = 'constant',
                **kwargs
                ):

        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = 3 * (kernel_size,)

        time_kernel_size, height_kernel_size, width_kernel_size = kernel_size
        self.time_kernel_size = time_kernel_size
        assert is_odd(height_kernel_size) and is_odd(width_kernel_size), "make sure `height_kernel_size` and `width_kernel_size` is odd number"

    




        



if __name__ == "__main__":

    object = CausalConv3d(input_channels=3,
                          output_chaannels=128,
                          kernel_size=3,
                          stride=1)

    