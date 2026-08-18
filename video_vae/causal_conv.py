import torch 
from torch import nn 
from typing import Union, Tuple
from timm.layers import trunc_normal_
from collections import deque

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

        dilation = kwargs.pop('dilation', 1)
        self.pad_mode = pad_mode

        if isinstance(stride, int):
            stride = (stride, 1, 1)

        time_pad = dilation * (time_kernel_size - 1)
        height_pad = height_kernel_size // 2 
        width_pad = width_kernel_size // 2 

        self.temporal_stride = stride[0]
        self.time_pad = time_pad
        self.time_causal_padding = (width_pad, width_pad, height_pad, height_pad, time_pad, 0)
        self.time_uncausal_padding = (width_pad, width_pad, height_pad, height_pad, 0, 0)

        self.conv = nn.Conv3d(in_channels=input_channels,
                              out_channels=output_chaannels,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=0,
                              dilation=dilation,
                              **kwargs)

        # initilaize an empty queue to act as a memory buffer. This stores the last frames of a video
        # chunk to feed into the next chunk during inference.
        self.cache_front_feat = deque()


    def _init__weight(self, m):
        if isinstance(m, (nn.Linear, nn.Conv2d, nn.Conv3d)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

        elif isinstance(m, (nn.LayerNorm, nn.GroupNorm)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


    



    def forward(self, 
                x, 
                is_init_image=True,
                temporal_chunk=False):


        if self.time_pad < x.shape[2]:
            pad_mode = self.pad_mode

        else:
            pad_mode = 'constant'


        if not temporal_chunk:
            x = torch.nn.functional.pad(input=x,
                                        pad=self.time_causal_padding,
                                        mode=pad_mode)


        else:
            assert not self.training, "The feature cache should not be used in training."


            if is_init_image:
                # Encode the first chunk.
                x = torch.nn.functional.pad(x, self.time_causal_padding, mode=pad_mode)
                ## <-- context_parallel --> ##
                

                # take the very last 2 frames of the chunk, detach them from the computation graph and store them in the cache.
                self.cache_front_feat.append(x[:, :, -2:].clone().detach())


            else:

                x = torch.nn.functional.pad(input=x,
                                            pad=self.time_uncausal_padding,
                                            mode=pad_mode)
                video_front_context = self.cache_front_feat.pop()
                ## <-- context_parallel --> ##

                if self.temporal_stride == 1 and self.time_kernel_size == 3:
                    x = torch.cat([video_front_context, x], dim=2)
                elif self.temporal_stride == 2 and self.time_kernel_size == 3:
                    x = torch.cat([video_front_context[:, :, -1:], x], dim=2)

                self.cache_front_feat.append(x[:, :, -2:].clone().detach())

            
        x = self.conv(x)
        return x 








        


    




        



if __name__ == "__main__":

    object = CausalConv3d(input_channels=3,
                          output_chaannels=128,
                          kernel_size=3,
                          stride=1)

    