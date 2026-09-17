import torch 
from torch import nn 

from .causal_conv import CausalConv3d


class Upsample(nn.Upsample):

    def forward(self, x):
        """Fix bfloat16 support for nearest neighbor interpolation."""

        return super().forward(x.float()).type_as(x)


class Resample(nn.Module):

    def __init__(self,
                 dim,
                 mode):

        assert mode in ('none', 
                        'upsample3d',
                        'downsample3d')
        super().__init__()
        self.dim = dim 
        self.mode = mode 

        if mode == 'downsample3d':
            self.resample = nn.Sequential(
                nn.ZeroPad2d((0, 1, 0, 1)),
                nn.Conv2d(dim, dim, 3, stride=(2, 2))
            )
            self.time_conv = CausalConv3d(
                dim, dim, (3, 1, 1), stride=(2, 1, 1), padding=(0, 0, 0)
            )

        # layers 
        elif mode == 'upsample3d':
            self.resample = nn.Sequential(
                Upsample(scale_factor=(2., 2.),
                            mode='nearest'),
                nn.Conv2d(in_channels=dim,
                            out_channels=dim//2,
                            kernel_size=3,
                            padding=1)
            )
            self.time_conv = CausalConv3d(dim, dim*2, (3, 1, 1), padding=(1, 0, 0))

        else:
            assert ValueError("The `mode` is None.")


    def forward(self, 
                x,
                feat_cache=None,
                feat_idx=[0]):

        B, C, T, H, W = x.size()

        if self.mode == 'downsample3d':
            if feat_cache is not None:
                idx = feat_idx[0]

