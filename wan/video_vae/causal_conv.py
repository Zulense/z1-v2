import torch 
from torch import nn 
from torch.nn import functional as F


class CausalConv3d(nn.Conv3d):

    def __init__(self, 
                 *args, 
                 **kwargs):

        super().__init__(*args, **kwargs)

        # time_pad, height_pad, width_pad -> [(2* time_pad (padding on past), 0*time_pad (padding on left)), :, :] 
        self._padding = (self.padding[2], self.padding[2], self.padding[1],
                         self.padding[1], 2 * self.padding[0], 0)
        # default system not to add any padding, because we are going to do it manually
        self.padding = (0, 0, 0)


    def forward(self, x, cache_x=None):
        padding = list(self._padding)

        ## <--- Why do we need this? --> ## 
        # High-quality videos are massive. If an AI tries to process a whole 10-second video at once, your computer's memory will crash. 
        # So, Wan-VAE processes the video in smaller chunks.
        # To make sure the current chunk flows smoothly from the previous chunk, we save the last few frames of the previous chunk in `cache_x`
        if cache_x is not None and self._padding[4] > 0:
            cache_x = cache_x.to(x.device)
            # stitches those cached past frames onto the beginning of our current video chunk (dim=2 is the time dimension).
            x = torch.cat([cache_x, x], dim=2)
            # Because we just added real past frames to the front of our video, we subtract that amount from our artificial "blank" padding so the math still aligns perfectly.
            padding[4] -= cache_x.shape[2]

        # adds the blank borders we calculated. Because the future padding is 0, our model remains totally blind to the future.
        x = F.pad(x, padding)

        return super().forward(x)


class RMS_norm(nn.Module):

    def __init__(self,
                 dim,
                 channel_first=True,
                 images=True,
                 bias=False):

        super().__init__()
        broadcastable_dims = (1, 1, 1) if not images else (1, 1)
        shape = (dim, *broadcastable_dims) if channel_first else (dim,)

        self.channel_first = channel_first
        self.scale = dim ** 0.5 
        self.gamma = nn.Parameter(torch.ones(shape))
        self.bias = nn.Parameter(torch.zeros(shape)) if bias else 0 


    def forward(self, x):
        return F.normalize(
            input=x,
            dim=(1 if self.channel_first else
                 -1)) * self.scale * self.gamma + self.bias


    

    