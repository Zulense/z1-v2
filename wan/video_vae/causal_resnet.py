import torch 
from torch import nn 
from einops import rearrange

from .causal_conv import CausalConv3d


CACHE_T = 2


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
                # Adds a tiny 1-pixel blank border to the `bottom` and `right` of our images so the math devides perfectly.
                nn.ZeroPad2d((0, 1, 0, 1)),
                nn.Conv2d(in_channels=dim, 
                          out_channels=dim, 
                          kernel_size=3, 
                          stride=(2, 2)  # -> means the filter skips every other pixel vertically and horizontally. This directly cuts the `Height` and `Width` of the Video frame in half
                          )
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
                if feat_idx[idx] is None:
                    feat_idx[idx] = "Rep"
                    feat_idx[0] += 1
                else:
                    cache_x = x[:, :, -CACHE_T:, :, :].clone()
                    if cache_x.shape[2] < 2 and feat_cache[
                            idx] is not None and feat_cache[idx] != 'Rep':
                        # cache last frame of last two chunk.
                        cache_x = torch.cat([
                            feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(
                                cache_x.device), cache_x
                        ], dim=2)

                    if cache_x.shape[2] < 2 and feat_cache[
                        idx] is not None and feat_cache[idx] == 'Rep':
                        cache_x = torch.cat([
                            torch.zeros_like(cache_x).to(cache_x.device),
                            cache_x
                        ], dim=2)

                    if feat_cache[idx] == 'Rep':
                        x = self.time_conv(x)
                    else:
                        x = self.time_conv(x, feat_cache[idx])
                    feat_cache[idx] = cache_x
                    feat_idx[0] += 1 

                    x = x.reshape(B, 2, C, T, H, W)
                    x = torch.stack((x[:, 0, :, :, :, :], x[:, 1, :, :, :, :]),
                                    dim=3)
                    x = x.reshape(B, C, T*2, H, W)

        t = x.shape[2]
        x = rearrange(x, 
                      'b c t h w -> (b t) c h w')
        x = self.resample(x)
        x = rearrange(x, 
                      '(b t) c h w -> b c t h w', t=t)


