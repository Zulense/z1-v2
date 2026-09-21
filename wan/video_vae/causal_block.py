import torch 
from torch import nn 

from .causal_conv import RMS_norm, CausalConv3d


CACHE_T=2

class ResidualBlock(nn.Module):

    def __init__(self,
                 in_dim,
                 out_dim,
                 dropout=0.0):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        # layers 
        self.residual = nn.Sequential(
            RMS_norm(dim=in_dim, images=False), nn.SiLU(),
            CausalConv3d(in_dim, out_dim, 3, padding=1),
            RMS_norm(dim=out_dim, images=False), nn.SiLU(), nn.Dropout(dropout),
            CausalConv3d(out_dim, out_dim, 3, padding=1)
        )
        self.shortcut = CausalConv3d(in_dim, out_dim, 1) if in_dim != out_dim else nn.Identity()

    def forward(self, 
                x,
                feat_cache=None,
                feat_idx=[0]):

        h = self.shortcut(x)
        for layer in self.residual:
            # if the current layer is a `CausalConv3d` and we are processing the video in chunks,
            # we find out which layer's memory we are on (`idx`), Then we grab the last few frames of the current video chunk (`cache_x`) to save for later.
            if isinstance(layer, CausalConv3d) and feat_cache is not None:
                idx = feat_idx[0]
                cache_x = x[:, :, -CACHE_T:, :, :].clone()

                # This is a safety net. If our current video chunk is super short (less than 2 frames), 
                # we reach back into the previous memory (feat_cache) to grab an older frame and glue them together.
                if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                    # cache last frame of last two chunk.
                    cache_x = torch.cat([
                        feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x
                    ], dim=2)

                # we run the `CausalConv3d` layer, feeding it the video `x` and the memory of the past chunk (`feat_cache`).
                # Then we overwrite the memory with `cache_x` (the end of our current chunk) so it's ready for the next time. 
                # we add `1` to `feat_idx` to move to the next layer's memory slot.
                x = layer(x, feat_cache[idx])
                feat_cache[idx] = cache_x
                feat_idx[0] += 1

            else:
                x = layer(x)

        return x + h

    