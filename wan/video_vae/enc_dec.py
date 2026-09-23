import torch 
from torch import nn 

from .causal_conv import CausalConv3d, RMS_norm
from .causal_block import ResidualBlock, AttentionBlock
from .causal_resnet import Resample


CACHE_T=2

class Encoder3d(nn.Module):

    def __init__(self,
                 dim=128,
                 z_dim=4,
                 dim_mult=[1, 2, 4, 4],
                 num_res_blocks=2,
                 attn_scales=[],
                 temporal_downsample=[True, True, False],
                 dropout=0.0):

        super().__init__()
        self.dim = dim 
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        self.temporal_downsample = temporal_downsample

        # dimensions 
        dims = [dim * u for u in [1] + dim_mult]
        scale = 1.0 

        # init block 
        self.conv1 = CausalConv3d(3, dims[0], 3, padding=1)

        # downsample blocks 
        downsamples = []
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):

            # residual (+attention) blocks 
            for _ in range(num_res_blocks):
                downsamples.append(ResidualBlock(in_dim=in_dim,
                                                 out_dim=out_dim,
                                                 dropout=dropout))
                if scale in attn_scales:
                    downsamples.append(AttentionBlock(dim=out_dim))
                in_dim = out_dim

            # downsample block 
            if i != len(dim_mult) -1:
                mode = 'downsample3d' if temporal_downsample[i] else None
                downsamples.append(Resample(dim=out_dim,
                                            mode=mode))
                scale /= 2.0 # <--------- I DON'T UNDERSTAND WHAT DOES NEED OF THIS `scale`

        self.downsamples = nn.Sequential(*downsamples)


        # middle blocks 
        self.middle = nn.Sequential(
            ResidualBlock(in_dim=out_dim, out_dim=out_dim, dropout=dropout),
            AttentionBlock(dim=out_dim),
            ResidualBlock(in_dim=out_dim, out_dim=out_dim, dropout=dropout)
        )

        # output blocks 
        self.head = nn.Sequential(
            RMS_norm(dim=out_dim, images=False), nn.SELU(),
            CausalConv3d(out_dim, z_dim, 3, padding=1)
        )


    def forward(self, x, feat_cache=None, feat_idx=[0]):

        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:, :, :].clone()

            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                # cache last frame of last two chunk 
                cache_x = torch.cat([feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x],
                                    dim=2)
                x = self.conv1(x, feat_cache[idx])
                feat_cache[idx] = cache_x
                feat_idx[0] += 1
        else:
            x = self.conv1(x)

        # downsamples
        for layer in self.downsamples:
            if feat_cache is not None:
                x = layer(x, feat_cache, feat_idx)
            else:
                x = layer(x)


        # middle 
        for layer in self.middle:
            if isinstance(layer, ResidualBlock) and feat_cache is not None:
                x = layer(x, feat_cache, feat_idx)
            else:
                x = layer(x)

        # head 
        for layer in self.head:
            if isinstance(layer, CausalConv3d) and feat_cache is not None:
                idx = feat_idx[0]
                cache_x = x[:, :, -CACHE_T:, :, :].clone()

                if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                    # cache last frame of last two chunk 
                    cache_x = torch.cat([
                        feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x
                    ], dim=2)

                x = layer(x, feat_cache[idx])
                feat_cache[idx] = cache_x
                feat_idx[0] += 1 

            else:
                x = layer(x)

        return x
    

            
class Decoder3d(nn.Module):

    def __init__(self,
                 dim=128,
                 z_dim=4,
                 dim_mult=[1, 2, 4, 4],
                 num_res_blocks=2,
                 attn_scales=[],
                 temporal_upsample=[False, True, True],
                 dropout=0.0):
        
        super().__init__()
        self.dim = dim 
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        self.temporal_upsample = temporal_upsample

        # dimensions 
        dims = [dim*u for u in [dim_mult[-1]] + dim_mult[::-1]]
        scale = 1.0 / 2 **(len(dim_mult) -2)

        # init block 
        self.conv1 = CausalConv3d(z_dim, dims[0], 3, padding=1)

        # middle blocks 
        self.middle = nn.Sequential(
            ResidualBlock(dims[0], dims[0], dropout), AttentionBlock(dims[0]),
            ResidualBlock(dims[0], dims[0], dropout)
        )

        # upsample blocks 
        upsamples = []
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):

            # residual (+attention) blocks 
            if i==1 or i==2 or i==3:
                in_dim = in_dim // 2 

            for _ in range(num_res_blocks + 1):
                upsamples.append(ResidualBlock(in_dim=in_dim, out_dim=out_dim, dropout=dropout))
                if scale in attn_scales:
                    upsamples.append(AttentionBlock(dim=out_dim))
                in_dim = out_dim

            # upsample block 
            if i != len(dim_mult) -1:
                mode = 'upsample3d' if temporal_upsample[i] else None
                upsamples.append(Resample(out_dim, mode=mode))
                scale *= 2.0

        self.upsamples = nn.Sequential(*upsamples)

        # output block
        self.head = nn.Sequential(
            RMS_norm(dim=out_dim, images=False), nn.SiLU(),
            CausalConv3d(out_dim, 3, 3, padding=1)
        )

    def forward(self, x, feat_cache=None, feat_idx=[0]):

        # conv1 
        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:, :, :].clone()

            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:

                # cache last frame of last two chunk
                cache_x = torch.cat([feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x], 
                                    dim=2)

            x = self.conv1(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1 

        else:
            x = self.conv1(x)


        # middle 
        for layer in self.middle:
            if isinstance(layer, ResidualBlock) and feat_cache is not None:
                x = layer(x, feat_cache, feat_idx)
            else:
                x = layer(x)


        # upsample 
        for layer in self.upsamples:
            if feat_cache is not None:
                x = layer(x, feat_cache, feat_idx)
            else:
                x = layer(x)

        # head 
        for layer in self.head:
            if isinstance(layer, CausalConv3d) and feat_cache is not None:
                idx = feat_idx[0]
                cache_x = x[:, :, -CACHE_T:, :, :].clone()

                if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                    # cache last frame of last two chunk.
                    cache_x = torch.cat([feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x],
                                        dim=2)

                x = layer(x, feat_cache[idx])
                feat_cache[idx] = cache_x
                feat_idx[0] += 1 

            else:
                x = layer(x)

        return x




