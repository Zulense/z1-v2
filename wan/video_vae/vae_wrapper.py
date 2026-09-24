import torch 
from torch import nn 

from .vae import WanVAE_


def _video_vae(z_dim=None,
               device='cpu',
               **kwargs):

    # params 
    cfg = dict(
        dim=96,
        z_dim=z_dim,
        dim_mult=[1, 2, 4, 4],
        num_res_blocks=2,
        attn_scales=[],
        temporal_downsample=[False, True, True],
        dropout=0.0
    )
    cfg.update(**kwargs)

    # init model 
    with torch.device('meta'):
        model = WanVAE_(**cfg)

    return model



class Wan2_1_VAE:

    def __init__(self,
                 z_dim=16,
                 dtype=torch.float,
                 device="cuda"):

        self.dtype = dtype
        self.device = device

        mean = [
            -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
            0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921
        ]
        std = [
            2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
            3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160
        ]

        self.mean = torch.tensor(mean, dtype=dtype, device=device)
        self.std = torch.tensor(std, dtype=dtype, device=device)
        self.scale = [self.mean, 1.0 / self.std]

        # init model 
        self.model = _video_vae(
            z_dim=z_dim
        ).eval().requires_grad_(True).to(device)


    def encode(self, videos):

        with torch.amp.autocast(device_type="cuda",
                                dtype=self.dtype):

            return [
                self.model.encode(x=u.unsqueeze(0), 
                                  scale=self.scale) for u in videos
            ]

    def decode(self, zs):

        with torch.amp.autocast(device_type="cuda",
                                dtype=self.dtype):

            return [
                self.model.decode(z=u.unsqueeze(0),
                                  scale=self.scale).float().clamp_(-1, 1).squeeze(0) for u in zs
            ]