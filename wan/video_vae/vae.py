import torch 
from torch import nn 

from .enc_dec import Encoder3d, Decoder3d
from .causal_conv import CausalConv3d

class WanVAE_(nn.Module):

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
        self.temporal_upsample = temporal_downsample[::-1]

        # modules 
        self.encoder = Encoder3d(dim=dim, 
                                 z_dim=z_dim*2, 
                                 dim_mult=dim_mult, 
                                 num_res_blocks=num_res_blocks,
                                 attn_scales=attn_scales,
                                 temporal_downsample=self.temporal_downsample,
                                 dropout=dropout)
        self.conv1 = CausalConv3d(z_dim*2, z_dim*2, 1)

        self.conv2 = CausalConv3d(z_dim, z_dim, 1)
        self.decoder = Decoder3d(dim=dim,
                                 z_dim=z_dim,
                                 dim_mult=dim_mult,
                                 num_res_blocks=num_res_blocks,
                                 attn_scales=attn_scales,
                                 temporal_upsample=self.temporal_upsample,
                                 dropout=dropout)


    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var=log_var)

        x_recon = self.decode(z)
        return x_recon, mu, log_var



    def encode(self, x, scale):
        self.clear_cache()

        ## cache 
        t = x.shape[2]
        iter_ = 1 + (t-1) // 4

        ## split the input x for encoding along the time dim into segments of 1, 4, 4, 4
        # एनकोडिंग के लिए इनपुट x को टाइम डाइमेंशन के साथ 1, 4, 4, 4... के सेगमेंट में बांटें।
        for i in range(iter_):
            self._enc_conv_idx = [0]
            if i == 0:
                out = self.encoder(
                    x[:, :, :1, :, :],
                    feat_cache=self._enc_feat_map,
                    feat_idx=self._enc_conv_idx
                )
            else:
                out_ = self.encoder(
                    x[:, :, 1+4*(i-1):1+4*i, :, :],
                    feat_cache=self._enc_feat_map,
                    feat_idx=self._enc_conv_idx
                )
                out = torch.cat([out, out_], 2)

        mu, log_var = self.conv1(out).chunk(2, dim=1)
        if isinstance(scale[0], torch.Tensor):
            mu = (mu - scale[0].view(1, self.z_dim, 1, 1, 1)) * scale[1].view(1, self.z_dim, 1, 1, 1)
        else:
            mu = (mu - scale[0]) * scale[1]

        self.clear_cache()
        return mu

    def decode(self, z, scale):

        self.clear_cache()
        # z: [b, c, t, h, w]
        if isinstance(scale[0], torch.Tensor):
            z = z / scale[1].view(1, self.z_dim, 1, 1, 1) + scale[0].view(1, self.z_dim, 1, 1, 1)
        else:
            z = z / scale[1] + scale[0]

        iter_ = z.shape[2]
        x = self.conv2(z)

        for i in range(iter_):
            self._conv_idx = [0]
            if i == 0:
                out = self.decoder(
                    x[:, :, i:i+1, :, :],
                    feat_cache=self._feat_map,
                    feat_idx=self._conv_idx
                )
            else:
                out_ = self.decoder(
                    x[:, :, i:i+1, :, :],
                    feat_cahe=self._feat_map,
                    feat_idx=self._conv_idx
                )
                out = torch.cat([out, out_], dim=2)

        self.clear_cache()
        return out


    def clear_cache(self):
        self._conv_num = count_conv3d(self.decoder)
        self._conv_idx = [0]
        self._feat_map = [None] * self._conv_num

        # cache encode 
        self._enc_conv_num = count_conv3d(self.encoder)
        self._enc_conv_idx = [0]
        self._enc_feat_map = [None] * self._enc_conv_num




    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return eps * std + mu



def count_conv3d(model):
    count = 0
    for m in model.modules():
        if isinstance(m, CausalConv3d):
            count += 1 
    return count
    