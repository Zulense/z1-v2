import torch 
from torch import nn
from einops import rearrange
import torch.nn.functional as F
from einops.layers.torch import Rearrange
import numpy as np


class PEG3D(nn.Module):
    """
    Its job is to help the model understand local 3D neighborhoods (across time, height, and width) 
    while keeping computation lightweight, preventing the model from "seeing into the future," 
    and allowing long videos to be processed in smaller memory-friendly chunks without visual glitches.
    """
    def __init__(self, dim):
        super().__init__()
        self.ds_conv = nn.Conv3d(in_channels=dim,
                                 out_channels=dim,
                                 kernel_size=(3, 3, 3),
                                 groups=dim)
        self.is_first_chunk = True
        self.causal_cached = None
        self.tile_inference = False

    def forward(self, x):
        x = rearrange(x, 'b t h w d -> b d t h w')
        if self.tile_inference:
            if self.is_first_chunk:
                x = F.pad(x, (1, 1, 1, 1, 2, 0), value=0.)
            else:
                x = F.pad(x, (1, 1, 1, 1, 0, 0), value=0.)
                x = torch.concatenate((self.causal_cached, x), dim=2)

            self.causal_cached = x[:, :, -2:].clone()

        else:
            x = F.pad(x, (1, 1, 1, 1, 2, 0), value=0.)
        x = self.ds_conv(x.contiguous())
        x = rearrange(x, 'b d t h w -> b t h w d')
        return x


class GEGLU(nn.Module):
    """
    GEGLU (Gated Linear Unit with GELU) acts as a multiplicative filter 
    where one half of your feature channels `dictates` how much of the other half is allowed to `pass` through.
    """
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        # The network applies the non-linear Gaussian Error Linear Unit (GELU) exclusively to the gate tensor
        # The model actively learns to use the gate vector to scale, modulate, or suppress the signal in the x vector.
        return F.gelu(gate) * x


def ffd(dim, mult=4, dropout=0.):
    """
    this block immediately follows the depthwise 3D convolution (PEG3D). While the convolution looks at neighboring pixels across space and time without mixing channels, 
    ffd does the exact opposite: it looks at one specific pixel and mixes all of its channels together to build higher-level representations.
    """
    inner_dim = int(mult * (2 / 3) * dim)
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, inner_dim * 2, bias=False),
        GEGLU(),
        nn.Dropout(dropout),
        nn.Linear(inner_dim, dim, bias=False)
    )

class NAF(nn.Module):
    """
    The NAF (Neighborhood-Aware Feedforward) class acts as the assembly line that links the two components you just learned (PEG3D and ffd) into a single, cohesive module. 
    It applies a alternating "1-2 punch" to the video data, repeating this process however many times you specify.
    """
    def __init__(self,
                 num_layer,
                 dim):
        super().__init__()
        self.num_layer = num_layer
        self.deconv_layer = nn.Sequential()
        self.ffd_layer = nn.Sequential()
        for _ in range(num_layer):
            self.ffd_layer.append(ffd(dim, 4))
            self.deconv_layer.append(PEG3D(dim))

    def forward(self, x):
        for i in range(self.num_layer):
            x = self.deconv_layer[i](x)
            x = self.ffd_layer[i](x)
        return x

class ResNAF(nn.Module):
    def __init__(self,
                 num_layer, 
                 dim,
                 ): 
        super(ResNAF, self).__init__()
        self.num_layer = num_layer
        self.dconv_layer = nn.Sequential()
        self.ffd_layer = nn.Sequential()
        for _ in range(num_layer):
            self.ffd_layer.append(ffd(dim, 4)) 
            self.dconv_layer.append(PEG3D(dim))
   
    def forward(self, x):
        for i in range(self.num_layer):
            x = x + self.dconv_layer[i](x)
            x = x + self.ffd_layer[i](x) 
        return x


class Encoder_Arch(nn.Module):

    def __init__(self,
                 l_dim=128,
                 h_dim=384,
                 sep_num_layer=2,
                 fusion_num_layer=4,
                 patch_size=(2, 4, 4),
                 in_channel=3):

        super().__init__()
        self.is_first_chunk = True
        self.tile_inference = False
        self.in_channel = in_channel

        self._build_linear_path(in_channel=in_channel,
                                out_channel_low=l_dim,
                                out_channel_high=h_dim,
                                pt=patch_size[0],
                                ph=patch_size[1],
                                pw=patch_size[2])

        self.low_layer = ResNAF(num_layer=sep_num_layer, dim=l_dim)
        self.high_layer = ResNAF(num_layer=sep_num_layer, dim=h_dim)
        self.fusion_layer = ResNAF(num_layer=fusion_num_layer, dim=l_dim + h_dim)

    def _build_linear_path(self,
                           in_channel=3,
                           out_channel_low=128,
                           out_channel_high=384,
                           pt=2,
                           ph=4,
                           pw=4):
        patch_config = {
            'video_low': (pt, ph, pw),
            'video_high': (pt, ph, pw),
            'image_low': (1, ph, pw),
            'image_high': (1, ph, pw)
        }

        for name, (t, h, w) in patch_config.items():
            if 'low' in name:
                in_dim = in_channel * t * h * w
                out_dim = out_channel_low
            else:
                out_dim = out_channel_high
                in_dim = in_channel * t * h * w * 7 if 'video' in name else in_channel * t * h * w * 3

            proj = nn.Sequential(
                Rearrange(f'b c (nt {t}) (nh {h}) (nw {w}) -> b nt nh nw (c {t} {h} {w})' if 'video' in name else f'b c (nh {h}) (nw {w}) -> b 1 nh nw (c {h} {w})'),
                nn.Linear(in_dim, out_dim)
            )
            self.add_module(f"{name}_proj", proj)


    def _linear_patch(self, x, proj_type):
        low_comp, high_comp = x[:, :self.in_channel], x[:, self.in_channel]
        return getattr(self, f"{proj_type}_low_proj"(low_comp)), getattr(self, f"{proj_type}_high_proj"(high_comp))

    def forward(self, x):
        xi, xv = x 
        xi_low, xi_high = self._linear_patch(xi, 'image')
        xv_low, xv_high = self._linear_patch(xv, 'video')

        low_x = torch.cat([xi_low, xv_low], dim=1)
        high_x = torch.cat([xi_high, xv_high], dim=1)

        high_x = self.high_layer(high_x)
        low_x = self.low_layer(low_x)
        x = torch.cat([low_x, high_x], dim=-1)
        x = self.fusion_layer(x)
        return x

    def encode(self, x):
        xi, xv = x
        if xi is not None and xv is not None:
            xi_low, xi_high = self._linear_patch(xi, 'image')
            xv_low, xv_high = self._linear_patch(xv, 'video')
    
            low_x = torch.cat([xi_low, xv_low], dim=1)
            high_x = torch.cat([xi_high, xv_high], dim=1)

        if xi is not None:
            low_x, high_x = self._linear_patch(xi, 'image')
        if xv is not None:
            low_x, high_x = self._linear_patch(xv, 'video')

        high_x = self.high_layer(high_x)
        low_x = self.low_layer(low_x)
        x = torch.cat([low_x, high_x], dim=-1)
        x = self.fusion_layer(x)
        return x


class Decoder_Arch(nn.Module):
    def __init__(self,
                 l_dim=128,
                 h_dim=384,
                 sep_num_layer=2,
                 fusion_num_layer=4,
                 patch_size=(2, 4, 4),
                 in_channel=3):
        super().__init__()
        self.l_dim = l_dim
        self.is_first_chunk = True
        self.tile_inference = False 

        self._build_linear_unpatch(in_channel=in_channel,
                                   out_channel_low=l_dim,
                                   out_channel_high=h_dim,
                                   pt=patch_size[0],
                                   ph=patch_size[1],
                                   pw=patch_size[2])

        self.low_layer = ResNAF(num_layer=sep_num_layer, dim=l_dim)
        self.high_layer = ResNAF(num_layer=sep_num_layer, dim=h_dim)
        self.fusion_layer = ResNAF(num_layer=fusion_num_layer, dim=l_dim+h_dim)


    def _build_linear_unpatch(self,
                              in_channel=3,
                              out_channel_low=128,
                              out_channel_high=384,
                              pt=2,
                              ph=4,
                              pw=4):

        patch_config = {
            'video_low': (pt, ph, pw),
            'video_high': (pt, ph, pw),
            'image_low': (1, ph, pw),
            'image_high': (1, ph, pw)
        }

        for name, (t, h, w) in patch_config.items():
            if 'low' in name:
                out_dim = in_channel * t * h * w
                in_dim = out_channel_low
            else:
                in_dim = out_channel_high
                out_dim = in_channel * t * h * w * 7 if 'video' in name else in_channel * t * h * w * 3

            proj = nn.Sequential(
                nn.Linear(in_dim, out_dim),
                Rearrange('b nt nh nw (c pt ph pw) -> b c (nt pt) (nh ph) (nw pw)', pt=t, ph=h, pw=w)
            )
            self.add_module(f"{name}_proj", proj)

    def _linear_unpatch(self, x, proj_type):
        low_comp, high_comp = getattr(self, f"{proj_type}_low_proj")(x[0]), getattr(self, f"{proj_type}_high_proj")(x[1])
        return torch.cat([low_comp, high_comp], dim=1)

    def _feature_transform(self, x):
        x = self.fusion_layer(x)
        low_x = self.low_layer(x[:, :, :, :, :self.l_dim])
        high_x = self.high_layer(x[:, :, :, :, self.l_dim:])

        return low_x, high_x

    def decode(self, x, is_image=False):
        low_x, high_x = self._feature_transform(x)

        if is_image:
            xi = self._linear_unpatch(x=(low_x, high_x), proj_type='image')
            return (xi, None)
        else:
            if self.tile_inference and not self.is_first_chunk:
                xv = self._linear_unpatch(x=(low_x, high_x), proj_type='video')
                return (None, xv)
            else:
                xi = self._linear_unpatch(x=(low_x[:, :1], high_x[:, :1]), proj_type='image')
                xv = self._linear_unpatch(x=(low_x[:, 1:], high_x[:, 1:]), proj_type='video')
                return (xi, xv)

    def forward(self, x):
        low_x, high_x = self._feature_transform(x)
        xi = self._linear_unpatch(x=(low_x[:, :1], high_x[:, :1]), proj_type='image')
        xv = self._linear_unpatch(x=(low_x[:, 1:], high_x[:, 1:]), proj_type='video')
        return (xi, xv)

    

class ISTA(nn.Module):
    """
    Iterative Shrinkage-Thresholding Algorithm.
    """
    def __init__(self,
                 points_num=512,
                 out_num=4,
                 iter_num=2,
                 layer_num=2):
        super().__init__()
        phi_init = np.random.normal(loc=0.0, 
                                    scale=(1 / points_num) ** 0.5,
                                    size=(out_num, points_num))
        self.phi = nn.Parameter(torch.from_numpy(phi_init).float(), requires_grad=True)
        self.Q = nn.Parameter(torch.from_numpy(np.transpose(phi_init)).float(), requires_grad=True)
        self.iter_num = iter_num
        self.forward_l = nn.ModuleList()
        self.backward_l = nn.ModuleList()

        for _ in range(self.iter_num):
            self.forward_l.append(NAF(num_layer=layer_num, dim=points_num))
            self.backward_l.append(NAF(num_layer=layer_num, dim=points_num))

        self.weights = nn.ParameterList()  # Learnable step to control how strongly we correct our guess at each step
        self.threshold = nn.ParameterList() # Learnable shrinkage thresholds (initialized to 0.01) to erase small noisy values close to zero.

        for _ in range(self.iter_num):
            self.threshold.append(nn.Parameter(torch.Tensor([0.01]), requires_grad=True))
            self.weights.append(nn.Parameter(torch.tensor(1.), requires_grad=True))

    def sample(self, x):
        # ([2, 5, 32, 32, 512])
        b, t, h, w, d = x.shape 
        # ([10240, 512]) @ ([512, 4]) -> ([10240, 4])
        y = x.view(-1, d) @ self.phi.T
        # ([2, 5, 32, 32, 4])
        return y.view(b, t, h, w, -1)

    def recon(self, y):
        # ([2, 5, 32, 32, 4])
        b, t, h, w, c = y.shape 
        # ([10240, 4])
        y = y.reshape(-1, c)
        # ([10240, 4]) @ ([4, 512]) -> ([10240, 512])
        recon = torch.mm(y, self.Q.t())
        _, d = recon.shape

        for i in range(self.iter_num):
            # ([10240, 512]) - 1.0 * {{([10240, 512])@ ([512, 4]) -> ([10240, 4]) - ([10240, 4])} @ ([4, 512]) -> ([10240, 512])}
            recon_r = recon - self.weights[i] * torch.mm((torch.mm(recon, self.phi.t()) - y), self.phi)
            # ([10240, 512]) -> ([2, 5, 32, 32, 512])
            recon = recon_r.reshape(b, t, h, w, -1)
            recon = self.forward_l[i](recon)
            r"""
                Notice what happened: the two weak noisy numbers (-0.004 and 0.008) were erased to 0.0, 
                while the two strong signals (-0.05 and 0.03) survived (shrunk slightly by 0.01 to -0.04 and 0.02). 
                In Compressed Sensing, this enforces sparsity (Equation 1, ||Ψp||₁), ensuring only strong, meaningful video features are passed to the backward NAF network.   
            """
            # {[0.05, 0.004, 0.008, 0.03] -> [-1, -1, +1, +1],  {[-0.05, -0.004, 0.008, 0.03] -> [0.05, 0.004, 0.008, 0.03] - 0.01 -> [0.04, -0.006, -0.002, 0.02]} -> [0.04, 0.0, 0.0, 0.02]} -> [-0.04, 0.0, 0.0, 0.02]
            recon = torch.mul(torch.sign(recon), F.relu(torch.abs(recon) - self.threshold[i]))
            # ([2, 5, 32, 32, 512])~[-0.04, 0.0, 0.0, 0.02, ...] -> ([10240, 512])
            recon = self.backward_l[i](recon).view(-1, d)
            # ([10240, 512]) + ([10240, 512])
            recon = recon_r + recon
        # ([2, 5, 32, 32, 512])
        return recon.view(b, t, h, w, -1)


    def forward(self, x):
        # ([2, 5, 32, 32, 512]) -> ([2, 5, 32, 32, 4])
        y = self.sample(x)
        # ([2, 5, 32, 32, 4]) -> ([2, 5, 32, 32, 512])
        recon = self.recon(y)
        return recon
    


    








if __name__ == "__main__":
    NAF_ob = NAF(num_layer=1, dim=512)
    print(NAF_ob)
    x = torch.randn(2, 8, 32, 32, 512)

    out = NAF_ob(x)
    print(out.shape)