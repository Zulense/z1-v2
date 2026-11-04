import torch
from torch import nn
import torch.nn.functional as F

_PERSISTENT = False

class Patcher(nn.Module):

    def __init__(self, rescale=True):
        super().__init__()
        self.register_buffer(
            "wavelets", 
            torch.tensor([0.7071067811865476, 0.7071067811865476]),
            persistent= _PERSISTENT
        )
        self.register_buffer(
            "_arange",
            torch.arange(2),
            persistent=_PERSISTENT
        )
        self.rescale = rescale
        for param in self.parameters():
            param.requires_grad = False

    def _2ddwt(self, x, mode="reflect", rescale=False):
        """Discrete Wavelet Transform (DWT)"""
        dtype = x.dtype
        h = self.wavelets
        x = x.squeeze(2)

        n = h.shape[0]
        g = x.shape[1]
        hl = h.flip(0).reshape(1, 1, -1).repeat(g, 1, 1).to(dtype)
        hh = (h * ((-1) ** self._arange)).reshape(1, 1, -1).repeat(g, 1, 1).to(dtype)

        x = F.pad(x, pad=(n-2, n-1, n-2, n-1), mode=mode).to(dtype)
        xl = F.conv2d(x, hl.unsqueeze(2), groups=g, stride=(1, 2))
        xh = F.conv2d(x, hh.unsqueeze(2), groups=g, stride=(1, 2))
        xll = F.conv2d(xl, hl.unsqueeze(3), groups=g, stride=(2, 1))
        xlh = F.conv2d(xl, hh.unsqueeze(3), groups=g, stride=(2, 1))
        xhl = F.conv2d(xh, hl.unsqueeze(3), groups=g, stride=(2, 1))
        xhh = F.conv2d(xh, hh.unsqueeze(3), groups=g, stride=(2, 1))

        out = torch.cat([xll, xlh, xhl, xhh], dim=1)
        if rescale:
            out = out * 2 

        return out.unsqueeze(2)


    def _3ddwt(self, x, mode="reflect", rescale=False):
        dtype = x.dtype
        h = self.wavelets

        n = h.shape[0]
        g = x.shape[1]
        hl = h.flip(0).reshape(1, 1, -1).repeat(g, 1, 1).to(dtype)
        hh = (h * ((-1) ** self._arange)).reshape(1, 1, -1).repeat(g, 1, 1).to(dtype)

        # Handle temporal axis.
        x = F.pad(x, pad=(max(0, n-2), n-1, n-2, n-1, n-2, n-1), mode=mode).to(dtype)
        xl = F.conv3d(x, hl.unsqueeze(3).unsqueeze(4), groups=g, stride=(2, 1, 1))
        xh = F.conv3d(x, hh.unsqueeze(3).unsqueeze(4), groups=g, stride=(2, 1, 1))

        # Handles spatial axes.
        xll = F.conv3d(xl, hl.unsqueeze(2).unsqueeze(4), groups=g, stride=(1, 2, 1))
        xlh = F.conv3d(xl, hh.unsqueeze(2).unsqueeze(4), groups=g, stride=(1, 2, 1))
        xhl = F.conv3d(xh, hl.unsqueeze(2).unsqueeze(4), groups=g, stride=(1, 2, 1))
        xhh = F.conv3d(xh, hh.unsqueeze(2).unsqueeze(4), groups=g, stride=(1, 2, 1))

        xlll = F.conv3d(xll, hl.unsqueeze(2).unsqueeze(3), groups=g, stride=(1, 1, 2))
        xllh = F.conv3d(xll, hh.unsqueeze(2).unsqueeze(3), groups=g, stride=(1, 1, 2))
        xlhl = F.conv3d(xlh, hl.unsqueeze(2).unsqueeze(3), groups=g, stride=(1, 1, 2))
        xlhh = F.conv3d(xlh, hh.unsqueeze(2).unsqueeze(3), groups=g, stride=(1, 1, 2))
        xhll = F.conv3d(xhl, hl.unsqueeze(2).unsqueeze(3), groups=g, stride=(1, 1, 2))
        xhlh = F.conv3d(xhl, hh.unsqueeze(2).unsqueeze(3), groups=g, stride=(1, 1, 2))
        xhhl = F.conv3d(xhh, hl.unsqueeze(2).unsqueeze(3), groups=g, stride=(1, 1, 2))
        xhhh = F.conv3d(xhh, hh.unsqueeze(2).unsqueeze(3), groups=g, stride=(1, 1, 2))

        out = torch.cat([xlll, xllh, xlhl, xlhh, xhll, xhlh, xhhl, xhhh], dim=1)
        if rescale:
            out = out * (2 * torch.sqrt(torch.tensor(2.0)))
        return out


    def forward(self, x):
        if x.shape[2] > 1:
            if x.shape[2] % 2 == 1:
                xi, xv = torch.split(x, [1, x.shape[2] - 1], dim=2)
                xi = self._2ddwt(xi, rescale=self.rescale)
                xv = self._3ddwt(xv, rescale=self.rescale)
                return xi, xv
            else:
                xv = self._3ddwt(x, rescale=self.rescale)
                return None, xv

        return (self._2ddwt(x, rescale=self.rescale), None)


    

        



        