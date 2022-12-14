import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride=1, padding=0, dilation=1,
                 act=nn.ReLU(), weight_norm=False, batchnorm=False):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, dilation)
        self.activation = act
        if weight_norm:
            self.conv = nn.utils.weight_norm(self.conv)
        self.bn = None
        if batchnorm:
            self.bn = nn.BatchNorm2d(out_ch, affine=True)

    def forward(self, x):
        out = self.conv(x)
        if self.bn is not None:
            out = self.bn(out)
        if self.activation is not None:
            out = self.activation(out)
        return out


class AffineCoupling1d(nn.Module):
    def __init__(self, dim, hid_dim, mode=1.):
        """
        dim (int) - dimnetion of the input data
        mode - 0 or 1 (from which number the mask starts)
        """
        super(AffineCoupling1d, self).__init__()
        self.mask = torch.arange(mode, mode + dim).unsqueeze(0) % 2
        self.s = nn.Sequential(
            nn.Linear(dim, hid_dim),
            nn.LeakyReLU(),
            nn.Linear(hid_dim, dim),
            nn.Tanh()
        )
        self.t = nn.Sequential(
            nn.Linear(dim, hid_dim),
            nn.LeakyReLU(),
            nn.Linear(hid_dim, dim)
        )

    def forward(self, x):
        """
        Make a forward pass and compute log determinant of the transformation
        """
        self.mask = self.mask.to(x.device)
        bx = self.mask * x
        s_bx = self.s(bx)
        t_bx = self.t(bx)
        z = bx + (1 - self.mask) * (x * torch.exp(s_bx) + t_bx)
        log_det = ((1 - self.mask) * s_bx).sum(1)
        return z, log_det

    def inverse(self, z):
        """
        Covert noize to data via inverse transformation
        """
        self.mask = self.mask.to(z.device)
        bz = self.mask * z
        s_bz = self.s(bz)
        t_bz = self.t(bz)
        x = bz + (1 - self.mask) * ((z - t_bz) * torch.exp(-s_bz))
        return x