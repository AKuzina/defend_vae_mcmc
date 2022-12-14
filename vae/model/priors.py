import torch
import torch.nn as nn
import math

from utils.distributions import log_Gaus_diag
from vae.utils.nn import AffineCoupling1d


class Prior(nn.Module):
    def __init__(self, hid_dim):
        super(Prior, self).__init__()
        self.hid_dim = hid_dim

    def log_prob(self, z):
        raise NotImplementedError

    def sample_n(self, N):
        raise NotImplementedError


class Normal(Prior):
    def __init__(self, hid_dim, mu, log_var):
        super(Normal, self).__init__(hid_dim)
        # self.hid_dim = hid_dim
        self.mu = nn.Parameter(mu, requires_grad=False)
        self.log_var = nn.Parameter(log_var, requires_grad=False)

    def log_prob(self, z):
        return log_Gaus_diag(z, self.mu, self.log_var, dim=1)

    def log_prob_comp(self, z):
        log_p = -0.5 * (math.log(2.0*math.pi) +
                        self.log_var +
                        torch.pow(z - self.mu, 2) / (torch.exp(self.log_var) + 1e-10))
        return log_p

    def sample_n(self, N):
        z_sample = torch.empty(N, self.hid_dim, device=self.mu.device)
        sigma = (0.5*self.log_var).exp()
        eps = z_sample.normal_()
        return self.mu + sigma*eps


class StandardNormal(Normal):
    def __init__(self, hid_dim):
        super(StandardNormal, self).__init__(hid_dim, torch.zeros(hid_dim),
                                             torch.zeros(hid_dim))


class RealNPV(Prior):
    def __init__(self, hid_dim, N_layers=3):
        super(RealNPV, self).__init__(hid_dim)
        layers = [AffineCoupling1d(hid_dim, hid_dim, i % 2) for i in range(N_layers)]
        self.layers = nn.ModuleList(layers)
        self.prior = StandardNormal(hid_dim)

    def forward(self, x):
        """
        -> to Standard normal
        Apply sequence of transformations and perform summation of log determinants
        """
        log_det = 0.
        z = x.clone()
        for f in self.layers:
            z, d = f(z)
            log_det += d
        return z, log_det

    def inverse(self, z):
        """
        <- From Standard Normal
        Apply sequence of transformations in the inverse order
        """
        x = z.clone()
        for f in reversed(self.layers):
            x = f.inverse(x)
        return x

    def log_prob(self, z):
        z, log_det = self.forward(z)
        log_px = self.prior.log_prob(z) + log_det
        return log_px

    def sample_n(self, N):
        z = self.prior.sample_n(N)
        z = self.inverse(z)
        return z
