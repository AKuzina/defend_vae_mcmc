import math
import torch
import torch.nn as nn
import torch.distributions as tdist

from utils.distributions import log_Gaus_standard, log_Gaus_diag


class VariationalPosterior(nn.Module):
    def __init__(self, hid_dim):
        super(VariationalPosterior, self).__init__()
        self.hid_dim = hid_dim

    def log_prob(self, z, *kwargs):
        raise NotImplementedError

    def sample_n(self, *kwargs):
        raise NotImplementedError


class NormalQ(VariationalPosterior):
    def __init__(self, hid_dim):
        super(NormalQ, self).__init__(hid_dim)

    def log_prob(self, z, mu, logvar):
        assert z.shape == mu.shape, \
            'Mean and point have different shapes: ' + str(z.shape) + ' ' + str(mu.shape)
        assert z.shape == logvar.shape, \
            'Variance and point have different shapes: ' + str(z.shape) + ' ' + str(logvar.shape)
        assert len(z.shape) == 1, 'Expect 2d point, got ' + str(z.shape) + ' instead'
        return log_Gaus_diag(z, mu, logvar, dim=1)

    def sample_n(self, mu, logvar):
        eps = torch.empty_like(mu, device=mu.device).normal_()
        sigma = (0.5*logvar).exp()
        return mu + sigma*eps

    def entropy(self, logvar):
        return 0.5 * torch.sum(1 + math.log(math.pi*2) + logvar, 1)