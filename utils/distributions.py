import math
import torch
import torch.nn.functional as F
import numpy as np


def log_Gaus_diag(x, mean, log_var, dim=None, average=False):
    log_normal = -0.5 * (math.log(2.0*math.pi) + log_var + torch.pow(x - mean, 2) / (torch.exp(log_var) + 1e-10))
    if average:
        return torch.mean(log_normal, dim)
    else:
        return torch.sum(log_normal, dim)


def log_Gaus_standard(x, dim=None, average=False):
    return log_Gaus_diag(x, torch.zeros_like(x), torch.zeros_like(x), average, dim)


def log_Bernoulli(x, mean, dim=None, average=False):
    probs = torch.clamp(mean, min=1e-5, max=1.-1e-5)
    log_bernoulli = x * torch.log(probs) + (1. - x) * torch.log(1. - probs)
    if average:
        return torch.mean(log_bernoulli, dim)
    else:
        return torch.sum(log_bernoulli, dim)
