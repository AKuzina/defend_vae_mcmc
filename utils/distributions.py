import math
import torch
import torch.nn.functional as F
import numpy as np

def log_Gaus_diag(x, mean, log_var, dim=None, average=False):
    log_normal = -0.5 * (math.log(2.0*math.pi) + log_var + torch.pow(x - mean, 2) / (torch.exp(log_var))+1e-5)
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


def log_Logistic_256(x, mean, logvar, dim=None, average=False):
    """
    Logistic LogLikelihood
    :param x:
    :param mean:
    :param logvar:
    :param dim:
    :return:
    """
    # check input and map to -1, 1
    assert torch.max(x) <= 1.0 and torch.min(x) >= 0.0
    x = 2 * x - 1.0

    centered = x - mean                                       # B, 3, H, W
    inv_stdv = torch.exp(- logvar)
    plus_in = inv_stdv * (centered + 1. / 255.)
    cdf_plus = torch.sigmoid(plus_in)
    min_in = inv_stdv * (centered - 1. / 255.)
    cdf_min = torch.sigmoid(min_in)
    log_cdf_plus = plus_in - F.softplus(plus_in)
    log_one_minus_cdf_min = - F.softplus(min_in)
    cdf_delta = cdf_plus - cdf_min
    mid_in = inv_stdv * centered
    log_pdf_mid = mid_in - logvar - 2. * F.softplus(mid_in)

    log_prob_mid_safe = torch.where(cdf_delta > 1e-5,
                                    torch.log(torch.clamp(cdf_delta, min=1e-10)),
                                    log_pdf_mid - math.log(127.5))
    log_logist_256 = torch.where(x < -0.999, log_cdf_plus,
                                 torch.where(x > 0.99, log_one_minus_cdf_min,
                                             log_prob_mid_safe))   # B, 3, H, W

    # bin_num = 256.
    # scale = torch.exp(logvar)
    #
    # x = (torch.floor(x * bin_num) / bin_num - mean) / scale
    # cdf_plus = torch.sigmoid(x + (1/bin_num) / scale)
    # cdf_minus = torch.sigmoid(x)
    #
    # # calculate final log-likelihood for an image
    # log_logist_256 = torch.log(cdf_plus - cdf_minus + 1e-10)
    if average:
        return torch.mean(log_logist_256, dim)
    else:
        return torch.sum(log_logist_256, dim)