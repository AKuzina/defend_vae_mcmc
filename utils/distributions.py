import math
import torch


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
    bin_size = 1. / 256.

    # implementation like https://github.com/openai/iaf/blob/master/tf_utils/distributions.py#L28
    scale = torch.exp(logvar)
    x = (torch.floor(x / bin_size) * bin_size - mean) / scale
    cdf_plus = torch.sigmoid(x + bin_size / scale)
    cdf_minus = torch.sigmoid(x)

    # calculate final log-likelihood for an image
    log_logist_256 = torch.log(cdf_plus - cdf_minus + 1.e-7)

    if average:
        return torch.mean(log_logist_256, dim)
    else:
        return torch.sum(log_logist_256, dim)