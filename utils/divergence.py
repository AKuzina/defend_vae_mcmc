import torch


def kl(q_1, q_2, N=1):
    # sample q_1
    x = q_1.sample(N)
    # kl
    kl = q_1.log_prob(x) - q_2.log_prob(x)
    return kl.mean(0)


def sym_kl(q_1, q_2, N=1):
    return 0.5*kl(q_1, q_2, N) + 0.5*kl(q_2, q_1, N)


def gaus_kl(q_mu, q_logsigmasq, p_mu, p_logsigmasq, dim=1):
    """
    Compute KL-divergence KL(q || p) between n pairs of Gaussians
    with diagonal covariational matrices.
    Do not divide KL-divergence by the dimensionality of the latent space.

    Input: q_mu, p_mu, Tensor of shape n x d - mean vectors for n Gaussians.
    Input: q_sigma, p_sigma, Tensor of shape n x d - standard deviation
           vectors for n Gaussians.
    Return: Tensor of shape n - each component is KL-divergence between
            a corresponding pair of Gaussians.
    """
    res = p_logsigmasq - q_logsigmasq - 1 + torch.exp(q_logsigmasq - p_logsigmasq)
    res = res + (q_mu - p_mu).pow(2) * torch.exp(-p_logsigmasq)
    if dim is not None:
        return 0.5 * res.sum(dim=dim)
    else:
        return 0.5 * res


def gaus_skl(q_mu, q_logsigmasq, p_mu, p_logsigmasq, dim=1):
    """
    Compute symmetric KL-divergence 0.5*KL(q || p) + 0.5*KL(p || q) between n pairs of Gaussians
    with diagonal covariational matrices.
    """
    logsigma_dif = p_logsigmasq - q_logsigmasq
    mu_dif = (q_mu - p_mu).pow(2)
    res = torch.exp(logsigma_dif) + torch.exp(-logsigma_dif)

    res = 1/4*(res + mu_dif * (torch.exp(-p_logsigmasq) + torch.exp(-q_logsigmasq))) - 1/2
    if dim is not None:
        return res.sum(dim=dim)
    else:
        return res


def bernoulli_kl(q_mu, p_mu, dim=1):
    res = q_mu * (torch.log(q_mu + 1e-5) - torch.log(p_mu + 1e-5))
    res += (1-q_mu) * (torch.log(1-q_mu + 1e-5) - torch.log(1-p_mu + 1e-5))
    if dim is not None:
        return res.sum(dim=dim)
    else:
        return res
