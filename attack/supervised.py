import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from utils.divergence import gaus_skl, gaus_kl
from attack.nvae.utils import VGGPerceptualLoss
from vae.model.priors import StandardNormal, RealNPV

def get_opt_perturbation(x_init, x_trg, vae, eps_norm=1., reg_type='penalty', loss_type='skl', lbd=1, use_perp=0.):
    # encode target
    with torch.no_grad():
        z_mean, z_logvar = vae.q_z(x_trg)
    eps = nn.Parameter(torch.zeros_like(x_init, device='cuda:0'), requires_grad=True)

    # define loss:
    loss_fn = {
        'skl': lambda m_a, logv_a, m_t, logv_t: gaus_skl(m_a, logv_a, m_t, logv_t, dim=1),
        'kl_forward': lambda m_a, logv_a, m_t, logv_t: gaus_kl(m_t, logv_t, m_a, logv_a, dim=1),
        'kl_reverse': lambda m_a, logv_a, m_t, logv_t: gaus_kl(m_a, logv_a, m_t, logv_t, dim=1),
        'means': lambda m_a, logv_a, m_t, logv_t: (m_a - m_t).pow(2).sum(1),
    }[loss_type]

    # learn
    lr = 0.01
#     if x_init.shape[-1] == 32:
#         lr = 0.001
    optimizer = torch.optim.SGD([eps], lr=lr)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, verbose=False,
                                                           patience=20, factor=0.5)
    perp_loss = VGGPerceptualLoss(x_init)
    perp_loss.to('cuda')
    loss_hist = []
    for i in range(1000):
        eps.data = torch.clamp(x_init + eps.data, 0, 1) - x_init
        optimizer.zero_grad()
        x = x_init + eps
        q_m, q_logv = vae.q_z(x)
        loss = loss_fn(q_m, q_logv, z_mean, z_logvar)
        if isinstance(vae.prior, RealNPV):
            loss = loss / np.prod(q_m.shape)
        #/ np.prod(eps.shape)
        if reg_type == 'penalty':
            pen_term = torch.clamp_min(torch.norm(eps) - eps_norm, 0.) #/ np.prod(eps.shape)
            loss = loss + lbd * pen_term 
        if use_perp > 0:
            pp = perp_loss(x)
            loss = loss + use_perp*pp
        loss.backward()
        optimizer.step()
        loss_hist.append(loss.item())
        scheduler.step(loss)
        if reg_type == 'projection':
            if torch.norm(eps.data) > eps_norm:
                eps.data = eps_norm * (eps.data / torch.norm(eps.data))
                
        if optimizer.param_groups[0]['lr'] < 1e-6:
            print('break after {} iterations'.format(len(loss_hist)))
            break
    return loss_hist, eps, torch.clamp(x_init + eps, 0, 1)


def generate_adv(all_trg, x_init, vae, args):
    x_adv = []
    for x_trg in tqdm(all_trg):
        x_trg = x_trg.unsqueeze(0)
        _, eps, x_opt = get_opt_perturbation(x_init, x_trg, vae, eps_norm=args.eps_norm,
                                             reg_type=args.reg_type, loss_type=args.loss_type, 
                                             lbd=args.lbd, use_perp=args.use_perp)
        x_adv.append(x_opt.detach())
    return x_adv


def scale(x):
    x_vec = x.reshape(x.shape[0], -1)
    x_min = x_vec.min(dim=1, keepdim=True)[0]
    x_max = x_vec.max(dim=1, keepdim=True)[0]
    res = (x_vec - x_min)/(x_max - x_min)
    return res.reshape(x.shape)

# def get_lr(optimizer):
#     for param_group in optimizer.param_groups:
#         return param_group['lr']