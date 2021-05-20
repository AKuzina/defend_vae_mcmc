import torch
import torch.nn as nn
from tqdm import tqdm
from utils.divergence import gaus_skl, gaus_kl


def get_jac_enc(x, vae, mean=True, logvar=False):
    x.requires_grad = True
    z_mean, z_logvar = vae.q_z(x)
    z_mean = z_mean.squeeze(0)
    z_logvar = z_logvar.squeeze(0)
    jac_mean = None
    if mean:
        jac_mean = torch.stack([torch.autograd.grad(z, x, retain_graph=True)[0] for z in z_mean])
    jac_logvar = None
    if logvar:
        jac_logvar = torch.stack([torch.autograd.grad(z, x, retain_graph=True)[0] for z in z_logvar])

    return jac_mean, jac_logvar


def reverse_kl(J_mean, J_logvar, z_logvar, eps):
    x_dim = eps.shape[-1]*eps.shape[-2]*eps.shape[-3]
    z_dim = z_logvar.shape[-1]
    logvar_diff = J_logvar @ eps.reshape(x_dim)
    mean_diff = J_mean @ eps.reshape(x_dim)

    kl = -logvar_diff.sum() + torch.exp(logvar_diff).sum()
    kl = kl + torch.sum(torch.exp(-z_logvar) * (mean_diff**2))
    return 0.5*(kl - z_dim)


def get_opt_perturbation(x_init, vae, eps_norm, reg_type='means', loss_type='penalty'):
    # define loss:
    loss_fn = {
        # 'means': lambda J_mean, J_logvar, z_logvar, eps: -torch.norm(J_mean @ eps.reshape(x_dim))**2,
        # 'kl_reverse': lambda J_mean, J_logvar, z_logvar, eps: -reverse_kl(J_mean.data, J_logvar.data, z_logvar.data, eps),
        'skl': lambda m_a, logv_a, m_t, logv_t: -gaus_skl(m_a, logv_a, m_t, logv_t, dim=1),
        'kl_forward': lambda m_a, logv_a, m_t, logv_t: -gaus_kl(m_t, logv_t, m_a, logv_a, dim=1),
        'kl_reverse': lambda m_a, logv_a, m_t, logv_t: -gaus_kl(m_a, logv_a, m_t, logv_t, dim=1),
        'means': lambda m_a, logv_a, m_t, logv_t: -(m_a - m_t).pow(2).sum(1),
    }[loss_type]

    # initialize
    eps = torch.randn_like(x_init)*0.4
    eps.requires_grad = True
    # x_dim = x_init.shape[-1]*x_init.shape[-2]*x_init.shape[-3]
    # J_mean, J_logvar = get_jac_enc(x_init, vae, True, True)
    # J_mean = J_mean.reshape(-1, x_dim)
    # J_logvar = J_logvar.reshape(-1, x_dim)
    # encode куаукутсу
    with torch.no_grad():
        z_mean, z_logvar = vae.q_z(x_init)


    # learn
    optimizer = torch.optim.SGD([eps], lr=.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, verbose=False,
                                                           patience=20, factor=0.5)
    loss_hist = []
    for i in range(1000):
        eps.data = torch.clamp(x_init + eps.data, 0, 1) - x_init
        optimizer.zero_grad()
        x = x_init + eps
        if reg_type == 'penalty':
            x = torch.clamp(x, 0, 1)
        q_m, q_logv = vae.q_z(x)
        loss = loss_fn(q_m, q_logv, z_mean, z_logvar)
        # loss = loss_fn(J_mean, J_logvar, z_logvar, eps)
        if reg_type == 'penalty':
            loss = loss + 50*torch.clamp_min(torch.norm(eps) - eps_norm, 0.)
        loss.backward()
        optimizer.step()
        loss_hist.append(loss.item())
        scheduler.step(loss)
        if reg_type == 'projection':
            if torch.norm(eps.data) > eps_norm:
                eps.data = eps_norm * (eps.data / torch.norm(eps.data))
        if optimizer.param_groups[0]['lr'] < 1e-6:
            # print('break after {} iterations'.format(len(loss_hist)))
            break
    return loss_hist, eps.data, torch.clamp(x_init + eps.data, 0, 1)


def generate_adv(x_init, vae, args, **kwargs):
    x_adv = []
    for _ in tqdm(range(args.N_adv)):
        _, eps, x_opt = get_opt_perturbation(x_init, vae, eps_norm=args.eps_norm,
                                             reg_type=args.reg_type, loss_type=args.loss_type)
        x_adv.append(x_opt.detach().cpu())
    return x_adv
