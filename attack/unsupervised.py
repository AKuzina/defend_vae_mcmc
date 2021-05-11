import torch
import torch.nn as nn
from tqdm import tqdm


def get_jac_enc(x, vae):
    x.requires_grad = True
    z_mean, z_logvar = vae.q_z(x)
    z_mean = z_mean.squeeze(0)
    jac = [torch.autograd.grad(z, x, retain_graph=True)[0]   for z in z_mean]
    return torch.stack(jac)


def get_opt_perturbation(x_init, vae, eps_norm, reg_type='means', loss_type='penalty'):
    # define loss:
    loss_fn = {
        'mean': lambda J, eps: -torch.norm(J @ eps.reshape(x_dim))
    }[loss_type]

    # initialize
    eps = torch.randn_like(x_init)*0.2
    eps.requires_grad = True
    x_dim = x_init.shape[-1]*x_init.shape[-2]*x_init.shape[-3]
    loss_hist = []
    J = get_jac_enc(x_init, vae)
    J = J.reshape(-1, x_dim)

    # learn
    optimizer = torch.optim.Adam([eps], lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, verbose=False,
                                                           patience=100, factor=0.5)
    loss_hist = []
    for i in range(500):
        optimizer.zero_grad()
        loss = loss_fn(J, eps)

        if reg_type == 'penalty':
            loss = loss + 1./eps_norm * torch.norm(eps)
        loss.backward()
        optimizer.step()
        loss_hist.append(loss.item())
        scheduler.step(loss)
        if reg_type == 'projection':
            eps.data = eps_norm * (eps.data / torch.norm(eps.data))
        if optimizer.param_groups[0]['lr'] < 1e-5:
            print('break after {} iterations'.format(len(loss_hist)))
            break

    return loss_hist, eps.data, eps.data + x_init


def generate_adv(x_init, vae, args, *kwargs):
    x_adv = []
    for _ in tqdm(range(args.N_adv)):
        _, eps, x_opt = get_opt_perturbation(x_init, vae, eps_norm=args.eps_norm,
                                             reg_type=args.reg_type, loss_type=args.loss_type)
        x_adv.append(x_opt.detach().cpu())
    return x_adv
