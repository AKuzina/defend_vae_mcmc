import torch
import torch.nn as nn
from tqdm import tqdm
from IPython.display import clear_output
import matplotlib.pylab as plt

from attack.nvae.utils import VGGPerceptualLoss
from utils.divergence import gaus_skl


def get_opt_perturbation(x_init, x_trg, model, connect=0, lbd=1., use_perp=True, eps_norm=1, reg_type='penalty'):
    eps = nn.Parameter(torch.zeros_like(x_init), requires_grad=True)

    with torch.no_grad():
        _, q_dist_trg, _, _, _, _ = model(x_trg, connect=connect)

    optimizer = torch.optim.SGD([eps], lr=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, verbose=True, patience=10, factor=0.2)
    if use_perp > 0:
        perp_loss = VGGPerceptualLoss(x_init)
        perp_loss.to('cuda')
    loss_hist = []
    # clear_output(wait=True);
    for i in range(1000):
        eps.data = torch.clamp(x_init + eps.data, 0, 1) - x_init
        optimizer.zero_grad()
        x = x_init + eps
        _, q_dist_curr, _, _, _, _ = model(x, connect=connect)
        loss = sup_loss(q_dist_trg, q_dist_curr)
        # compute loss
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
        if optimizer.param_groups[0]["lr"] < 1e-6:
            break

    return loss_hist, eps, torch.clamp(x_init + eps, 0., 1.)


def sup_loss(q_dist_trg, q_dist_curr):
    loss = 0.
    for q_1, q_2 in zip(q_dist_trg, q_dist_curr):
        sym_kl = gaus_skl(q_1.mu, 2*torch.log(q_1.sigma), q_2.mu, 2*torch.log(q_2.sigma))
        loss += sym_kl.sum()
    return loss


def generate_adv(all_trg, x_init, vae, args):
    x_inits = []
    for x_trg in tqdm(all_trg):
        x_trg = x_trg.unsqueeze(0)
        _, eps, x_opt = get_opt_perturbation(x_init, x_trg, vae, connect=args.model.connect,
                                             lbd=args.attack.lbd, use_perp=args.attack.use_perp,
                                             eps_norm=args.attack.eps_norm, reg_type=args.attack.reg_type)
        x_inits.append(x_opt.detach())
    return x_inits


def min_max_scale(x):
    return (x - x.min())/(x.max() - x.min())