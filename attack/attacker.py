import torch
import torch.nn as nn
from torch.nn.functional import cross_entropy
import numpy as np
from tqdm import tqdm
from utils.divergence import gaus_skl, gaus_kl
import wandb


def rel_norm(x, p):
    return torch.norm(x, p=p) / torch.norm(torch.ones_like(x), p=p)


class AttackVAE:
    def __init__(self, vae, attack_args):
        self.type = attack_args.type
        self.loss_type = attack_args.loss_type
        self.p = attack_args.p
        self.eps_norm = attack_args.eps_norm
        self.lr = attack_args.lr
        self.loss_fn = self.get_loss_fun()
        self.vae = vae
        self.hmc_steps = attack_args.hmc_steps_attack
        self.hmc_eps = attack_args.hmc_eps

    def get_loss_fun(self):
        if self.type == 'supervised':
            loss_fn = {
                'skl': lambda m_a, logv_a, y_a, m_t, logv_t, y_t: gaus_skl(m_a, logv_a, m_t, logv_t, dim=None).sum(1),
                'kl_forward': lambda m_a, logv_a, y_a, m_t, logv_t, y_t: gaus_kl(m_t, logv_t, m_a, logv_a, dim=None).sum(1),
                'kl_reverse': lambda m_a, logv_a, y_a, m_t, logv_t, y_t: gaus_kl(m_a, logv_a, m_t, logv_t, dim=None).sum(1),
                'means': lambda m_a, logv_a, y_a, m_t, logv_t, y_t: (m_a - m_t).pow(2).sum(1),
                'clf': lambda m_a, logv_a, y_a, m_t, logv_t, y_t: cross_entropy(y_a, y_t),
            }[self.loss_type]
        else:
            loss_fn = {
                'skl': lambda m_a, logv_a, y_a, m_r, logv_r, y_r: -gaus_skl(m_a, logv_a, m_r, logv_r, dim=None).sum(1),
                'kl_forward': lambda m_a, logv_a, y_a, m_r, logv_r, y_r: -gaus_kl(m_r, logv_r, m_a, logv_a, dim=None).sum(1),
                'kl_reverse': lambda m_a, logv_a, y_a, m_r, logv_r, y_r: -gaus_kl(m_a, logv_a, m_r, logv_r, dim=None).sum(1),
                'means': lambda m_a, logv_a, y_a, m_r, logv_r, y_r: -(m_a - m_r).pow(2).sum(1),
                'clf': lambda m_a, logv_a, y_a, m_r, logv_r, y_r: -cross_entropy(y_a, y_r),
            }[self.loss_type]
        return loss_fn

    def compute_loss(self, x_adv, z_info, task):
        z_m, z_logv, y_class = z_info
        if task is not None:
            y_class = y_class.argmax(1)
        z_a_m, z_a_logv, y_a_pred = self.get_z_info(x_adv, task, adv=True)
        loss = self.loss_fn(z_a_m, z_a_logv, y_a_pred, z_m, z_logv, y_class)
        return loss

    def get_z_info(self, x, task, adv=False):
        """
        Computes all info about reference / target point, required for an attack
        """
        q_m, q_logvar = self.vae.vae.q_z(x)
        if self.hmc_steps > 0 and adv:
            q_m, _ = self.vae.vae.sample_posterior(q_m, x, self.hmc_steps, self.hmc_eps)
        y_pred = 0.
        if task is not None:
            y_pred = self.vae.classifier[task](q_m)
        return (q_m, q_logvar, y_pred)

    def train_eps(self, x_ref, z_info, task):
        eps = nn.Parameter(torch.zeros_like(x_ref, device='cuda:0'), requires_grad=True)
        if self.type == 'unsupervised':
            eps.data += torch.randn_like(eps) * 0.2
            max_iter = 50
        else:
            max_iter = 500
        optimizer = torch.optim.SGD([eps], lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, verbose=False,
                                                               patience=20, factor=0.5)
        mi, ma = x_ref.min(), x_ref.max()
        for i in range(max_iter):
            eps.data = torch.clamp(x_ref + eps.data, mi, ma) - x_ref
            optimizer.zero_grad()
            x = x_ref + eps
            loss = self.compute_loss(x, z_info, task)
            loss.backward()
            optimizer.step()
            scheduler.step(loss)
            if i == 0:
                loss_0 = loss.data
            if self.p == 'inf':
                eps.data = torch.clamp(eps.data, -self.eps_norm, self.eps_norm)
            elif torch.norm(eps.data, p=int(self.p)) > self.eps_norm:
                normalizer = self.eps_norm / rel_norm(eps.data, p=int(self.p))
                eps.data = eps.data * normalizer
            if optimizer.param_groups[0]['lr'] < 1e-6:
                print('break after {} iterations'.format(i))
                break
        wandb.log({'loss': loss_0 - loss.data})
        return torch.clamp(x_ref + eps, mi, ma)

    def get_attack(self, x_ref, all_trg=None, task=None):
        if self.type == 'unsupervised':
            # 10 random 'restart' for each point
            x_loop = x_ref.repeat(10, 1, 1, 1)
        else:
            x_loop = all_trg
        x_adv = []
        for x in tqdm(x_loop):
            with torch.no_grad():
                z_info = self.get_z_info(x.unsqueeze(0), task, adv=False)
            xa = self.train_eps(x_ref, z_info, task)
            x_adv.append(xa.detach())
        return x_adv
