import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from utils.divergence import gaus_skl, gaus_kl


class AttackVAE:
    def __init__(self, vae, attack_args):
        self.type = attack_args.type
        self.loss_type = attack_args.loss_type
        self.p = attack_args.p
        self.eps_norm = attack_args.eps_norm
        self.lr = attack_args.lr
        self.loss_fn = self.get_loss_fun()
        self.vae = vae

    def get_loss_fun(self):
        if self.type == 'supervised':
            loss_fn = {
                'skl': lambda m_a, logv_a, m_t, logv_t: gaus_skl(m_a, logv_a, m_t, logv_t, dim=None).sum(1),
                'kl_forward': lambda m_a, logv_a, m_t, logv_t: gaus_kl(m_t, logv_t, m_a, logv_a, dim=None).sum(1),
                'kl_reverse': lambda m_a, logv_a, m_t, logv_t: gaus_kl(m_a, logv_a, m_t, logv_t, dim=None).sum(1),
                'means': lambda m_a, logv_a, m_t, logv_t: (m_a - m_t).pow(2).sum(1),
            }[self.loss_type]
        else:
            loss_fn = {
                'skl': lambda m_a, logv_a, m_r, logv_r: -gaus_skl(m_a, logv_a, m_r, logv_r, dim=None).sum(1),
                'kl_forward': lambda m_a, logv_a, m_r, logv_r: -gaus_kl(m_r, logv_r, m_a, logv_a, dim=None).sum(1),
                'kl_reverse': lambda m_a, logv_a, m_r, logv_r: -gaus_kl(m_a, logv_a, m_r, logv_r, dim=None).sum(1),
                'means': lambda m_a, logv_a, m_r, logv_r: -(m_a - m_r).pow(2).sum(1),
            }[self.loss_type]
        return loss_fn

    def compute_loss(self, x_adv, z_info):
        z_m, z_logv = z_info
        q_m, q_logv = self.vae.q_z(x_adv)
        loss = self.loss_fn(q_m, q_logv, z_m, z_logv)
        return loss

    def get_z_info(self, x):
        """
        Computes all info about reference / target point, required for an attack
        """
        return self.vae.q_z(x)

    def train_eps(self, x_ref, z_info):
        eps = nn.Parameter(torch.zeros_like(x_ref, device='cuda:0'), requires_grad=True)
        optimizer = torch.optim.SGD([eps], lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, verbose=False,
                                                               patience=20, factor=0.5)
        for i in range(2000):
            eps.data = torch.clamp(x_ref + eps.data, 0, 1) - x_ref
            optimizer.zero_grad()
            x = x_ref + eps
            loss = self.compute_loss(x, z_info)
            loss.backward()
            eps.grad.data = torch.clamp(eps.grad.data, -self.eps_norm, self.eps_norm)
            optimizer.step()
            scheduler.step(loss)
            if self.p == 'inf':
                eps.data = torch.clamp(eps.data, -self.eps_norm, self.eps_norm)
            elif torch.norm(eps.data, p=int(self.p)) > self.eps_norm:
                normalizer = self.eps_norm / torch.norm(eps.data, p=int(self.p))
                eps.data = eps.data * normalizer
            if optimizer.param_groups[0]['lr'] < 1e-6:
                print('break after {} iterations'.format(i))
                break
        return torch.clamp(x_ref + eps, 0, 1)

    def get_attack(self, x_ref, all_trg=None):
        if self.type == 'unsupervised':
            x_loop = x_ref
        else:
            x_loop = all_trg
        x_adv = []
        for x in tqdm(x_loop):
            with torch.no_grad():
                z_info = self.get_z_info(x.unsqueeze(0))
            xa = self.train_eps(x_ref, z_info)
            x_adv.append(xa.detach())
        return x_adv


class AttackNVAE(AttackVAE):
    def __init__(self, vae, attack_args):
        super(AttackNVAE, self).__init__(vae, attack_args)

    def get_loss_fun(self):
        if self.type == 'supervised':
            loss_fn = {
                'skl': lambda q_a, q_t: self.q_kls(q_a, q_t, 'skl'),
                'kl_forward': lambda q_a, q_t: self.q_kls(q_t, q_a, 'kl'),
                'kl_reverse': lambda q_a, q_t: self.q_kls(q_a, q_t, 'kl'),
                # 'means': lambda q_a, q_t: (m_a - m_t).pow(2).sum(1),
            }[self.loss_type]
        else:
            loss_fn = {
                'skl': lambda q_a, q_t: -self.q_kls(q_a, q_t, 'skl'),
                'kl_forward': lambda q_a, q_t: -self.q_kls(q_t, q_a, 'kl'),
                'kl_reverse': lambda q_a, q_t: -self.q_kls(q_a, q_t, 'kl'),
                # 'means': lambda q_a, q_t: -(m_a - m_r).pow(2).sum(1),
            }[self.loss_type]
        return loss_fn

    def q_kls(self, q_1, q_2, mode='skl'):
        kl = 0.
        for i in range(self.vae.n_connect):
            if mode == 'skl':
                kl += 0.5*(q_1[i].kl(q_2[i]) + q_2[i].kl(q_2[i])).sum()
            elif mode == 'kl':
                kl += q_1[i].kl(q_2[i]).sum()
        return kl

    def compute_loss(self, x_adv, q_s):
        q_adv, z_s = self.vae.get_q(x_adv)
        loss = self.loss_fn(q_adv, q_s)
        return loss

    def get_z_info(self, x):
        q_s, _ = self.vae.get_q(x)
        return q_s

    # def get_attack(self, x_ref, all_trg=None):
    #     if self.type == 'unsupervised':
    #         x_loop = x_ref
    #     else:
    #         x_loop = all_trg
    #     x_adv = []
    #     for x in tqdm(x_loop):
    #         with torch.no_grad():
    #             q_s, _ = self.vae.get_q(x.unsqueeze(0))
    #         xa = self.train_eps(q_s, x_ref)
    #         x_adv.append(xa.detach())
    #     return x_adv