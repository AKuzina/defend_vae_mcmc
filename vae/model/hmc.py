import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from tqdm import tqdm
import pytorch_lightning as pl
from itertools import chain


class VaeTarget:
    def __init__(self, decoder, prior, log_lik):
        self.decoder = decoder
        self.prior = prior
        self.log_lik = log_lik

    def E(self, z, x):
        """
         Out:
             energy E from p = exp(-E(x)) (torch tensor (B,))
             Where p(z|x) = p_{\theta}(x|z)p(z) / p_{\theta}(x)
        """
        x_mean, x_logvar = self.decoder(z)
        MB = x_mean.shape[0]
        log_pxz = self.log_lik(x.reshape(MB, -1), x_mean.reshape(MB, -1),
                               x_logvar.reshape(MB, -1), dim=1)
        log_pz = self.prior.log_prob(z)
        return - log_pxz - log_pz

    def E_ratio(self, z_s, z_p, x):
        """
        z_s: initial z
        z_p: proposed z
        x: data point
        """
        return self.E(z_s, x) - self.E(z_p, x)

    def grad_E(self, z, x):
        """
         Out:
             grad of E from p = exp(-E(x)) (torch tensor (B,D))
        """
        f = False
        if not torch.is_grad_enabled():
            f = True
            torch.set_grad_enabled(True)
        z = z.clone().requires_grad_()
        assert z.grad == None
        E = self.E(z, x)
        E.sum().backward()
        if f:
            torch.set_grad_enabled(False)
        return z.grad.data


class HMC_sampler:
    def __init__(self, target, eps, L=5, adaptive=True, N_vars=1):
        """
        eps: initial step size in leap frog
        L: num of step in leap frog
        adaptive: bool, where to adapt eps during burn in
        N_vars: int, number of variables
        """
        self.eps = eps
        self.L = L
        self.adaptive = adaptive
        self.target = target
        self.N_vars = N_vars

    def update_var(self, curr_val, weight, upd_vals):
        if self.N_vars > 1:
            curr_val = [curr_val[i] + weight * upd_vals[i] for i in range(self.N_vars)]
        else:
            curr_val += weight * upd_vals
        return curr_val

    def transition(self, z_s, x):
        """
        z_s: z start
        x: data point
        """
        if self.N_vars > 1:
            p_s = [torch.randn_like(z_s_curr) for z_s_curr in z_s]
            z_end = [z_curr.clone().detach() for z_curr in z_s]
            p_end = [p_curr.clone().detach() for p_curr in p_s]
        else:
            p_s = torch.randn_like(z_s)
            z_end = z_s.clone().detach()
            p_end = p_s.clone().detach()

        z_grad = self.target.grad_E(z_end, x)
        p_end = self.update_var(p_end, -0.5 * self.eps, z_grad)

        for i in range(self.L):
            z_end = self.update_var(z_end, self.eps, p_end)
            z_grad = self.target.grad_E(z_end, x)
            if i < self.L - 1:
                p_end = self.update_var(p_end, -self.eps, z_grad)
            else:
                p_end = self.update_var(p_end, -0.5*self.eps, z_grad)

        if self.N_vars == 1:
            MB = p_s.shape[0]
            p_s = p_s.reshape(MB, -1).pow(2).sum(1)
            p_end = p_end.reshape(MB, -1).pow(2).sum(1)
        else:
            MB = p_s[0].shape[0]
            p_s = torch.stack([p_curr.reshape(MB, -1).pow(2).sum(1) for p_curr in p_s], 0).sum(0)
            p_end = torch.stack([p_curr.reshape(MB, -1).pow(2).sum(1) for p_curr in p_end], 0).sum(0)

        q_ratio = 0.5 * (p_s - p_end)
        target_ratio = self.target.E_ratio(z_s, z_end, x)
        return z_end, q_ratio + target_ratio

    def sample(self, z_0, x, N, burn_in=1):
        """
        Code is based on
        https://github.com/franrruiz/vcd_divergence/blob/master/demo_main.m and
        https://github.com/evgenii-egorov/sk-bdl/blob/main/seminar_18/notebook/MCMC.ipynb
        z_0: first sample
        x: corresponding data point
        N: number of points in the chain
        burn_in: number of points for burn in
        """
        chain = [z_0]
        acceptance_rate = []
        if self.N_vars > 1:
            print(len(z_0))
            print(z_0[0].shape)
            z_curr = [z.clone() for z in z_0]
            MB = z_0[0].shape[0]
        else:
            z_curr = z_0.clone()
            MB = z_0.shape[0]

        for i in tqdm(range(burn_in + N)):
            # propose point
            z_proposed, ratio = self.transition(z_curr, x)
            # compute MH test
            log_u = torch.log(torch.rand(MB, device=x.device))
            accept_flag = log_u < ratio

            # make a step or stay at the same point
            if self.N_vars > 1:
                for i in range(self.N_vars):
                    z_curr[i][accept_flag] = z_proposed[i][accept_flag]
            else:
                z_curr[accept_flag] = z_proposed[accept_flag]
            # if burn in - adapt step size, else - collect samples
            prop_accepted = torch.mean(accept_flag.float())
            # if i < burn_in:
            if self.adaptive:
                self.eps += 0.01*((prop_accepted - 0.9)/0.9)*self.eps
            if i >= burn_in:
                acceptance_rate.append(prop_accepted)
                chain.append(z_curr)
        return chain[-1], acceptance_rate
