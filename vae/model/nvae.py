import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
import pytorch_lightning as pl
from itertools import chain


# from vae.model.var_posterior import NormalQ
# from vae.model.vcd import HMC_sampler, VaeTarget


from NVAE.distributions import Normal, DiscMixLogistic
from NVAE.utils import reconstruction_loss

class NVAE(nn.Module):
    def __init__(self, nvae):
        super().__init__()
        self.nvae = nvae
#         self.automatic_optimization = False
#         self.q_0 = NormalQ(hparams.z_dim)
#         self.target = VaeTarget(self.vae.decoder, self.vae.prior, self.vae.log_lik)
#         self.Q_t = HMC_sampler(self.target, 5./hparams.z_dim, L=5, adaptive=True)
#         self.C = torch.zeros(1)

    def forward(self, x, connect=-1, t=1., use_hmc=False, return_samples=False):
        if connect == -1:
            connect = len(self.nvae.dec_tower)

        s = self.nvae.stem(2 * x - 1.0)

        # perform pre-processing
        for cell in self.nvae.pre_process:
            s = cell(s)

        # run the main encoder tower
        combiner_cells_enc = []
        combiner_cells_s = []
        for cell in self.nvae.enc_tower:
            if cell.cell_type == 'combiner_enc':
                combiner_cells_enc.append(cell)
                combiner_cells_s.append(s)
            else:
                s = cell(s)

        # reverse combiner cells and their input for decoder
        combiner_cells_enc.reverse()
        combiner_cells_s.reverse()

        idx_dec = 0
        ftr = self.nvae.enc0(s)                            # this reduces the channel dimension
        param0 = self.nvae.enc_sampler[idx_dec](ftr)
        mu_q, log_sig_q = torch.chunk(param0, 2, dim=1)
        dist = Normal(mu_q, log_sig_q)   # for the first approx. posterior
        z, _ = dist.sample()
        log_q_conv = dist.log_p(z)
        z_samples = [z.data]
        # apply normalizing flows
        nf_offset = 0
        for n in range(self.nvae.num_flows):
            z, log_det = self.nvae.nf_cells[n](z, ftr)
            log_q_conv -= log_det
        nf_offset += self.nvae.num_flows
        all_q = [dist]
        all_q_only = [dist]
        all_log_q = [log_q_conv]


        # To make sure we do not pass any deterministic features from x to decoder.
        s = 0

        # prior for z0
        dist = Normal(mu=torch.zeros_like(z), log_sigma=torch.zeros_like(z))
        log_p_conv = dist.log_p(z)
        all_p = [dist]
        all_log_p = [log_p_conv]

        idx_dec = 0
        s = self.nvae.prior_ftr0.unsqueeze(0)
        batch_size = z.size(0)
        s = s.expand(batch_size, -1, -1, -1)
        for cell in self.nvae.dec_tower:
            if cell.cell_type == 'combiner_dec':
                if idx_dec > 0:
                    # form prior
                    param = self.nvae.dec_sampler[idx_dec - 1](s)
                    mu_p, log_sig_p = torch.chunk(param, 2, dim=1)


                    # form encoder
                    ftr = combiner_cells_enc[idx_dec - 1](combiner_cells_s[idx_dec - 1], s)
                    param = self.nvae.enc_sampler[idx_dec](ftr)
                    mu_q, log_sig_q = torch.chunk(param, 2, dim=1)
                    if len(all_log_q) < connect:
                        dist = Normal(mu_p + mu_q, log_sig_p + log_sig_q) if self.nvae.res_dist else Normal(mu_q, log_sig_q)
                        z, _ = dist.sample()
                        log_q_conv = dist.log_p(z)
                        z_samples.append(z)
                        # apply NF
                        for n in range(self.nvae.num_flows):
                            z, log_det = self.nvae.nf_cells[nf_offset + n](z, ftr)
                            log_q_conv -= log_det
                        nf_offset += self.nvae.num_flows
                        all_log_q.append(log_q_conv)
                        all_q.append(dist)
                        all_q_only.append(Normal(mu_q, log_sig_q))
                    else:
                        dist = Normal(mu_p, log_sig_p, t)
                        z, _ = dist.sample()

                    # evaluate log_p(z)
                    # if len(all_log_q) <= connect:
                    dist = Normal(mu_p, log_sig_p)
                    log_p_conv = dist.log_p(z)
                    all_p.append(dist)
                    all_log_p.append(log_p_conv)

                # 'combiner_dec'
                s = cell(s, z)
                idx_dec += 1
            else:
                s = cell(s)

        if self.nvae.vanilla_vae:
            s = self.nvae.stem_decoder(z)

        for cell in self.nvae.post_process:
            s = cell(s)

        logits = self.nvae.image_conditional(s)

        # compute kl
        kl_all = []
        kl_diag = []
        log_p, log_q = 0., 0.
#         for q, p, log_q_conv, log_p_conv in zip(all_q, all_p, all_log_q, all_log_p):
#             if self.nvae.with_nf:
#                 kl_per_var = log_q_conv - log_p_conv
#             else:
#                 kl_per_var = q.kl(p)

#             kl_diag.append(torch.mean(torch.sum(kl_per_var, dim=[2, 3]), dim=0))
#             kl_all.append(torch.sum(kl_per_var, dim=[1, 2, 3]))
#             log_q += torch.sum(log_q_conv, dim=[1, 2, 3])
#             log_p += torch.sum(log_p_conv, dim=[1, 2, 3])

        if return_samples:
            return z_samples, logits, all_q_only, all_q, all_p, kl_all, kl_diag
        else:
            return logits, all_q_only, all_q, all_p, kl_all, kl_diag
        
    def get_z(self, x, n_connect, t):
        z_samples, logits, q_dist_hist, _, _, _, _ = self.forward(x, connect=n_connect, t=t, return_samples=True)
        MB = x.shape[0]
        z_samples = torch.stack(z_samples, 1)
        return z_samples.reshape(MB, -1)
    
    def decode(self, z, t=1.):
        """
        z: (MB, N_connect, z_dim)
        """
        z_curr = z[:, 0]
        
        # prior for z0
        dist = Normal(mu=torch.zeros_like(z_curr), log_sigma=torch.zeros_like(z_curr))
        log_p_conv = dist.log_p(z_curr)
        all_log_p = [log_p_conv]

        idx_dec = 0
        s = self.nvae.prior_ftr0.unsqueeze(0)
        batch_size = z_curr.size(0)
        s = s.expand(batch_size, -1, -1, -1)
        
        for cell in self.nvae.dec_tower:
            if cell.cell_type == 'combiner_dec':
                if idx_dec > 0:
                    # get nex z sample (from HMC or from p(z_i|z_{>i})
                    param = self.nvae.dec_sampler[idx_dec - 1](s)
                    mu_p, log_sig_p = torch.chunk(param, 2, dim=1)
                    if idx_dec < z.shape[1]:
                        z_curr = z[:, idx_dec]
                    else:
                        dist = Normal(mu_p, log_sig_p, t)
                        z_curr, _ = dist.sample()
                      
                    # eval prior
                    dist = Normal(mu_p, log_sig_p)
                    log_p_conv = dist.log_p(z_curr)
                    all_log_p.append(log_p_conv)
                        
                # 'combiner_dec'
                s = cell(s, z_curr)
                idx_dec += 1
            else:
                s = cell(s)

        for cell in self.nvae.post_process:
            s = cell(s)

        # sum priors:
        log_p_z = 0.
        for log_p_conv in all_log_p:
            log_p_z += torch.sum(log_p_conv, dim=[1, 2, 3])
        
        # cond likelihood
        logits = self.nvae.image_conditional(s)
        return logits, log_p_z
        
        
    def E(self, z, x):
        """
        z: (MB, N_connect, z_dim)
         Out:
             energy E from p = exp(-E(x)) (torch tensor (B,))
             Where p(z|x) = p_{\theta}(x|z)p(z) / p_{\theta}(x)
        """
        logits, log_p_z = self.decode(z)
        output = self.nvae.decoder_output(logits)
        log_pxz = - reconstruction_loss(output, x, crop=self.nvae.crop_output)
        return - log_pxz - log_p_z

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

        