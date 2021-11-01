import wandb
import math
import os
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from itertools import chain

# from vae.model.var_posterior import NormalQ
# from vae.model.vcd import HMC_sampler, VaeTarget
from NVAE.distributions import Normal, DiscMixLogistic
from NVAE.utils import reconstruction_loss
from vae.model.hmc import HMC_sampler
from thirdparty.pytorch_msssim import msssim


class NF:
    def __init__(self, p_0, f):
        self.p_0 = p_0
        self.f = f

    def sample(self):
        z, _ = self.p_0.sample()
        log_p = self.p_0.log_p(z)
        for f_i in self.f:
            z, log_det = f_i(z)
            log_p -= log_det
        return z, log_p

    def kl(self, nf_2):
        z, _ = self.p_0.sample()
        log_p_1 = self.p_0.log_p(z)
        log_p_2 = nf_2.p_0.log_p(z)
        return log_p_1 - log_p_2

    # def inv_kl(self, nf_2):
    #     z, _ = nf_2.p_0.sample()
    #     log_p_2 = nf_2.p_0.log_p(z)
    #     log_p_1 = self.p_0.log_p(z)
    #     return log_p_2 - log_p_1

    def log_p(self, z):
        log_p = self.p_0.log_p(z)
        for f_i in self.f:
            z, log_det = f_i(z)
            log_p -= log_det
        return log_p


class NVAE(nn.Module):
    def __init__(self, nvae, t=1.0, n_connect=-1):
        """
        n_connect: number to 'skip-connections' to use in forward pas (reconstruction)
            0 - the same as sampling from prior
            1 - use z_0 provided as input - the rest from the prior
            ...
            -1 - make 'full' forward pass

        t: which temperature to use, when sampling from prior
        """
        super().__init__()
        self.nvae = nvae
        self.t = t
        self.n_connect = n_connect
        if self.n_connect == -1:
            self.n_connect = len(self.nvae.dec_tower)

    def encode(self, x):
        """
        Performs *deterministic* forward pass through the encoder
        """
        # maps input to [-1, 1]
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

        # Get first log_q and sample from it
        ftr = self.nvae.enc0(s)
        mu_q_0, log_sig_q_0 = torch.chunk(self.nvae.enc_sampler[0](ftr), 2, dim=1)
        q_0 = NF(Normal(mu_q_0, log_sig_q_0), self.nvae.nf_cells[:self.nvae.num_flows])
        z, log_q = q_0.sample()
        return combiner_cells_enc, combiner_cells_s, z, log_q, q_0

    def decode(self, combiner_cells_enc, combiner_cells_s, z_L=None, q_L=None, n_connect=-1., return_x=True):
        """
        return_x:bool. If False, stop decoding after n_connect layers
        """
        # initialize
        idx_dec = 0
        batch_size = z_L.size(0)
        z = z_L

        # Save distributions
        priors = []
        q_s = [q_L]
        z_samples = [z_L]

        # prior for z_L
        p_L = Normal(mu=torch.zeros_like(z_L), log_sigma=torch.zeros_like(z_L))
        priors.append(p_L)

        s = self.nvae.prior_ftr0.unsqueeze(0).expand(batch_size, -1, -1, -1)
        nf_offset = self.nvae.num_flows
        for cell in self.nvae.dec_tower:
            if cell.cell_type == 'combiner_dec':
                if idx_dec > 0:
                    # form prior
                    param = self.nvae.dec_sampler[idx_dec - 1](s)
                    mu_p, log_sig_p = torch.chunk(param, 2, dim=1)
                    priors.append(Normal(mu_p, log_sig_p))

                    # form posterior
                    if len(q_s) < n_connect:
                        # if we use signal from encoder
                        ftr = combiner_cells_enc[idx_dec - 1](combiner_cells_s[idx_dec - 1], s)
                        param = self.nvae.enc_sampler[idx_dec](ftr)
                        mu_q, log_sig_q = torch.chunk(param, 2, dim=1)
                        dist_0 = Normal(mu_p + mu_q, log_sig_p + log_sig_q) if self.nvae.res_dist else Normal(mu_q, log_sig_q)
                        nf_layers = self.nvae.nf_cells[nf_offset:(nf_offset + self.nvae.num_flows)]
                        q_curr = NF(dist_0, nf_layers)
                        z, log_det = q_curr.sample()
                        nf_offset += self.nvae.num_flows
                    else:
                        # only use signal from decoder
                        q_curr = Normal(mu_p, log_sig_p, self.t)
                        z, _ = q_curr.sample()
                    q_s.append(q_curr)
                    z_samples.append(z)
                    # if we only need q:
                    if len(q_s) >= n_connect and not return_x:
                        return q_s, z_samples
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
        return logits, priors, q_s, z_samples

    def cond_decode(self, z_list):
        """
        Do decoding using z provided as input instead of sample from prior.
        """
        idx_dec = 0
        z = z_list[idx_dec]
        batch_size = z_list[0].size(0)

        # Save distributions
        log_p = []

        # prior for z0
        p_L = Normal(mu=torch.zeros_like(z_list[0]), log_sigma=torch.zeros_like(z_list[0]))
        log_p.append(p_L.log_p(z).sum(dim=[1, 2, 3]))

        s = self.nvae.prior_ftr0.unsqueeze(0).expand(batch_size, -1, -1, -1)
        for cell in self.nvae.dec_tower:
            if cell.cell_type == 'combiner_dec':
                if idx_dec > 0:
                    # form prior
                    param = self.nvae.dec_sampler[idx_dec - 1](s)
                    mu_p, log_sig_p = torch.chunk(param, 2, dim=1)
                    prior = Normal(mu_p, log_sig_p)

                    z = z_list[idx_dec]
                    log_p.append(prior.log_p(z).sum(dim=[1, 2, 3]))

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
        return logits, log_p

    def forward(self, x):
        # ENCODER (deterministic path)
        combiner_cells_enc, combiner_cells_s, z_L, log_q_L, q_L = self.encode(x)
        # DECODER
        logits, priors, q_is, z_samples = self.decode(combiner_cells_enc, combiner_cells_s,
                                                      z_L, q_L, self.n_connect)
        # COND. LIKELIHOOD
        log_pxz = self.nvae.decoder_output(logits)
        if isinstance(log_pxz, DiscMixLogistic):
            x_sample = log_pxz.sample(t=self.t)
        else:
            x_sample = log_pxz.sample()
        return x_sample, priors, log_pxz, q_is, z_samples

    def get_q(self, x):
        # ENCODER (deterministic path)
        combiner_cells_enc, combiner_cells_s, z_L, log_q_L, q_L = self.encode(x)
        # DECODER
        q_is, z_samples = self.decode(combiner_cells_enc, combiner_cells_s,
                                      z_L, q_L, self.n_connect, return_x=False)
        return q_is, z_samples

    def sample(self, num_samples):
        # Sample z_L
        zL_size = [num_samples] + self.nvae.z0_size
        p_L = Normal(mu=torch.zeros(zL_size).cuda(), log_sigma=torch.zeros(zL_size).cuda(), temp=self.t)
        z_L, _ = p_L.sample()
        # DECODER
        logits, _, _, _ = self.decode([], [], z_L, None, 0)
        # COND. LIKELIHOOD
        log_pxz = self.nvae.decoder_output(logits)
        if isinstance(log_pxz, DiscMixLogistic):
            x_sample = log_pxz.sample(t=self.t)
        else:
            x_sample = log_pxz.sample()
        return x_sample

    def sample_posterior(self, z_init, x, n_steps, step_size):
        """
        x - input
        z - list of latent variables
        """
        # target = VaeTarget(self.vae.decoder, self.vae.prior, self.vae.log_lik)
        Q_t = HMC_sampler(self, step_size, L=20, adaptive=True, N_vars=len(z_init))
        z_t, acc = Q_t.sample(z_init, x, n_steps, int(0.5*n_steps))
        logs = {
            'hmc_acc_rate': torch.stack(acc).mean(0).item(),
            'hmc_eps': Q_t.eps
        }
        return z_t, logs

    def eval_attack(self, x_ref, y_ref, x_adv, step, clf_model=None,
                    x_trg=None, hmc_steps=0, **kwargs):
        torch.save(x_adv.cpu(), os.path.join(wandb.run.dir, 'x_adv_{}.pth'.format(step)))
        torch.save(x_ref.cpu(), os.path.join(wandb.run.dir, 'x_ref_{}.pth'.format(step)))

        # get reconstructions
        logs = {}
        with torch.no_grad():
            x_ref_rec, ref_priors, ref_log_pxz, ref_qs, z_ref_sampl = self.forward(x_ref)
            x_adv_rec, adv_priors, adv_log_pxz, adv_qs, z_adv_sampl = self.forward(x_adv)
            adv_dist = (x_adv, x_adv_rec, adv_log_pxz)
            z_adv = (z_adv_sampl, adv_qs)
        if hmc_steps > 0:
            z_adv_t, hmc_logs = self.sample_posterior(z_adv_sampl, x_adv, hmc_steps, kwargs['hmc_eps'])
            logs.update(hmc_logs)
            x_adv_rec_t, log_p = self.cond_decode(z_adv_t)
            z_adv = (z_adv_t, adv_qs)
            adv_dist = (x_adv, x_adv_rec_t, adv_log_pxz)

        ref_logs = self.eval_attack_reference(
            ref_dist=(x_ref, x_ref_rec, ref_log_pxz),
            adv_dist=adv_dist,
            z_ref=(z_ref_sampl, ref_qs),
            z_adv=z_adv
        )
        logs.update(ref_logs)

        # Add supervised-only metrics
        if x_trg is not None:
            trg_logs = self.eval_attack_target(x_trg, adv_dist[1], step)
            logs.update(trg_logs)

        # Add classifier accuracy
        # z = z_adv if hmc_steps == 0 else z_adv_t
        # clf_logs = self.eval_attack_classifier(clf_model, z_ref, y_ref, z)
        # logs.update(clf_logs)
        return logs

    def eval_attack_reference(self, ref_dist, adv_dist, z_ref, z_adv):
        x_ref, x_ref_rec, ref_log_pxz = ref_dist
        x_adv, x_adv_rec, adv_log_pxz = adv_dist
        z_ref_sample, q_ref = z_ref
        z_adv_sample, q_adv = z_adv

        # eps norm
        eps_norms = [torch.norm(x_ref - x_a.unsqueeze(0)).cpu() for x_a in x_adv]

        # msssims
        ref_sim = [msssim(x_ref, x_a.unsqueeze(0), 14, normalize='relu').data.cpu() for x_a in x_adv]
        ref_rec_sim = [msssim(x_ref_rec, x_a.unsqueeze(0), 14, normalize='relu').data.cpu() for x_a in x_adv_rec]

        # latent space
        # mus = [ for i in range(self.n_connect)]
        s_kl = 0.
        mus = 0.
        for i in range(self.n_connect):
            mus += (z_ref_sample[i] - z_adv_sample[i]).pow(2).sum(1).mean()
            s_kl += 0.5*(q_ref[i].kl(q_adv[i]) + q_adv[i].kl(q_ref[i])).mean()

        # log-likelihood
        get_nll = lambda x, log_pxz: - reconstruction_loss(log_pxz, x, crop=self.nvae.crop_output)
        if self.nvae.dataset == 'mnist':
            x_adv_b = torch.bernoulli(x_adv)
            # x_ref = torch.bernoulli(x_ref)
        log_p_xa_zr = get_nll(x_adv_b, ref_log_pxz)
        log_p_xr_zr = get_nll(x_ref, ref_log_pxz)
        log_p_xa_za = get_nll(x_adv_b, adv_log_pxz)
        log_p_xr_za = get_nll(x_ref, adv_log_pxz)

        logs = {
            'Adversarial Inputs': wandb.Image(x_adv.cpu()),
            'Adversarial Rec': wandb.Image(x_adv_rec.cpu()),

            'ref_sim': np.mean(ref_sim),
            'ref_rec_sim': np.mean(ref_rec_sim),
            'eps_norm': np.mean(eps_norms),

            'omega': s_kl.cpu(),
            'z_dist': mus.cpu(),

            '-log_p_xa_zr': log_p_xa_zr.cpu(),
            '-log_p_xr_zr': log_p_xr_zr.cpu(),
            '-log_p_xa_za': log_p_xa_za.cpu(),
            '-log_p_xr_za': log_p_xr_za.cpu(),
        }
        return logs

    def eval_attack_target(self, x_trg, x_adv_rec, step):
        logs_trg = {}
        if step == 0:
            x_trg_rec, _, _, _, _ = self.forward(x_trg)
            self.x_trg_rec = x_trg_rec
            torch.save(x_trg.cpu(), os.path.join(wandb.run.dir, 'x_trg.pth'))
            torch.save(self.x_trg_rec.detach().cpu(), os.path.join(wandb.run.dir, 'x_trg_rec.pth'))
            logs_trg['Target Inputs'] = wandb.Image(x_trg.cpu())
            logs_trg['Target Rec'] = wandb.Image(x_trg_rec.detach().cpu())

        trg_rec_sim = [msssim(self.x_trg_rec[i:i+1], x_adv_rec[i:i+1], 14, normalize='relu').data.cpu()
                       for i in range(x_trg.shape[0])]
        logs_trg['trg_rec_sim'] = np.mean(trg_rec_sim)
        return logs_trg

    def E(self, z, x):
        """
        z: list of latent variables for an input x
         Out:
             energy E from p = exp(-E(x)) (torch tensor (B,))
             Where p(z|x) = p_{\theta}(x|z)p(z) / p_{\theta}(x)
        """
        if self.nvae.dataset == 'mnist':
            x = torch.bernoulli(x)

        logits, log_p = self.cond_decode(z)
        output = self.nvae.decoder_output(logits)

        log_pxz = - reconstruction_loss(output, x, crop=self.nvae.crop_output).sum()
        log_p_z = torch.cat(log_p).sum()
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

        z = [z_curr.clone().requires_grad_() for z_curr in z]
        # assert z.grad == None
        E = self.E(z, x)
        E.backward()
        if f:
            torch.set_grad_enabled(False)
        return [z_curr.grad.data for z_curr in z]
