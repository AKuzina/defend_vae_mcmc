import math
import numpy as np
import torch
import torch.nn as nn
import wandb
import pytorch_lightning as pl

# from utils.distributions import log_Bernoulli, log_Gaus_diag, log_Logistic_256
# from vae.model.priors import StandardNormal, RealNPV
from vae.utils.architecture import get_architecture
from vae.model.vae import StandardVAE
from vae.model.var_posterior import NormalQ


class VCD_VAE(StandardVAE):
    def __init__(self, hparams):
        super().__init__(hparams)
        self.automatic_optimization = False
        self.q_0 = NormalQ(hparams.z_dim)
        self.target = VaeTarget(self.vae.decoder, self.vae.prior, self.vae.log_lik)
        self.Q_t = HMC_sampler(self.target, 0.5/hparams.z_dim, L=5, adaptive=True)

    def training_step(self, batch, batch_idx):
        enc_opt, dec_opt = self.optimizers()

        # forward pass
        z_0, z_mean_0, z_logvar_0, x_mean_0, x_logvar_0, z_t, x_mean_t, x_logvar_t = self.forward(batch[0])
        # decoder
        dec_opt.zero_grad()
        loss_dec = self.decoder_loss(z_t, x_mean_t, x_logvar_t, batch[0])
        self.manual_backward(loss_dec)

        # encoder
        enc_opt.zero_grad()
        loss_1, loss_2, loss_3, re, kl = self.encoder_loss(z_0, z_mean_0, z_logvar_0, x_mean_0, x_logvar_0, batch[0], z_t, loss_dec.detach())
        loss_enc = loss_1 + loss_2 - loss_3
        self.manual_backward(loss_enc)

        dec_opt.step()
        enc_opt.step()

        # logging
        with torch.no_grad():
            q_0_entropy = self.q_0.entropy(z_logvar_0).mean()

        # plain elbo with q_0
        self.log('train_loss', loss_1.item(), on_step=False,
                 on_epoch=True, prog_bar=True, logger=True)
        self.log('train_kl', kl.detach(), on_step=True,
                 on_epoch=True, prog_bar=False, logger=True)
        self.log('train_re', re.detach(), on_step=True,
                 on_epoch=True, prog_bar=False, logger=True)
        self.log('q_0_entropy', q_0_entropy.detach(), on_step=False,
                 on_epoch=True, prog_bar=False, logger=True)

    def decoder_loss(self, z_t, x_mean_t, x_logvar_t, x):
        """
        Find MLE parameters of the generatie model: E_{q^t} \log p_{\theta}(x, z) -> max

        Input:
            z_t: sample from the improved posterior
            x_mean_t, x_logvar_t: parameters of p_{\theta}(x|z_t)
            x: true point
        """
        # q^t is fixed
        z_t = z_t.detach()
        # cond. likelihood
        lop_pxz = self.vae.log_lik(x, x_mean_t, x_logvar_t, dim=1)
        # prior
        log_pz = self.vae.prior.log_prob(z_t)
        loss = - lop_pxz - log_pz
        return loss

    def encoder_loss(self, z_0, z_mean_0, z_logvar_0, x_mean_0, x_logvar_0, x, z_t, decoder_loss):
        """
        Find parameters of the Variational posterior using Variational Contractive Divergence

        z_mean_0, z_logvar_0: parameters of q_0
        z_0: sample from the reparametrizable q_0
        x_mean_0, x_logvar_0: parameters of p_{\theta}(x|z_0)
        x: true point
        z_t: sample from the improved posterior (after HMC)
        """
        # part 1 (conventional ELBO with q_0)
        lop_pxz = self.vae.log_lik(x, x_mean_0, x_logvar_0, dim=1)
        log_pz = self.vae.prior.log_prob(z_0)
        log_qz0 = self.q_0.log_prob(z_0, z_mean_0, z_logvar_0)
        loss_1 = lop_pxz + log_pz - log_qz0

        # part 2
        loss_2 = self.q_0.log_prob(z_t, z_mean_0, z_logvar_0)

        # part 3
        with torch.no_grad():
            w_0 = - decoder_loss - self.q_0.log_prob(z_t, z_mean_0, z_logvar_0)
            # substract control variates here

        loss_3 = w_0 * self.q_0.log_prob(z_0, z_mean_0, z_logvar_0)

        return loss_1, loss_2, loss_3, -lop_pxz, log_qz0 - log_pz


    def forward(self, x):
        # encode and decode using q^0
        x_mean_0, x_logvar_0, z_0, z_mean_0, z_logvar_0 = self.vae(x)

        # HMC - sample z from q^t (we do not need to backprop through transition kernel)
        with torch.no_grad():
            z_t = self.Q_t.sample(z_0.detach(), self.hparams.HMC_steps, self.hparams.HMC_burn_in)

        # decode z_t
        x_mean_t, x_logvar_t = self.vae.p_x(z_t)
        x_mean_t = x_mean_t.reshape(x_mean.shape[0], -1)
        x_logvar_t = x_logvar_t.reshape(x_mean.shape[0], -1)

        return z_0, z_mean_0, z_logvar_0, x_mean_0, x_logvar_0, z_t, x_mean_t, x_logvar_t

    def validation_step(self, batch, batch_idx):
        # q_0 only
        x_mean_0, x_logvar_0, z_0, z_mean_0, z_logvar_0 = self.vae(batch[0])
        re = self.vae.log_lik(bathc[0], x_mean_0, x_logvar_0, dim=1).mean(0)
        kl = torch.mean(self.q_0.log_prob(z_0, z_mean_0, z_logvar_0) - self.prior.log_prob(z_0), 0)

        # logging
        self.log('val_loss', re + kl, on_step=False, on_epoch=True, prog_bar=True,
                 logger=True)
        self.log('val_kl', kl, on_step=False, on_epoch=True, prog_bar=False,
                 logger=True)
        self.log('val_re', re, on_step=False, on_epoch=True, prog_bar=False,
                 logger=True)
        # reconstructions
        if self.x_rec is None:
            self.x_rec = batch[0][:25]

    def validation_epoch_end(self, outputs):
        # samples
        sample = self.vae.generate_x(9)
        sample = sample.reshape(9, self.params.image_size[0], self.params.image_size[1], -1).detach()
        self.log('Prior_sample', wandb.Image(sample))

        # reconstructions
        x_mean_0, _, z_0, _, z_logvar_0 = self.vae(self.x_rec[:9])
        plot_rec = x_mean_0.detach()
        plot_rec = plot_rec.reshape(9, self.params.image_size[0],
                                  self.params.image_size[1], -1).detach()
        self.log('Reconstructions', wandb.Image(plot_rec))

        # reconstructions after HMC
        z_t = self.Q_t.sample_n(z_0.detach())
        x_mean_t, _ = self.vae.p_x(z_t)
        plot_rec = x_mean_t.reshape(9, self.params.image_size[0],
                                    self.params.image_size[1], -1).detach()
        self.log('Reconstructions_t', wandb.Image(plot_rec))


    def test_step(self, batch, batch_idx):
        # elbo
        loss, re, kl, _ = self.vae.ELBO(batch[0], beta=self.params.beta, average=True)
        # IWAE
        nll = self.vae.estimate_nll(batch[0], self.params.is_k)
        # logging
        self.log('test_elbo', -loss.detach(), on_step=True, on_epoch=False, prog_bar=False,
                 logger=True)
        return {'nll':nll, 'labels':batch[1]}

    def test_epoch_end(self, outputs):
        nll = torch.cat([x['nll'] for x in outputs]).data.cpu()
        labels = torch.cat([x['labels'] for x in outputs]).data.cpu()

        # NLL on the whole test set
        self.log('test_nll', nll.mean())
        logger = self.logger.experiment
        # Per task eval
        for l in np.unique(labels):
            idx = np.where(labels == l)[0]
            data = nll[idx]
            for i in range(len(data)):
                logger.log({'test_nll_task{}'.format(l): data[i]})

        # samples
        N = 100
        sample = self.vae.generate_x(N)
        sample = sample.reshape(N, self.params.image_size[0], self.params.image_size[1], -1).detach()
        self.log('Samples', wandb.Image(sample))

    def configure_optimizers(self):
        optimizers = [
            torch.optim.Adam(self.vae.encoder.parameters(), lr=self.params.lr),
            torch.optim.Adam([self.vae.decoder.parameters(), self.vae.prior.parameters()], lr=self.params.lr)
        ]

        schedulers = [
            {
                'scheduler': ReduceLROnPlateau(optimizers[0],
                                               factor=self.params.lr_factor,
                                               patience=self.params.lr_patience),
                'monitor': 'val_loss',
                'reduce_on_plateau': True
            },
            {
                'scheduler': ReduceLROnPlateau(optimizers[1],
                                               factor=self.params.lr_factor,
                                               patience=self.params.lr_patience),
                'monitor': 'val_loss',
                'reduce_on_plateau': True
            }
        ]

        return optimizers, schedulers


class VaeTarget:
    def __init__(self, decoder, prior, log_lik):
        self.decoder = decoder
        self.prior = prior
        self.log_lik = log_lik

    def E(self, x, z):
        '''
         Out:
             energy E from p = exp(-E(x)) (torch tensor (B,))
             Where p(z|x) = p_{\theta}(x|z)p(z) / p_{\theta}(x)
        '''
        MB = x_mean.shape[0]
        x_mean, x_logvar = self.decoder(z)
        log_pxz = self.log_lik(x.reshape(MB, -1), x_mean.reshape(MB, -1),
                               x_logvar.reshape(MB, -1), dim=1)
        log_pz = self.prior.log_prob(z, 1)
        return - log_pxz - log_pz

    def E_ratio(self, x_s, x_p):
        return -(self.E(x_p) - self.E(x_s))

    def grad_E(self, x):
        '''
         Out:
             grad of E from p = exp(-E(x)) (torch tensor (B,D))
        '''
        x = x.clone().requires_grad_()
        E = self.E(x)
        E.sum().backward()
        return x.grad


class HMC_sampler:
    def __init__(self, target, eps, L=5, adaptive=True):
        """
        eps: initial step size in leap frog
        L: num of step in leap frog
        adaptive: bool, where to adapt eps during burn in
        """
        self.eps = eps
        self.L = L
        self.adaptive = adaptive
        self.target = target

    def transition(self, x_s):
        p_s = torch.randn_like(x_s)

        x_end = x_s.clone()
        p_end = p_s.clone()

        p_end -= 0.5 * self.eps * self.target.grad_E(x_end)
        for i in range(self.L):
            x_end += self.eps * p_end
            if i < self.L - 1:
                p_end -= self.eps * self.target.grad_E(x_end)
            else:
                p_end -= 0.5 * self.eps * self.target.grad_E(x_end)

        q_ratio = 0.5 * (torch.sum(p_s ** 2, dim=1) - torch.sum(p_end ** 2, dim=1))
        return x_end, q_ratio

    def sample(self, x_0, N, burn_in=1):
        """
        Code is based on https://github.com/franrruiz/vcd_divergence/blob/master/demo_main.m
        and https://github.com/evgenii-egorov/sk-bdl/blob/main/seminar_18/notebook/MCMC.ipynb

        N: number of points in the chain
        burn_in: number of points for burn in
        """
        chain = [x_0]
        for i in range(burn_in + N):
            # propose point
            x_proposed, q_ratio = self.transition(chain[i])
            # compute MH test
            target_ratio = self.target.E_ratio(chain[i], x_proposed)
            log_u = torch.log(torch.rand(x_proposed.size(0)))
            accept_flag = log_u < (target_ratio + q_ratio)

            # make a step or stay at the same point
            x_proposed[~accept_flag] = chain[i][~accept_flag]
            # if burn in - adapt step size, else collect sample
            if i < burn_in:
                if self.adaptive:
                    prop_acepted = mean(accept_flag)
                    self.eps += 0.01*((prop_acepted - 0.9)/0.9)*self.eps
            else:
                chain.append(x_proposed)
        return chain[-1]