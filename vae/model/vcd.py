import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
import pytorch_lightning as pl
from itertools import chain

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
        self.Q_t = HMC_sampler(self.target, 5./hparams.z_dim, L=5, adaptive=True)
        self.C = torch.zeros(1)

    def forward(self, x):
        z_0, z_mean_0, z_logvar_0, z_t = self.encode(batch)
        x_mean_t, x_logvar_t = self.decode(z_t)
        return x_mean_t, x_logvar_t, z_0, z_mean_0, z_logvar_0

    def encode(self, x):
        z_mean_0, z_logvar_0 = self.vae.encoder(x)
        z_0 = self.vae.reparametrize(z_mean_0, z_logvar_0)
        z_t, acc = self.Q_t.sample(z_0, x, self.params.HMC_steps,
                                   self.params.HMC_burn_in)
        return z_0, z_mean_0, z_logvar_0, z_t

    def decode(self, z):
        MB = z.shape[0]
        x_mean, x_logvar = self.vae.p_x(z)
        return x_mean.reshape(MB, -1), x_logvar.reshape(MB, -1)

    def training_step(self, batch, batch_idx, optimizer_idx):
        MB = batch[0].shape[0]
        x_long = batch[0].reshape(MB, -1)

        enc_opt, dec_opt = self.optimizers()
        z_0, z_mean_0, z_logvar_0, z_t = self.encode(batch[0])

        ## ENCODER
        # forward
        enc_opt.zero_grad()
        x_mean_0, x_logvar_0 = self.decode(z_0)
        x_mean_t, x_logvar_t = self.decode(z_t)
        # backward
        loss_1, loss_2, loss_3, re, kl = self.encoder_loss(z_0, z_mean_0, z_logvar_0,
                                                           x_mean_0, x_logvar_0, x_long,
                                                           z_t.detach(), x_mean_t, x_logvar_t)
        loss_enc = -(loss_1 + loss_2 + loss_3)
        self.manual_backward(loss_enc.mean(0))
        # nn.utils.clip_grad_norm_(
        #     self.vae.encoder.parameters(), 1
        # )
        enc_opt.step()
        # enc_opt.zero_grad()


        ## DECODER
        # forward pass
        dec_opt.zero_grad()
        z_0 = self.vae.reparametrize(z_mean_0, z_logvar_0)
        z_t, acc = self.Q_t.sample(z_0, batch[0], self.params.HMC_steps,
                                   self.params.HMC_burn_in)
        x_mean_t, x_logvar_t = self.decode(z_t.detach())

        # backward
        loss_dec = self.decoder_loss(z_t, x_mean_t, x_logvar_t, x_long)
        self.manual_backward(loss_dec.mean(0))
        dec_opt.step()

        # logging
        with torch.no_grad():
            q_0_entropy = self.q_0.entropy(z_logvar_0).mean()

        # plain elbo with q_0
        self.log('train_loss', (re + kl).mean(0).item(), on_step=False,
                 on_epoch=True, prog_bar=True, logger=True)
        self.log('train_kl', kl.mean(0).item(), on_step=True,
                 on_epoch=True, prog_bar=False, logger=True)
        self.log('train_re', re.mean(0).item(), on_step=True,
                 on_epoch=True, prog_bar=False, logger=True)
        self.log('q_0_entropy', q_0_entropy.item(), on_step=False,
                 on_epoch=True, prog_bar=False, logger=True)

        # log VCD losses
        self.log('decoder_loss', loss_dec.mean(0).item(), on_step=False,
                 on_epoch=True, prog_bar=False, logger=True)
        self.log('encoder_loss', loss_enc.mean(0).item(), on_step=False,
                 on_epoch=True, prog_bar=False, logger=True)
        self.log('HMC acc. rate', torch.stack(acc).mean(0).item(), on_step=True,
                 on_epoch=False, prog_bar=False, logger=True)
        self.log('HMC eps', self.Q_t.eps, on_step=True,
                 on_epoch=False, prog_bar=False, logger=True)

    def decoder_loss(self, z_t, x_mean_t, x_logvar_t, x):
        """
        Find MLE parameters of the generatie model: E_{q^t} log p_{\theta}(x, z) -> max

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
        loss = - lop_pxz  - log_pz
        self.log('- log pxt | zt', -lop_pxz.mean(0).item(), on_step=False,
                 on_epoch=True, prog_bar=False, logger=True)
        self.log('- log pzt', -log_pz.mean(0).item(), on_step=False,
                 on_epoch=True, prog_bar=False, logger=True)
        return loss

    def encoder_loss(self, z_0, z_mean_0, z_logvar_0, x_mean_0, x_logvar_0, x, z_t, x_mean_t, x_logvar_t):
        """
        Find parameters of the Variational posterior using
        Variational Contractive Divergence

        z_mean_0, z_logvar_0: parameters of q_0
        z_0: sample from the reparametrizable q_0
        x_mean_0, x_logvar_0: parameters of p_{\theta}(x|z_0)
        x: true point
        z_t: sample from the improved posterior (after HMC)
        """
        # part 1 (conventional ELBO with q_0)
        lop_pxz = self.vae.log_lik(x, x_mean_0, x_logvar_0, dim=1)
        log_pz = self.vae.prior.log_prob(z_0)
        neg_log_qz0 = self.q_0.entropy(z_logvar_0)
        loss_1 = lop_pxz + log_pz + neg_log_qz0

        self.log('- log pz0', -log_pz.mean(0).item(), on_step=False,
                 on_epoch=True, prog_bar=False, logger=True)

        # part 2
        z_t_ = z_t.detach()
        loss_2 = self.q_0.log_prob(z_t_, z_mean_0, z_logvar_0)

        # part 3
        z_0_ = z_0.detach()
        # with torch.no_grad():
        log_pxzt = self.vae.log_lik(x, x_mean_t.detach(), x_logvar_t.detach(), dim=1)
        log_pzt = self.vae.prior.log_prob(z_t_)
        w_0 = log_pxzt + log_pzt - self.q_0.log_prob(z_t_, z_mean_0.detach(), z_logvar_0.detach())
        # substract control variates here
        self.C = self.C.to(w_0.device)
        w_0_C = w_0.detach() - self.C
        # update C
        self.C = 0.9*self.C + 0.1*w_0.mean().detach()

        # save stats
        self.log('w_0', w_0.mean(0).item(), on_step=True,
                 on_epoch=False, prog_bar=False, logger=True)
        self.log('Control var', self.C.item(), on_step=True,
                 on_epoch=False, prog_bar=False, logger=True)

        loss_3 = -w_0_C * self.q_0.log_prob(z_0_, z_mean_0, z_logvar_0)
        return loss_1, loss_2, loss_3, -lop_pxz, -neg_log_qz0 - log_pz

    def validation_step(self, batch, batch_idx):
        # q_0 only
        x_mean_0, x_logvar_0, z_0, z_mean_0, z_logvar_0 = self.vae(batch[0])
        MB = x_mean_0.shape[0]
        x = batch[0].reshape(MB, -1)

        re = -self.vae.log_lik(x, x_mean_0, x_logvar_0, dim=1).mean(0)
        log_qz = self.q_0.log_prob(z_0, z_mean_0, z_logvar_0)
        kl = torch.mean(log_qz - self.vae.prior.log_prob(z_0), 0)

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
        sample = sample.reshape(9, self.params.image_size[0],
                                self.params.image_size[1], -1).detach()
        self.log('Prior_sample', wandb.Image(sample))

        # reconstructions
        x_mean_0, _, z_0, _, z_logvar_0 = self.vae(self.x_rec[:9])
        plot_rec = x_mean_0.detach()
        plot_rec = plot_rec.reshape(9, self.params.image_size[0],
                                    self.params.image_size[1], -1).detach()
        self.log('Reconstructions', wandb.Image(plot_rec))

        # reconstructions after HMC
        z_t, _ = self.Q_t.sample(z_0.detach(), self.x_rec[:9], self.params.HMC_steps,
                                 self.params.HMC_burn_in)
        x_mean_t, _ = self.vae.p_x(z_t)
        plot_rec = x_mean_t.reshape(9, self.params.image_size[0],
                                    self.params.image_size[1], -1).detach()
        self.log('Reconstructions_t', wandb.Image(plot_rec))

    def test_step(self, batch, batch_idx):
        # elbo
        loss, re, kl, _ = self.vae.ELBO(batch[0], beta=1, average=True)
        # IWAE
        nll = self.vae.estimate_nll(batch[0], self.params.is_k)
        # logging
        self.log('test_elbo', -loss.detach(), on_step=True, on_epoch=False,
                 prog_bar=False, logger=True)
        return {'nll': nll, 'labels': batch[1]}

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
        sample = sample.reshape(N, self.params.image_size[0],
                                self.params.image_size[1], -1).detach()
        self.log('Samples', wandb.Image(sample))

    def configure_optimizers(self):
        optimizers = [
            optim.Adam(self.vae.encoder.parameters(), lr=self.params.lr),
            optim.Adam(chain(self.vae.decoder.parameters(),
                             self.vae.prior.parameters()), lr=self.params.lr)
        ]

        schedulers = [
            {
                'scheduler': optim.lr_scheduler.ReduceLROnPlateau(
                    optimizers[0],
                    factor=self.params.lr_factor,
                    patience=self.params.lr_patience
                ),
                'monitor': 'val_loss',
                'reduce_on_plateau': True
            },
            {
                'scheduler': optim.lr_scheduler.ReduceLROnPlateau(
                    optimizers[1],
                    factor=self.params.lr_factor,
                    patience=self.params.lr_patience
                ),
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

    def transition(self, z_s, x):
        """
        z_s: z start
        x: data point
        """
        p_s = torch.randn_like(z_s)

        z_end = z_s.clone().detach()
        p_end = p_s.clone().detach()

        p_end -= 0.5 * self.eps * self.target.grad_E(z_end, x)
        for i in range(self.L):
            z_end += self.eps * p_end
            if i < self.L - 1:
                p_end -= self.eps * self.target.grad_E(z_end, x)
            else:
                p_end -= 0.5 * self.eps * self.target.grad_E(z_end, x)
        if len(p_s.shape) > 2:
            p_s = p_s.reshape(p_s.shape[0], -1)
            p_end = p_end.reshape(p_s.shape[0], -1)

        q_ratio = 0.5 * (torch.sum(p_s ** 2, dim=1) - torch.sum(p_end ** 2, dim=1))
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
        z_curr = z_0.clone()

        for i in range(burn_in + N):
            # propose point
            z_proposed, ratio = self.transition(z_curr, x)
            # compute MH test
            log_u = torch.log(torch.rand(z_proposed.size(0), device=x.device))
            accept_flag = log_u < ratio

            # make a step or stay at the same point
            z_curr[accept_flag] = z_proposed[accept_flag]
            # if burn in - adapt step size, else - collect samples
            prop_accepted = torch.mean(accept_flag.float())
            if i < burn_in:
                if self.adaptive:
                    self.eps += 0.01*((prop_accepted - 0.9)/0.9)*self.eps
            else:
                acceptance_rate.append(prop_accepted)
                chain.append(z_curr)
        return chain[-1], acceptance_rate
