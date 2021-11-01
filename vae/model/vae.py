import math
import numpy as np
import torch
import torch.nn as nn
import wandb
import os
import pytorch_lightning as pl

from utils.distributions import log_Bernoulli, log_Gaus_diag, log_Logistic_256
from utils.divergence import gaus_skl
from vae.model.priors import StandardNormal, RealNPV
from vae.model.hmc import HMC_sampler, VaeTarget
from vae.utils.architecture import get_architecture
from thirdparty.pytorch_msssim import msssim


class VAE(nn.Module):
    def __init__(self, architecture, likelihood, prior):
        super(VAE, self).__init__()
        assert likelihood in ['bernoulli', 'gaussian', 'logistic'], \
            'unknown likelihood type {}'.format(likelihood)
        if likelihood == 'bernoulli':
            self.log_lik = lambda x, x_mean, x_logvar, dim: log_Bernoulli(x, x_mean, dim=dim)
        elif likelihood == 'gaussian':
            self.log_lik = log_Normal_diag
        elif likelihood == 'logistic':
            self.log_lik = log_Logistic_256

        self.likelihood = likelihood
        self.prior = prior

        self.encoder, self.decoder = architecture()

    def q_z(self, x):
        """
        Encoder
        :param x: input image
        :return: parameters of q(z|x), (MB, z_dim)
        """
        z_q_mean, z_q_logvar = self.encoder(x)
        return z_q_mean, z_q_logvar

    def p_x(self, z):
        """
        Decoder
        :param z: latent vector          (MB, z_dim)
        :return: parameters of p(x|z)    (MB, inp_dim)
        """
        x_mean, x_logvar = self.decoder(z)
        return x_mean, x_logvar

    def forward(self, x):
        # z ~ q(z | x)
        z_q_mean, z_q_logvar = self.q_z(x)
        z_q = self.reparametrize(z_q_mean, z_q_logvar)
        x_mean, x_logvar = self.p_x(z_q)

        # reshape for convolutional architectures
        x_mean = x_mean.reshape(x_mean.shape[0], -1)
        x_logvar = x_logvar.reshape(x_mean.shape[0], -1)
        return x_mean, x_logvar, z_q, z_q_mean, z_q_logvar

    def reconstruct_x(self, x):
        x_mean, _, _, _, _ = self.forward(x)
        return x_mean

    def reconstruction_error(self, x, x_mean, x_logvar):
        return -self.log_lik(x, x_mean, x_logvar, 1)

    def kl(self, z, z_mean, z_logvar):
        """
        KL-divergence between q(z|x) and p(z)
        :param z:           (MB, z_dim) sample from q
        :param z_mean:      (MB, z_dim) mean of q
        :param z_logvar:    (MB, z_dim) log variance of q
        :return: KL         (MB, )
        """
        log_p_z = self.prior.log_prob(z)
        log_q_z = log_Gaus_diag(z, z_mean, z_logvar, dim=1)
        kl_value = log_q_z - log_p_z
        return kl_value

    def ELBO(self, x, beta=1., average=False):
        """
        :param x:   (MB, inp_dim)
        :param beta: Float
        :param average: Compute average over mini batch or not, bool
        :return: Re + beta * KL
        """
        x_mean, x_logvar, z_q, z_q_mean, z_q_logvar = self.forward(x)
        MB = x.shape[0]

        # data term
        re = self.reconstruction_error(x.view(MB, -1), x_mean, x_logvar)
        # KL-divergence
        kl = self.kl(z_q, z_q_mean, z_q_logvar)

        loss = re + beta * kl

        if average:
            loss = torch.mean(loss, 0)
            re = torch.mean(re, 0)
            kl = torch.mean(kl, 0)

        q_entropy = 0.5 * torch.sum(1 + math.log(math.pi*2) + z_q_logvar.detach(), 1)
        q_entropy = torch.mean(q_entropy, 0)
        return loss, re, kl, q_entropy

    def estimate_nll(self, X, K=1000):
        """
        Estimate NLL by importance sampling
        :param X: mini-batch, (N, x_dim(s))
        :param samples: Samples per observation
        :return: IS estimate
        """
        N = X.shape[0]
        mu_z, logvar_z = self.q_z(X)  # -> (N, z_dim)
        total_nll = []

        for j in range(N):
            mu_z_curr, logvar_z_curr = mu_z[j:j+1].repeat(K, 1), logvar_z[j:j+1].repeat(K, 1)

            # for r in range(0, rep):
            z_q = self.reparametrize(mu_z_curr, logvar_z_curr)
            x_mean, x_logvar = self.p_x(z_q)
            log_p_x = -self.reconstruction_error(X[j:j+1].view(1, -1),
                                                 x_mean.reshape(K, -1),
                                                 x_logvar.reshape(K, -1))
            log_p_z = self.prior.log_prob(z_q)
            log_q_z = log_Gaus_diag(z_q, mu_z_curr, logvar_z_curr, dim=-1)
            ll_curr = log_p_x + log_p_z - log_q_z
            ll_obs = torch.logsumexp(ll_curr, 0) - np.log(K)
            total_nll.append(-ll_obs)
        return torch.tensor(total_nll)

    def generate_x(self, N=25):
        z_sample =self.prior.sample_n(N)
        x_mean, _ = self.p_x(z_sample)
        return x_mean

    @staticmethod
    def reparametrize(mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = torch.FloatTensor(std.size()).normal_().to(mu.device)
        return eps.mul(std).add_(mu)


class StandardVAE(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()

        arc, hparams = get_architecture(hparams)
        if hparams.prior == 'standard':
            prior = StandardNormal(hparams.z_dim)
        elif hparams.prior == 'realnvp':
            prior = RealNPV(hparams.z_dim, 3)

        self.vae = VAE(arc, hparams.likelihood, prior)
        self.params = hparams
        self.save_hyperparameters()
        self.x_rec = None

    def forward(self, x):
        """
        Returns reconstructions (x_mu, x_logvar) as vector (MB, -1)
        """
        return self.vae(x)

    def forward_reshaped(self, x):
        z_q_mean, z_q_logvar = self.vae.q_z(x)
        z_q = self.vae.reparametrize(z_q_mean, z_q_logvar)
        x_mean, x_logvar = self.vae.p_x(z_q)

        return x_mean, x_logvar, z_q, z_q_mean, z_q_logvar

    def training_step(self, batch, batch_idx):
        beta = self.params.beta
        if self.params.warmup > 0:
            beta *= min(1, self.current_epoch/self.params.warmup)

        loss, re, kl, q_ent = self.vae.ELBO(batch[0], beta=beta, average=True)
        # logging
        self.log('train_loss', loss, on_step=False,
                 on_epoch=True, prog_bar=True, logger=True)
        self.log('train_kl', kl.detach(), on_step=True,
                 on_epoch=True, prog_bar=False, logger=True)
        self.log('train_re', re.detach(), on_step=True,
                 on_epoch=True, prog_bar=False, logger=True)
        self.log('q_0_entropy', q_ent.detach(), on_step=False,
                 on_epoch=True, prog_bar=False, logger=True)
        self.log('beta', beta, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, re, kl, _ = self.vae.ELBO(batch[0], beta=self.params.beta, average=True)
        # logging
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True,
                 logger=True)
        self.log('val_kl', kl.detach(), on_step=False, on_epoch=True, prog_bar=False,
                 logger=True)
        self.log('val_re', re.detach(), on_step=False, on_epoch=True, prog_bar=False,
                 logger=True)
        # reconstructions
        if self.x_rec is None:
            self.x_rec = batch[0]

    def validation_epoch_end(self, outputs):
        # samples
        sample = self.vae.generate_x(9)
        if len(sample.shape) < 3:
            sample = sample.reshape(9, self.params.image_size[0],
                                    self.params.image_size[1], -1)
        self.log('Prior_sample', wandb.Image(sample.detach()))
        # reconstructions
        plot_rec = self.vae.reconstruct_x(self.x_rec[:9])
        if len(plot_rec.shape) < 3:
            plot_rec = plot_rec.reshape(9, self.params.image_size[0],
                                        self.params.image_size[1], -1)
        self.log('Reconstructions', wandb.Image(plot_rec.detach()))

        # latent space
        z_mean, _ = self.vae.q_z(self.x_rec)
        dta = [[x[0], x[1]] for x in z_mean]
        table = wandb.Table(data=dta, columns=["x", "y"])
        logger = self.logger.experiment
        logger.log({'Latent_space':  wandb.plot.scatter(table, "x", "y")})

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

        # bpd
        size_coef = self.params.image_size[0]*self.params.image_size[1]*self.params.image_size[2]
        bpd_coeff = 1. / np.log(2.) / size_coef
        bpd = nll.mean() * bpd_coeff
        self.log('test_bpd', bpd)

        # samples
        N = 100
        sample = self.vae.generate_x(N)
        if len(sample.shape) < 3:
            sample = sample.reshape(N, self.params.image_size[0],
                                    self.params.image_size[1], -1)
        self.log('Samples', wandb.Image(sample.detach()))

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.params.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                               factor=self.params.lr_factor,
                                                               patience=self.params.lr_patience)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'reduce_on_plateau': True,
                'monitor': 'val_loss'
            }
        }

    def sample_posterior(self, z_init, x, n_steps, step_size):
        target = VaeTarget(self.vae.decoder, self.vae.prior, self.vae.log_lik)
        Q_t = HMC_sampler(target, step_size, L=20, adaptive=True)
        z_t, acc = Q_t.sample(z_init, x, n_steps, int(0.5*n_steps))
        logs = {
            'hmc_acc_rate': torch.stack(acc).mean(0).item(),
            'hmc_eps': Q_t.eps
        }
        return z_t, logs

    def eval_attack(self, x_ref, y_ref, x_adv, step, clf_model, x_trg=None, hmc_steps=0, **kwargs):
        """
        x_trg: torch.tensor (N_trg, x_dim)
        x_ref: torch.tensor (1, x_dim)
        x_adv: torch.tensor (N_trg, x_dim)
        """
        torch.save(x_adv.cpu(), os.path.join(wandb.run.dir, 'x_adv_{}.pth'.format(step)))
        torch.save(x_ref.cpu(), os.path.join(wandb.run.dir, 'x_ref_{}.pth'.format(step)))

        # get reconstructions
        logs = {}
        with torch.no_grad():
            x_ref_m, x_ref_lv, z_ref, z_ref_m, z_ref_lv = self.forward_reshaped(x_ref)
            x_adv_m, x_adv_lv, z_adv, z_adv_m, z_adv_lv = self.forward_reshaped(x_adv)
            adv_dist = (x_adv, x_adv_m, x_adv_lv)
            zadv_dist = (z_adv, z_adv_m, z_adv_lv)
        if hmc_steps > 0:
            # z_ref_t = self.sample_posterior(z_ref, x_ref, hmc_steps, kwargs['hmc_eps'])
            # x_ref_m_t, x_ref_lv_t = self.vae.p_x(z_ref_t)
            z_adv_t, hmc_logs = self.sample_posterior(z_adv, x_adv, hmc_steps, kwargs['hmc_eps'])
            logs.update(hmc_logs)
            x_adv_m_t, x_adv_lv_t = self.vae.p_x(z_adv_t)
            adv_dist = (x_adv, x_adv_m_t, x_adv_lv_t)
            zadv_dist = (z_adv_t, z_adv_m, z_adv_lv)

        ref_logs = self.eval_attack_reference(
            ref_dist=(x_ref, x_ref_m, x_ref_lv),
            adv_dist=adv_dist,
            zref_dist=(z_ref, z_ref_m, z_ref_lv),
            zadv_dist=zadv_dist
        )
        logs.update(ref_logs)

        # Add supervised-only metrics
        if x_trg is not None:
            trg_logs = self.eval_attack_target(x_trg, adv_dist[1], step)
            logs.update(trg_logs)

        # Add classifier accuracy
        z = z_adv if hmc_steps == 0 else z_adv_t
        clf_logs = self.eval_attack_classifier(clf_model, z_ref, y_ref, z)
        logs.update(clf_logs)
        return logs

    def eval_attack_target(self, x_trg, x_adv_m, step):
        x_trg_m, _, _, _, _ = self.forward_reshaped(x_trg)

        trg_rec_sim = [msssim(x_trg_m[i:i+1], x_adv_m[i:i+1], 14, normalize='relu').data.cpu()
                       for i in range(x_trg.shape[0])]
        logs_trg = {
            'trg_rec_sim': np.mean(trg_rec_sim)
        }
        if step == 0:
            torch.save(x_trg.cpu(), os.path.join(wandb.run.dir, 'x_trg.pth'.format(step)))
            torch.save(x_trg_m.cpu(), os.path.join(wandb.run.dir, 'x_trg_rec.pth'.format(step)))
            logs_trg['Target Inputs'] = wandb.Image(x_trg.cpu())
            logs_trg['Target Rec'] = wandb.Image(x_trg_m.cpu())
        return logs_trg

    def eval_attack_reference(self, ref_dist, adv_dist, zref_dist, zadv_dist):
        x_ref, x_ref_m, x_ref_lv = ref_dist
        x_adv, x_adv_m, x_adv_lv = adv_dist
        z_ref, z_ref_m, z_ref_lv = zref_dist
        z_adv, z_adv_m, z_adv_lv = zadv_dist
        N_a = x_adv.shape[0]

        # eps norm
        eps_norms = [torch.norm(x_ref - x_a.unsqueeze(0)).cpu() for x_a in x_adv]

        # msssims
        ref_sim = [msssim(x_ref, x_a.unsqueeze(0), 14, normalize='relu').data.cpu() for x_a in x_adv]
        ref_rec_sim = [msssim(x_ref_m, x_a.unsqueeze(0), 14, normalize='relu').data.cpu() for x_a in x_adv_m]

        # latent space
        s_kl = gaus_skl(z_ref_m, z_ref_lv, z_adv_m, z_adv_lv).mean()
        mus = (z_ref - z_adv).pow(2).sum(1).mean()

        # log-likelihood
        get_nll = lambda x, m, lv: self.vae.reconstruction_error(x, m, lv).mean(0)
        log_p_xa_zr = get_nll(x_adv.reshape(N_a, -1), x_ref_m.reshape(1, -1), x_ref_lv.reshape(1, -1))
        log_p_xr_zr = get_nll(x_ref.reshape(1, -1), x_ref_m.reshape(1, -1), x_ref_lv.reshape(1, -1))
        log_p_xa_za = get_nll(x_adv.reshape(N_a, -1), x_adv_m.reshape(N_a, -1), x_adv_lv.reshape(N_a, -1))
        log_p_xr_za = get_nll(x_ref.reshape(1, -1), x_adv_m.reshape(N_a, -1), x_adv_lv.reshape(N_a, -1))

        logs = {
            'Adversarial Inputs': wandb.Image(x_adv.cpu()),
            'Adversarial Rec': wandb.Image(x_adv_m.cpu()),

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

    def eval_attack_classifier(self, clf_model, z_ref, y_ref, z_adv):
        # classifier accuracy
        logs = {}
        if isinstance(clf_model, list) and len(clf_model) > 1:
            for ind, m in enumerate(clf_model):
                y_ref_pred = m(z_ref).argmax(1)
                y_adv_pred = m(z_adv).argmax(1)
                logs['ref_acc_{}'.format(ind)] = sum(y_ref_pred.cpu() == y_ref[ind].cpu())
                logs['adv_acc_{}'.format(ind)] = sum(y_adv_pred.cpu() == y_ref[ind].cpu())/y_adv_pred.shape[0]
        else:
            if not isinstance(clf_model, list):
                m = clf_model
            elif len(clf_model) == 1:
                m = clf_model[0]
            y_ref_pred = m(z_ref).argmax(1)
            y_adv_pred = m(z_adv).argmax(1)
            logs['ref_acc'] = sum(y_ref_pred.cpu() == y_ref.cpu())
            logs['adv_acc'] = sum(y_adv_pred.cpu() == y_ref.cpu())/y_adv_pred.shape[0]
        return logs


