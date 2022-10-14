import math
import torch

from utils.distributions import log_Gaus_diag
from vae.model.vae import StandardVAE


class TCVAE(StandardVAE):
    def __init__(self, hparams):
        super(TCVAE, self).__init__(hparams)

    def calc_entropies(self, z_sample, z_q_mean, z_q_logvar):
        MB, z_dim = z_sample.shape

        z_sample = z_sample.view(MB, 1, z_dim)
        z_q_mean = z_q_mean.view(1, MB, z_dim)
        z_q_logvar = z_q_logvar.view(1, MB, z_dim)

        log_qz_i = -0.5 * (math.log(2.0*math.pi) +
                          z_q_logvar +
                          torch.pow(z_sample - z_q_mean, 2) / (torch.exp(z_q_logvar) + 1e-10))  # MB x MB x z_dim

        marginal_entropies = (math.log(MB) - torch.logsumexp(log_qz_i, dim=0)) # MB x z_dim
        log_qz = log_qz_i.sum(2) # MB x MB
        joint_entropy = math.log(MB) - torch.logsumexp(log_qz, dim=0)  # MB
        return marginal_entropies, joint_entropy

    def training_step(self, batch, batch_idx):
        beta = self.params.beta
        if self.params.warmup > 0:
            beta *= min(1, self.current_epoch/self.params.warmup)

        x_mean, x_logvar, z_q, z_q_mean, z_q_logvar = self.forward(batch[0])
        log_q_zx = log_Gaus_diag(z_q, z_q_mean, z_q_logvar, dim=1)  # MB
        log_pz = self.vae.prior.log_prob(z_q)  # MB

        MB = batch[0].shape[0]

        # data term
        re = self.vae.reconstruction_error(batch[0].view(MB, -1), x_mean, x_logvar)
        # Entropies

        marginal_entropies, joint_entropy = self.calc_entropies(z_q, z_q_mean, z_q_logvar)


        # Mutual information: log q(z|x) - log q(z)
        MI = log_q_zx + joint_entropy  # MB
        TC = (marginal_entropies.sum(1) - joint_entropy)  # MB
        dim_kl = (-marginal_entropies.sum(1) - log_pz)

        loss = re.mean(0) + \
               MI.mean(0) + \
               beta * TC.mean(0) +\
               dim_kl.mean(0)

        kl = (log_q_zx.detach() - log_pz.detach()).mean(0)
        # logging
        self.log('train_loss', loss, on_step=False,
                 on_epoch=True, prog_bar=True, logger=True)
        self.log('train_kl', kl.detach(), on_step=True,
                 on_epoch=True, prog_bar=False, logger=True)
        self.log('train_re', re.mean(0).detach(), on_step=True,
                 on_epoch=True, prog_bar=False, logger=True)
        self.log('train_tc', TC.mean(0).detach(), on_step=False,
                 on_epoch=True, prog_bar=False, logger=True)
        self.log('beta', beta, on_step=False, on_epoch=True, prog_bar=False, logger=True)

        return loss
