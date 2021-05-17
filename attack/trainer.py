import torch
import wandb
import numpy as np
import os
from sklearn.model_selection import train_test_split

from thirdparty.pytorch_msssim import msssim
from utils.divergence import gaus_skl


def train(model, clf_model, dataloader, args):
    # get reference point
    x, y = iter(dataloader).__next__()
    x_trg, x_ref, y_trg, y_ref = train_test_split(x, y, stratify=y, test_size=args.attack.N_ref)

    if args.attack.type == 'supervised':
        _, x_trg, _, y_trg = train_test_split(x_trg, y_trg, stratify=y_trg,
                                              test_size=args.attack.N_trg)
    else:
        x_trg = None
    # train adversarial samples
    train_fn(model, clf_model, x_ref, y_ref, args, x_trg)


def train_fn(model, clf_model, x_ref, y_ref, args, x_trg=None):
    total_logs = {
        'Av_ref_sim': 0.,
        'Av_ref_rec_sim': 0.,
        'Av_omega': 0.,
        'Similarity_diff': 0.,
        'Av_ref_acc': 0.,
        'Av_adv_acc': 0.
    }

    if x_trg is not None:
        total_logs['Av_trg_rec_sim'] = 0.
        # save target images
        with torch.no_grad():
            trg_recon, _, _, _, _ = model.forward(x_trg)
        trg_recon = trg_recon.reshape(x_trg.shape).detach()

        # get training function
        from attack.supervised import generate_adv as attack_fun
    else:
        trg_recon = None
        from attack.unsupervised import generate_adv as attack_fun

    # loop over reference images
    for step, xi in enumerate(x_ref):
        xi = xi.unsqueeze(0)
        x_hist = [xi]
        x_adv = attack_fun(all_trg=x_trg, x_init=xi, vae=model.vae, args=args.attack)
        x_hist.append(torch.cat(x_adv))

        # save hist and reconstuctions
        x_hist = torch.cat(x_hist)
        torch.save(x_hist, os.path.join(wandb.run.dir, 'x_hist_{}.pth'.format(step)))
        with torch.no_grad():
            z_mu, z_logvar = model.vae.q_z(x_hist)
            z_sample = model.vae.reparametrize(z_mu, z_logvar)
            if args.attack.hmc_steps > 0:
                from vae.model.vcd import HMC_sampler, VaeTarget
                target = VaeTarget(model.vae.decoder, model.vae.prior, model.vae.log_lik)
                Q_t = HMC_sampler(target, 0.02, L=10, adaptive=False)
                z_t, acc = Q_t.sample(z_sample[1:], x_hist[1:], args.attack.hmc_steps, 0)
                z_sample = torch.cat([z_sample[:1], z_t])
                x_recon, x_logvar = model.vae.decoder(z_sample)
                z_mu[1:] = z_t
            else:
                x_recon, x_logvar, _, _, _ = model.forward(x_hist)

        x_recon = x_recon.reshape(x_hist.shape).detach()
        torch.save(x_recon, os.path.join(wandb.run.dir, 'x_recon_{}.pth'.format(step)))

        # compute metrics
        logs = save_stats(x_hist.detach(), y_ref[step], x_recon.detach(), x_logvar.detach(),
                          z_sample, z_mu, z_logvar, model.vae, clf_model, x_trg, trg_recon)
        if args.attack.hmc_steps > 0:
            logs['HMC acc.rate'] = torch.stack(acc).mean(0).item()
            logs['HMC eps'] = Q_t.eps
        wandb.log(logs)

        total_logs['Av_ref_sim'] += logs['Ref similarity']
        total_logs['Av_ref_rec_sim'] += logs['Rec similarity']
        total_logs['Av_omega'] += logs['SKL [q_a | q]']
        total_logs['Av_ref_acc'] += logs['ref_acc']
        total_logs['Av_adv_acc'] += logs['adv_acc']
        total_logs['Similarity_diff'] = total_logs['Av_ref_sim'] - total_logs['Av_ref_rec_sim']
        if x_trg is not None:
            total_logs['Av_trg_rec_sim'] += logs['Target Rec similarity']

        for k in total_logs:
            wandb.run.summary[k] = total_logs[k]/(step+1)


def save_stats(x_hist, y_ref, x_mu, x_logvar, z, z_mu, z_logvar, vae, clf_model,
               trg=None, trg_recon=None):
    MB = x_hist.shape[0]

    # similarity with x_ref
    ref_sim = [msssim(x_hist[:1], x_a.unsqueeze(0), window_size=14,
                      normalize='relu').data for x_a in x_hist[1:]]
    # similarity with reconstractions
    rec_sim = [msssim(x_mu[:1], x_a.unsqueeze(0), window_size=14,
                      normalize='relu').data for x_a in x_mu[1:]]
    # sKL in latent space
    s_kl = gaus_skl(z_mu[:1], z_logvar[:1], z_mu[1:], z_logvar[1:]).mean()
    mus = (z_mu[:1] - z_mu[1:]).pow(2).sum(1).mean()

    # eps norm
    eps_norms = [torch.norm(x_hist[:1]-x_a.unsqueeze(0)) for x_a in x_hist[1:]]

    # reconstruction errors (log likelihoods)
    x_a = x_hist[1:].reshape(MB-1, -1)
    x_r = x_hist[:1].reshape(1, -1)
    x_mu_a = x_mu[1:].reshape(MB-1, -1)
    x_logvar_a = x_logvar[1:].reshape(MB-1, -1)
    x_mu_r = x_mu[:1].reshape(1, -1)
    x_logvar_r = x_logvar[:1].reshape(1, -1)

    log_p_xa_zr = vae.reconstruction_error(x_a, x_mu_r, x_logvar_r).mean(0)
    log_p_xr_zr = vae.reconstruction_error(x_r, x_mu_r, x_logvar_r).mean(0)
    log_p_xa_za = vae.reconstruction_error(x_a, x_mu_a, x_logvar_a).mean(0)
    log_p_xr_za = vae.reconstruction_error(x_r, x_mu_a, x_logvar_a).mean(0)

    # classifier accuracy
    y_ref_pred = clf_model(z[:1]).argmax(1)
    y_adv_pred = clf_model(z[1:]).argmax(1)
    logs = {
        'Adversarial Inputs': wandb.Image(x_hist[1:], mode='L'),
        'Adversarial Rec': wandb.Image(x_mu[1:], mode='L'),
        'Ref similarity': np.mean(ref_sim),
        'Rec similarity': np.mean(rec_sim),
        'SKL [q_a | q]': s_kl,
        'Mean dist': mus,
        'Eps norm': np.mean(eps_norms),
        '-log_p_xa_zr': log_p_xa_zr,
        '-log_p_xr_zr': log_p_xr_zr,
        '-log_p_xa_za': log_p_xa_za,
        '-log_p_xr_za': log_p_xr_za,
        'ref_acc': sum(y_ref_pred == y_ref),
        'adv_acc': sum(y_adv_pred == y_ref)/y_adv_pred.shape[0]
    }

    if trg is not None:
        trg_rec_sim = [msssim(trg_recon[i-1:i], x_mu[i:i+1], window_size=14,
                              normalize='relu').data for i in range(1, MB)]
        logs['Target Rec similarity'] = np.mean(trg_rec_sim)
        logs['Target Inputs'] = wandb.Image(trg, mode='L')
        logs['Target Rec'] = wandb.Image(trg_recon, mode='L')

    return logs


def batch_min_max_scale(x):
    MB = x.shape[0]
    mi = x.reshape(MB, -1).min(dim=1).values.reshape(MB, 1,1,1)
    ma = x.reshape(MB, -1).max(dim=1).values.reshape(MB, 1,1,1)
    return (x - mi)/(ma - mi)


