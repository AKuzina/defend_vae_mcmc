import torch
import wandb
import numpy as np
import os
from sklearn.model_selection import train_test_split

import NVAE.utils
# from attack.nvae.supervised import generate_adv, sup_loss
from thirdparty.pytorch_msssim import msssim
from utils.divergence import gaus_skl


def train(model, clf_model, dataloader, args):
      # get reference point
    model = model.to('cuda:0')
    clf_model = [m.to('cuda:0') for m in clf_model]
    x, y = iter(dataloader).__next__()
    y = torch.stack(y, 1)

    x_trg, x_ref, y_trg, y_ref = train_test_split(x, y, test_size=args.attack.N_ref)
    print(y_trg[0])
    if args.attack.type == 'supervised':
        _, x_trg, _, y_trg = train_test_split(x_trg, y_trg, test_size=args.attack.N_trg)
        x_trg = x_trg.to('cuda:0')
    else:
        x_trg = None
    # train adversarial samples
    x_ref = x_ref.to('cuda:0')
    y_ref = y_ref.to('cuda:0')
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

    for i in range(len(clf_model)):
        total_logs['Av_ref_acc_{}'.format(i)] = 0.
        total_logs['Av_adv_acc_{}'.format(i)] = 0.


    if x_trg is not None:
        total_logs['Av_trg_rec_sim'] = 0.
        # save target images
        with torch.no_grad():
            logits, _, _, _, _, _ = model(x_trg, connect=args.model.connect, t=args.model.temp)
        trg_recon = model.nvae.decoder_output(logits).sample()

        # get training function
        from attack.nvae.supervised import generate_adv as attack_fun
    else:
        trg_recon = None
        from attack.nvae.unsupervised import generate_adv as attack_fun
        
    
    # train adversarial attack
    for step, xi in enumerate(x_ref):
        xi = xi.unsqueeze(0)
        x_hist = [xi]
        x_adv = attack_fun(all_trg=x_trg, x_init=xi, vae=model, args=args)
        x_hist.append(torch.cat(x_adv))

        x_hist = torch.cat(x_hist)
    
        # save hist and reconstuctions
        torch.save(x_hist, os.path.join(wandb.run.dir, 'x_hist_{}.pth'.format(step)))
        with torch.no_grad():
            z_sample_adv, logits, q_dist_adv, _, _, _, _ = model(x_hist[1:], connect=args.model.connect,
                                                              t=args.model.temp, return_samples=True)
            z_sample_adv = torch.stack(z_sample_adv, 1) # MB x n_con x z_dim
            if args.attack.hmc_steps > 0:
                from vae.model.vcd import HMC_sampler
                Q_t = HMC_sampler(model, args.attack.hmc_eps, L=40, adaptive=False)
                new_z = []
                all_acc = []
                for i in range(z_sample_adv.shape[0]):
                    z, acc = Q_t.sample(z_sample_adv[i:i+1], x_hist[i+1:i+2], args.attack.hmc_steps, 0)
                    new_z.append(z)
                    all_acc.append(acc)
                z_sample_adv = torch.cat(new_z)
                logits, log_p_z = model.decode(z_sample_adv, t=args.model.temp)
                
            x_recon_adv = model.nvae.decoder_output(logits).sample()
            z_sample_ref, logits, q_dist_ref, _, _, _, _ = model(x_hist[:1], connect=args.model.connect, 
                                                   t=args.model.temp, return_samples=True)
            z_sample_ref = torch.stack(z_sample_ref, 1) # 1 x n_con x z_dim
            print(x_hist[1:].shape, x_trg.shape, z_sample_adv.shape, z_sample_ref.shape)
            x_recon_ref = model.nvae.decoder_output(logits).sample()
            

        torch.save(x_recon_ref.cpu(), os.path.join(wandb.run.dir, 'x_recon_ref_{}.pth'.format(step)))
        torch.save(x_recon_adv.cpu(), os.path.join(wandb.run.dir, 'x_recon_adv_{}.pth'.format(step)))
        logs = save_stats(x_hist.detach(), y_ref[step], x_recon_ref.detach(), x_recon_adv.detach(), 
                          q_dist_ref, q_dist_adv, z_sample_ref, z_sample_adv, model, clf_model, x_trg, trg_recon)
        if args.attack.hmc_steps > 0:
            logs['HMC acc.rate'] = torch.stack(acc).mean(0).item()
            logs['HMC eps'] = Q_t.eps
        wandb.log(logs)

        total_logs['Av_ref_sim'] += logs['Ref similarity']
        total_logs['Av_ref_rec_sim'] += logs['Rec similarity']
        total_logs['Av_omega'] += logs['SKL [q_a | q]']
        total_logs['Similarity_diff'] = total_logs['Av_ref_sim'] - total_logs['Av_ref_rec_sim']
        if x_trg is not None:
            total_logs['Av_trg_rec_sim'] += logs['Target Rec similarity']
        for i in range(len(clf_model)):
            total_logs['Av_ref_acc_{}'.format(i)] += logs['ref_acc_{}'.format(i)]
            total_logs['Av_adv_acc_{}'.format(i)] += logs['adv_acc_{}'.format(i)]
            
        for k in total_logs:
            wandb.run.summary[k] = total_logs[k]/(step+1)
    wandb.run.summary['Av_ref_acc'] = np.mean([total_logs['Av_ref_acc_{}'.format(i)]/(step+1) for i in range(len(clf_model))])
    wandb.run.summary['Av_adv_acc'] = np.mean([total_logs['Av_adv_acc_{}'.format(i)]/(step+1) for i in range(len(clf_model))])



def save_stats(x_hist, y_ref, x_r_recon, x_a_recon, q_dist_ref, q_dist_hist, z_ref, z_adv, model, 
               clf_model, trg=None, trg_recon=None):
    MB = x_hist.shape[0]

    # similarity with x_ref
    ref_sim = [msssim(x_hist[:1], x_hist[i:i + 1], normalize='relu').data.cpu() for i in range(1, MB)]
    
    # sKL in latent space
    s_kl = skl(q_dist_ref, q_dist_hist).data.cpu()
    z_adv = z_adv.reshape(x_a_recon.shape[0], -1)
    z_ref = z_ref.reshape(1, -1)
    mus = (z_ref - z_adv).pow(2).sum(1).mean()

    # eps norm
    eps_norms = [torch.norm(x_hist[:1]-x_a.unsqueeze(0)).cpu() for x_a in x_hist[1:]]

    # similarity with reference (rec)
    rec_sim = [msssim(x_r_recon, x_a_recon[i-1:i], normalize='relu').data.cpu() for i in range(1, MB)]
    print(ref_sim, rec_sim)
    # list of ELBO_k
#     elbos = elbo_k(x_hist, model)

    logs = {
        'Adversarial Inputs': wandb.Image(x_hist[1:].cpu()),
        'Adversarial Rec': wandb.Image(x_a_recon.cpu()),
        'Ref similarity': np.mean(ref_sim),
        'Rec similarity': np.mean(rec_sim),
        'SKL [q_a | q]': s_kl, #np.mean(s_kl),
        'Mean dist': mus.cpu(),
        'Eps norm': np.mean(eps_norms),
    }

    all_tasks = [4, 15, 22, 31, 36, 37]
    for ind, m in enumerate(clf_model):
        y_ref_pred = m(z_ref).argmax(1)
        y_adv_pred = m(z_adv).argmax(1)
        logs['ref_acc_{}'.format(ind)] = sum(y_ref_pred.cpu() == y_ref[all_tasks[ind]].cpu())
        logs['adv_acc_{}'.format(ind)] = sum(y_adv_pred.cpu() == y_ref[all_tasks[ind]].cpu())/y_adv_pred.shape[0]

    if trg is not None:
#         print(MB, trg_recon[0].data, x_a_recon[0])
        trg_rec_sim = [msssim(trg_recon[i-1:i], x_a_recon[i-1:i],
                       normalize='relu').data.cpu() for i in range(1, MB)]
        logs['Target Rec similarity'] = np.mean(trg_rec_sim)
        logs['Target Inputs'] = wandb.Image(trg.cpu())
        logs['Target Rec'] = wandb.Image(trg_recon.cpu())

    return logs


def skl(q_ref, q_adv):
    loss = 0.
    for q_1, q_2 in zip(q_ref, q_adv):
        # print(q_1.mu.shape, q_2.mu.shape)
        sym_kl = gaus_skl(q_1.mu, 2*torch.log(q_1.sigma), q_2.mu, 2*torch.log(q_2.sigma), (1,2,3))
        loss += sym_kl.mean()
    return loss


def batch_min_max_scale(x):
    MB = x.shape[0]
    mi = x.reshape(MB, -1).min(dim=1).values.reshape(MB, 1,1,1)
    ma = x.reshape(MB, -1).max(dim=1).values.reshape(MB, 1,1,1)
    return (x - mi)/(ma - mi)


def elbo_k(x, model):
    x = torch.clamp(x, 0, 1).cuda()
    N_max = 35
    res = []
    for i in range(N_max):
        with torch.no_grad():
            logits, _, all_q, all_p, kl_all, _ = model(x, connect=i)
        output = model.decoder_output(logits)
        recon_loss = NVAE.utils.reconstruction_loss(output, x, crop=model.crop_output)
        balanced_kl, _, _ = NVAE.utils.kl_balancer(kl_all, kl_balance=False)
        nelbo = recon_loss + balanced_kl
        # bits per dim
        bpd_coeff = 1. / np.log(2.) / (3 * 64 * 64)
        nelbo = nelbo*bpd_coeff
        res.append(nelbo.cpu())
    return torch.stack(res, 1)