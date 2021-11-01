import torch
import wandb
import numpy as np
import os
from sklearn.model_selection import train_test_split

from thirdparty.pytorch_msssim import msssim
from utils.divergence import gaus_skl
from utils.wandb import *
from attack.attacker import AttackVAE, AttackNVAE


def train(model, clf_model, dataloader, args):
    # get reference point
    model = model.to('cuda:0')
    clf_model = [m.to('cuda:0') for m in clf_model]
    x, y = iter(dataloader).__next__()
    if len(y.shape) > 1 or args.mode == 'attack_nvae':
        x_trg, x_ref, y_trg, y_ref = train_test_split(x, y, test_size=args.attack.N_ref)
    else:
        x_trg, x_ref, y_trg, y_ref = train_test_split(x, y, stratify=y, test_size=args.attack.N_ref)

    if args.attack.type == 'supervised':
        if len(y.shape) > 1 or args.mode == 'attack_nvae':
            _, x_trg, _, y_trg = train_test_split(x_trg, y_trg, test_size=args.attack.N_trg)
        else:
            _, x_trg, _, y_trg = train_test_split(x_trg, y_trg, stratify=y_trg, test_size=args.attack.N_trg)
        x_trg = x_trg.to('cuda:0')
    else:
        x_trg = None

    # init attacker
    if args.mode == 'attack_vae':
        attkr = AttackVAE(model.vae, args.attack)
    elif args.mode == 'attack_nvae':
        attkr = AttackNVAE(model, args.attack)

    # train adversarial samples
    x_ref = x_ref.to('cuda:0')
    y_ref = y_ref.to('cuda:0')
    train_fn(attkr, model, clf_model, x_ref, y_ref, args, x_trg)


def train_fn(attkr, model, clf_model, x_ref, y_ref, args, x_trg=None):
    total_logs = {}
    # loop over reference images
    for step, xi in enumerate(x_ref):
        xi = xi.unsqueeze(0)
        if args.attack.attack_id is not None:
            file = wandb.restore('x_adv_{}.pth'.format(step),
                                 os.path.join(USER, PROJECT, args.attack.attack_id), root='wandb', replace=True)
            x_adv = torch.load(file.name)
            x_adv = x_adv.to('cuda')
        else:
            x_adv = attkr.get_attack(xi, all_trg=x_trg)
            x_adv = torch.cat(x_adv)
        logs = model.eval_attack(xi, y_ref[step], x_adv, step, clf_model,
                                 x_trg=x_trg, hmc_steps=args.attack.hmc_steps,
                                 hmc_eps=args.attack.hmc_eps)
        wandb.log(logs)

        for k in logs.keys():
            if not isinstance(logs[k], wandb.Image):
                if 'Av_' + k not in total_logs.keys():
                    total_logs['Av_' + k] = 0.
                total_logs['Av_' + k] += logs[k]
        total_logs['Similarity_diff'] = total_logs['Av_ref_sim'] - total_logs['Av_ref_rec_sim']

        for k in total_logs:
            wandb.run.summary[k] = total_logs[k]/(step+1)

    for k in ['Av_ref_acc', 'Av_adv_acc']:
        if k not in total_logs.keys():
            wandb.run.summary[k] = np.mean([
                    total_logs[k+'_{}'.format(i)]/(step+1) for i in range(len(clf_model))
                ])
