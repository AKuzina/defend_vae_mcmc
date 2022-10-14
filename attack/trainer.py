import torch
import wandb
import numpy as np
from sklearn.model_selection import train_test_split
import os

from utils.wandb import *
from attack.attacker import AttackVAE


def train(model, dataloader, args):
    # get reference point
    model = model.to('cuda:0')
    x, y = iter(dataloader).__next__()
    if isinstance(y, list):
        y = y[0]
    if len(y.shape) > 1:
        x_trg, x_ref, y_trg, y_ref = train_test_split(x, y, test_size=args.attack.N_ref)
    else:
        x_trg, x_ref, y_trg, y_ref = train_test_split(x, y, stratify=y, test_size=args.attack.N_ref)

    if args.attack.type == 'supervised':
        if len(y.shape) > 1:
            _, x_trg, _, y_trg = train_test_split(x_trg, y_trg, test_size=args.attack.N_trg)
        else:
            _, x_trg, _, y_trg = train_test_split(x_trg, y_trg, stratify=y_trg, test_size=args.attack.N_trg)
        x_trg = x_trg.to('cuda:0')
    else:
        x_trg = None

    # init attacker
    attkr = AttackVAE(model, args.attack)

    # train adversarial samples
    x_ref = x_ref.to('cuda:0')
    y_ref = y_ref.to('cuda:0')

    # For classifier attack we may have  more than 2 task (e.g. class and color for color mnist)
    if 'clf' in args.attack.loss_type:
        task_loop = range(len(args.model.n_classes))
    else:
        task_loop = [None]

    for task in task_loop:
        train_fn(attkr, model, x_ref, y_ref, args, x_trg, task)


def train_fn(attkr, model, x_ref, y_ref, args, x_trg=None, task=None):
    total_logs = {}
    # loop over reference images
    for step, (xi, yi) in enumerate(zip(x_ref, y_ref)):
        xi = xi.unsqueeze(0)
        yi = yi.unsqueeze(0)

        # if attack already trained: load
        try:
            pref = ''
            if task is not None:
                pref = f'task{task}_'
            file = wandb.restore(f'{pref}x_adv_{step}.pth',
                                 os.path.join(USER, PROJECT, args.attack.attack_id),
                                 root='wandb', replace=True)
            x_adv = torch.load(file.name)
            x_adv = x_adv.to('cuda')
        # otherwise: train attack
        except:
            x_adv = attkr.get_attack(xi, all_trg=x_trg, task=task)
            x_adv = torch.cat(x_adv)
        logs = model.eval_attack(xi, yi, x_adv, step, x_trg=x_trg,
                                 hmc_steps=args.attack.hmc_steps,
                                 hmc_eps=args.attack.hmc_eps,
                                 task=task)

        for k in logs.keys():
            if not isinstance(logs[k], wandb.Image):
                if 'Av_' + k not in total_logs.keys():
                    total_logs['Av_' + k] = 0.
                total_logs['Av_' + k] += logs[k]
        # add task num
        if task is not None:
            for k in list(logs.keys()):
                logs[f'task{task}/{k}'] = logs.pop(k)
        wandb.log(logs)

        if task is not None:
            pref = f'task{task}/'
        else:
            pref = ''
        for k in total_logs:
            wandb.run.summary[pref+k] = total_logs[k]/(step+1)

    for k in ['Av_ref_acc', 'Av_adv_acc']:
        if k not in total_logs.keys() and k+'_0' in total_logs.keys():
            wandb.run.summary[k] = np.mean([
                    total_logs[k+f'_{i}']/(step+1) for i in range(len(model.classifier))
                ])
