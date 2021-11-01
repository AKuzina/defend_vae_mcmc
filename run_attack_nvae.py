import os
import sys
import wandb
import numpy as np
import pandas as pd
from tqdm import tqdm
import copy

import torchvision
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from ml_collections import ConfigDict


from NVAE.utils import get_arch_cells
from NVAE.model import AutoEncoder
from NVAE.datasets import get_loaders

from vae.model.nvae import NVAE
from utils.wandb import get_experiments, load_model, load_classifier, creat_attack_conf
from attack import trainer

# args
from absl import app
from absl import flags
from ml_collections.config_flags import config_flags

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", default="config.py:attack_nvae")


def cli_main(_):
    pl.seed_everything(1234)

    if "absl.logging" in sys.modules:
        import absl.logging

        absl.logging.set_verbosity("info")
        absl.logging.set_stderrthreshold("info")
    args = FLAGS.config
    with args.unlocked():
        args.model.chckpt_path = {
            'celeba_64': '../NVAE/checkpoint/celeba_64.pt',
            'mnist': '../NVAE/checkpoint/mnist.pt'
        }[args.model.dset]
        args.model.dset_path = {
            'celeba_64': '../NVAE/datasets/celeba64_lmdb',
            'mnist': '../NVAE/datasets/mnist'
        }[args.model.dset]

    # load model checkpoint
    checkpoint = torch.load(args.model.chckpt_path, map_location='cpu')
    nvae_args = checkpoint['args']
    print(ConfigDict(vars(nvae_args)))
    nvae_args.data = args.model.dset_path
    nvae_args.distributed = False
    with args.unlocked():
        args.nvae = ConfigDict(vars(nvae_args))
        # if not args.nvae.contains('min_groups_per_scale'):
        args.nvae.min_groups_per_scale = args.nvae.get('min_groups_per_scale', default=1)

    ## ------------
    # data
    # ------------
    _, _, test_dataset, _ = get_loaders(args.nvae)
    dataloader = DataLoader(test_dataset, batch_size=1000, shuffle=True)

    # ------------
    # load pretrained model
    # ------------
    arch_instance = get_arch_cells(args.nvae.arch_instance)
    model = AutoEncoder(args.nvae, None, arch_instance)

    model.load_state_dict(checkpoint['state_dict'], strict=False)
    model = model.cuda()
    model = model.eval()
    for param in model.parameters():
        param.requires_grad = False
        
    nvae_model = NVAE(model, t=args.model.temp, n_connect=args.model.connect)
    # load classifier
    # ids = dict(
    #     mnist={
    #         1: '2pzktaxw',#wrong
    #         2: '2pzktaxw',
    #         4: '3rh9ja97'
    #     }[args.model.connect],
    #     celeba_64={
    #         1: '2pzktaxw',#wrong
    #         2: '2pzktaxw',
    #         4: '3rh9ja97'
    #     }[args.model.connect]
    # )[args.model.dset]
    # clf_model = load_classifier(ids, N=6)
    clf_model = []

    # load attacks, if already trained
    CONF_attack = dict(args.attack)
    CONF_attack['hmc_steps'] = 0
    CONF_attack.pop('hmc_eps')
    CONF_model = dict(args.model)
    CONF_model.pop('chckpt_path')
    CONF_model.pop('dset_path')
    print(CONF_attack, CONF_model)
    ids = get_experiments(config=creat_attack_conf(CONF_model,
                                                   CONF_attack,
                                                   {'mode': 'attack_nvae'}))
    attack_id = None
    if len(ids) > 0:
        print('Attack already found:')
        print(ids[0])
        attack_id = ids[0]
    with args.unlocked():
        args.attack.attack_id = attack_id

    # ------------
    # wandb
    # ------------
    os.environ["WANDB_API_KEY"] = '5532aa3f6f581daa33de08ae6bccd7bbdf271c12'
    tags = [
        args.mode,
        args.nvae.dataset,
        args.attack.type,
        args.attack.loss_type,
        args.attack.reg_type
    ]

    wandb.init(
        project="vcd_vae",
        tags=tags,
        entity='anna_jakub',  # USER NAME HERE
        config=copy.deepcopy(dict(args))
    )
    wandb.config.update(flags.FLAGS)

    # run attack
    trainer.train(nvae_model, clf_model, dataloader, args)


if __name__ == "__main__":
    app.run(cli_main)
