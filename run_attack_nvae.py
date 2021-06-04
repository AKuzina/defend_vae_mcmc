import os
import sys
import wandb
import numpy as np
import pandas as pd
from tqdm import tqdm

import torchvision
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from ml_collections import ConfigDict


from NVAE.utils import get_arch_cells
from NVAE.model import AutoEncoder
from NVAE.datasets import get_loaders
# from NVAE.distributions import Normal

from attack.nvae import trainer
from vae.model.nvae import NVAE
from utils.wandb import load_classifier

# args
from absl import app
from absl import flags
from ml_collections.config_flags import config_flags

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", default="config.py:nvae")


def cli_main(_):
    pl.seed_everything(1234)

    if "absl.logging" in sys.modules:
        import absl.logging

        absl.logging.set_verbosity("info")
        absl.logging.set_stderrthreshold("info")
    args = FLAGS.config

    # load model checkpoint
    checkpoint = torch.load(args.model.chckpt_path, map_location='cpu')
    nvae_args = checkpoint['args']
    nvae_args.data = args.model.dset_path
    nvae_args.distributed = False
    with args.unlocked():
        args.nvae = ConfigDict(vars(nvae_args))

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
        
    nvae_model = NVAE(model)
    # load classifier
    ids = {
    1:'34zr5d3j',    
    2:'2pzktaxw',
    4:'3rh9ja97',
    }[args.model.connect]
    clf_model = load_classifier(ids, N=6)

    # ------------
    # wandb
    # ------------
    os.environ["WANDB_API_KEY"] = '5532aa3f6f581daa33de08ae6bccd7bbdf271c12'
    tags = [
        'nvae',
        args.nvae.dataset,
        args.attack.type,
        args.attack.loss_type,
        args.attack.reg_type
    ]

    wandb.init(
        project="vcd_vae",
        tags=tags,
        entity='anna_jakub'  # USER NAME HERE
    )
    wandb.config.update(flags.FLAGS)

    # run attack
    trainer.train(nvae_model, clf_model, dataloader, args)


if __name__ == "__main__":
    app.run(cli_main)
