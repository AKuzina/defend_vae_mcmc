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


from datasets import load_dataset
from utils.wandb import get_experiments, load_model
from clf import trainer_nvae
from vae.model.nvae import NVAE

# args
from absl import app
from absl import flags
from ml_collections.config_flags import config_flags

#nvae
from NVAE.utils import get_arch_cells
from NVAE.model import AutoEncoder
from NVAE.datasets import get_loaders

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", default="config.py:clf_nvae")


def model_name(args):
    n = args.prior + '_' + args.arc_type + '_' + str(args.z_dim)
    return n


def cli_main(_):
    pl.seed_everything(1234)

    if "absl.logging" in sys.modules:
        import absl.logging
        absl.logging.set_verbosity("info")
        absl.logging.set_stderrthreshold("info")
    args = FLAGS.config
    print(args)

  # load model checkpoint
    checkpoint = torch.load(args.model.chckpt_path, map_location='cpu')
    nvae_args = checkpoint['args']
    nvae_args.data = args.model.dset_path
    nvae_args.distributed = False
    with args.unlocked():
        args.nvae = ConfigDict(vars(nvae_args))

    # ------------
    # data
    # ------------
    args.nvae.batch_size = 150
    train_dataloader, val_dataloader, _, _ = get_loaders(args.nvae)
#     train_dataloader = DataLoader(train_dataset, batch_size=1000, shuffle=True)
#     val_dataloader = DataLoader(val_dataset, batch_size=1000, shuffle=False)

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
    
    
#     ids = get_experiments(config=args.model)
#     print(ids)
#     # model = load_model(ids[0]).vae
#     model = load_model(ids[0])
#     model.eval()
#     with args.unlocked():
#         args['model_id'] = ids[0]

    # ------------
    # wandb
    # ------------
    os.environ["WANDB_API_KEY"] = '5532aa3f6f581daa33de08ae6bccd7bbdf271c12'  # WAND API KEY HERE
    tags = [
        'nvae',
        args.nvae.dataset,
    ]

    wandb.init(
        project="vcd_vae",
        tags=tags,
        entity='anna_jakub',  # USER NAME HERE
        config=copy.deepcopy(dict(args)),
    )
    wandb.config.update(flags.FLAGS)

    # run attack
    trainer_nvae.train(nvae_model, train_dataloader, val_dataloader, [4, 15, 22, 31, 36, 37], args)


if __name__ == "__main__":
    app.run(cli_main)