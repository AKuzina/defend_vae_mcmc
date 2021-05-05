import os
import sys
import copy
import torch
import wandb
import pytorch_lightning as pl
import numpy as np

from datasets import load_dataset
from vae.model import *

# args
from absl import app
from absl import flags
from ml_collections.config_flags import config_flags

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", default="config.py:train")


def model_name(args):
    n = args.prior + '_' + args.model + '_' + str(args.z_dim)
    return n


def cli_main(_):
    if "absl.logging" in sys.modules:
        import absl.logging

        absl.logging.set_verbosity("info")
        absl.logging.set_stderrthreshold("info")
    args = FLAGS.config
    print(args)
    # Set the seed
    pl.seed_everything(1234+args.iter)
    torch.manual_seed(1234+args.iter)
    np.random.seed(1234+args.iter)

    if args.model in ['conv']:
        vae = StandardVAE
    elif args.model in ['vcd']:
        vae = VCD_VAE
    else:
        raise ValueError('Unknown model type')

    # ------------
    # data
    # ------------
    data_module, args = load_dataset(args)

    # ------------
    # model
    # ------------
    model = vae(args)

    # ------------
    # training
    # ------------
    checkpnts = pl.callbacks.ModelCheckpoint(
        monitor='val_loss',
        mode='min',
        save_last=True,
    )
    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='epoch')

    early_stop = pl.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=int(args.lr_patience*1.5),
        verbose=True,
        mode='min',
        strict=False
    )

    # ------------
    # weight and bias + trainer
    # ------------
    os.environ["WANDB_API_KEY"] = '5532aa3f6f581daa33de08ae6bccd7bbdf271c12'
    if args.debug:
        os.environ["WANDB_MODE"] = "dryrun"
    tags = [
        args.prior,
        args.model,
        args.dataset_name
    ]

    wandb_logger = pl.loggers.WandbLogger(project='vcd_vae',
                                          tags=tags,
                                          config=copy.deepcopy(dict(args)),
                                          log_model=True,
                                          entity="akuzina",
                                          )

    trainer = pl.Trainer(gpus=args.gpus,
                         max_epochs=args.max_epochs,
                         callbacks=[early_stop, lr_monitor],
                         logger=wandb_logger,
                         checkpoint_callback=checkpnts  # in newer lightning this goes to callbaks as well
                         )

    no_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    wandb_logger.experiment.summary["no_params"] = no_params
    trainer.fit(model, datamodule=data_module)

    # ------------
    # testing
    # ------------
    result = trainer.test(datamodule=data_module)
    print(result)


if __name__ == "__main__":
    app.run(cli_main)