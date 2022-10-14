import os
import sys
import copy
import torch
import wandb
import pytorch_lightning as pl
import numpy as np

from datasets import load_dataset
from vae.model import *
from utils.wandb import USER, PROJECT, API_KEY

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
    elif args.model in ['tc_conv']:
        vae = TCVAE
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
    model.data_module = data_module


    # ------------
    # weight and bias + trainer
    # ------------
    os.environ["WANDB_API_KEY"] = API_KEY
    if args.debug:
        os.environ["WANDB_MODE"] = "dryrun"
    tags = [
        args.prior,
        args.model,
        args.dataset_name,
        'train_vae'
    ]

    wandb_logger = pl.loggers.WandbLogger(project=PROJECT,
                                          tags=tags,
                                          config=copy.deepcopy(dict(args)),
                                          log_model=True,
                                          entity=USER,
                                          )

    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='epoch')

    trainer = pl.Trainer(gpus=args.gpus,
                         max_epochs=args.max_epochs,
                         callbacks=[lr_monitor],
                         logger=wandb_logger,
                         )

    no_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    wandb_logger.experiment.summary["no_params"] = no_params
    trainer.fit(model, datamodule=data_module)
    trainer.save_checkpoint(os.path.join(wandb.run.dir, f'checkpoints/last.ckpt'))
    # ------------
    # testing
    # ------------
    result = trainer.test(datamodule=data_module)
    print(result)


if __name__ == "__main__":
    app.run(cli_main)
