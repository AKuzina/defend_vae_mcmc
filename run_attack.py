import os
import sys
import wandb
import pytorch_lightning as pl

from datasets import load_dataset
from utils.wandb import get_experiments, load_model, load_classifier
from attack import trainer

# args
from absl import app
from absl import flags
from ml_collections.config_flags import config_flags

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", default="config.py:attack")


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
    assert args.attack.loss_type in ['skl', 'kl_forward', 'kl_reverse']
    assert args.attack.reg_type in ['penalty', 'projection']

    # ------------
    # data
    # ------------
    data_module, args.model = load_dataset(args.model, binarize=False)
    data_module.setup('test')
    dataloader = data_module.test_dataloader()

    # ------------
    # load pretrained model
    # ------------
    ids = get_experiments(config=args.model)
    print(ids)
    model = load_model(ids[0])
    model.eval()

    # classifier
    clf_args = {'model_id': ids[0]}
    ids = get_experiments(config=clf_args)
    print(ids)
    clf_model = load_classifier(ids[0])

    # ------------
    # wandb
    # ------------
    os.environ["WANDB_API_KEY"] = '5532aa3f6f581daa33de08ae6bccd7bbdf271c12'  # WAND API KEY HERE
    tags = [
        args.model.prior,
        args.model.dataset_name,
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
    trainer.train(model, clf_model, dataloader, args)


if __name__ == "__main__":
    app.run(cli_main)