import os
import sys
import wandb
import pytorch_lightning as pl
import copy

from datasets import load_dataset
from utils.wandb import get_experiments, load_model, load_classifier, creat_attack_conf
from attack import trainer

# args
from absl import app
from absl import flags
from ml_collections.config_flags import config_flags

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", default="config.py:attack_vae")


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
    assert args.attack.loss_type in ['skl', 'kl_forward', 'kl_reverse', 'means']
    assert args.attack.reg_type in ['penalty', 'projection']

    # ------------
    # load pretrained model
    # ------------
    model_req = {k: v for k, v in dict(args.model).items() if v is not None}
    ids = get_experiments(config=model_req)
    print(ids)
    model = load_model(ids[0])
    model.eval()
    with args.unlocked():
        args['model_id'] = ids[0]

    # classifier
    N = 1
    if args.model.dataset_name == 'celeba':
        N = len(data_module.task_ids)
    clf_model = load_classifier(args['model_id'], N=N)

    # load attacks, if already trained
    CONF_attack = dict(args.attack)
    CONF_attack['hmc_steps'] = 0
    CONF_attack.pop('hmc_eps')
    print('Looking for an attack:')
    ids = get_experiments(config=creat_attack_conf({}, CONF_attack,
                                                   {'model_id': args.model_id,
                                                    'mode': 'attack_vae'}))
    attack_id = None
    if len(ids) > 0:
        print('Attack already found:')
        print(ids[0])
        attack_id = ids[0]
    with args.unlocked():
        args.attack.attack_id = attack_id

    # ------------
    # data
    # ------------
    args.model.batch_size = 1024
    args.model.test_batch_size = 1024
    data_module, args.model = load_dataset(args.model, binarize=False)
    data_module.setup('test')
    dataloader = data_module.test_dataloader()


    # ------------
    # wandb
    # ------------
    os.environ["WANDB_API_KEY"] = '5532aa3f6f581daa33de08ae6bccd7bbdf271c12'  # WAND API KEY HERE
    tags = [
        args.mode,
        args.model.prior,
        args.model.dataset_name,
        args.attack.type,
        args.attack.loss_type,
        args.attack.reg_type
    ]

    wandb.init(
        project="vcd_vae",
        tags=tags,
        entity='anna_jakub',  # USER NAME HERE\
        config=copy.deepcopy(dict(args))
    )
    wandb.config.update(flags.FLAGS)

    # run attack
    trainer.train(model, clf_model, dataloader, args)


if __name__ == "__main__":
    app.run(cli_main)