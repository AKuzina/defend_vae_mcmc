import wandb
import os
from ml_collections import ConfigDict
from vae.model.vae import StandardVAE
from datasets import load_dataset

USER = ''  # wandb user name
PROJECT = ''  # wandb project name
API_KEY = ''  # wandb api key (required to login)
api = wandb.Api()


def load_model(idx):
    best_model = wandb.restore('checkpoints/last.ckpt',
                               run_path=os.path.join(USER, PROJECT, idx),
                               replace=True, root='wandb')
    # load model from the saved checkpoint
    vae = StandardVAE.load_from_checkpoint(checkpoint_path=best_model.name)
    return vae

def load_data(idx, mode='test'):
    run = api.run(os.path.join(USER, PROJECT, idx))
    data_module, _ = load_dataset(ConfigDict(run.config), binarize=False)
    data_module.prepare_data()
    if mode == 'test':
        data_module.setup('test')
        return data_module.test_dataloader()
    elif mode == 'train':
        data_module.setup('fit')
        return data_module.train_dataloader()


def check_run(run, name, config):
    if name is None or run.name == name:
        if config is None:
            return True
        for k, v in zip(config, config.values()):
            if not (k in run.config.keys()):
                return False
            if run.config[k] != v:
                return False
        return True


def get_experiments(name=None, config=None):
    """
    Return ids of all the expriments based on the name and/or configuration
    """
    runs = api.runs(os.path.join(USER, PROJECT))
    run_list = []
    for run in runs:
        if check_run(run, name, config):
            run_list.append(run.id)
    if len(run_list) != 1:
        print(f"Found {len(run_list)} runs")
    return run_list


def creat_attack_conf(CONF_model, CONF_attack, CONF_other={}):
    conf = {}
    for k in CONF_attack:
        conf['config.attack.{}'.format(k)] = CONF_attack[k]
    for k in CONF_model:
        conf['config.model.{}'.format(k)] = CONF_model[k]
    conf.update(CONF_other)
    return conf
