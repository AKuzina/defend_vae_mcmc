from ml_collections import ConfigDict
from ml_collections.config_dict import config_dict


def get_config(type):
    default_config = dict(
        ### EXPERIMENT
        debug=False,
        # iter number to evaluate std
        iter=0,
        # input batch size for training and testing
        batch_size=config_dict.placeholder(int),
        test_batch_size=config_dict.placeholder(int),
        max_epochs=config_dict.placeholder(int),
        gpus=config_dict.placeholder(int),
        seed=config_dict.placeholder(int),
        dataset_name=config_dict.placeholder(str),
        # If load pretrained model
        resume=config_dict.placeholder(bool),

        ### OPTIMIZATION
        # learning rate (initial if scheduler is used)
        lr=config_dict.placeholder(float),
        lr_factor=config_dict.placeholder(float),
        lr_patience=config_dict.placeholder(float),

        ### NNs
        # conv (later hierarcy, or other models)'
        model=config_dict.placeholder(str),
        # if true, use latent code without spacial dimention
        latent_long=config_dict.placeholder(bool),
        # number of channel in the first layer of the encoder and decoder (to be doubled twice)
        num_ch=config_dict.placeholder(int),

        ### VAE
        # latent size
        z_dim=config_dict.placeholder(int),

        # prior: standard, realnvp
        prior=config_dict.placeholder(str),
        # type of the loglikelihood for continuous input: logistic, normal, l1
        likelihood=config_dict.placeholder(str),
        beta=config_dict.placeholder(float),
        warmup=config_dict.placeholder(int),
        is_k=config_dict.placeholder(int),

        ### VCD
        # HMC_steps=0,
        # HMC_burn_in=0
    )

    attack_config = dict(
        # supervised for attack with target and unsupervised for attack without target
        type='supervised',
        N_ref=10,
        # ['kl_forward', 'kl_reverse', 'skl', 'means']
        #   kl_forward = KL(q || q_a)
        #   kl_reverse = KL(q_a || q)
        #   means = ||mu_1 - mu_2||^2
        loss_type='skl',
        # ['projection', 'penalty']
        #   projection - make projection after each gradient step
        #   penalty - add regularization to the loss
        reg_type='projection',
        eps_norm=1.,
        p='inf',  # other options: '1', '2' for l1 and l2 norms
        lr=1e-2,  # Initial Learning rate for an attack
        # For vae + hmc: num steps and step size
        hmc_steps=0,
        hmc_eps=0.02,

        # for attack with target (supervised)
        N_trg=5
    )

    clf_config = dict(
        lr=1e-3,
        max_epoch=100
    )

    nvae_config = dict(
        dset='celeba_64',
        connect=1,
        temp=0.8,
    )

    default_config = {
        'train': ConfigDict(default_config),
        'attack_vae': ConfigDict({
            'model': default_config,
            'attack': ConfigDict(attack_config),
            'mode': type
        }),
        'clf': ConfigDict({
            'model': default_config,
            'classifier': ConfigDict(clf_config),
            'mode': type
        }),
        'attack_nvae': ConfigDict({
            'model': ConfigDict(nvae_config),
            'attack': ConfigDict(attack_config),
            'mode': type
        }),
        'clf_nvae': ConfigDict({
            'model': ConfigDict(nvae_config),
            'classifier': ConfigDict(clf_config),
            'mode': type
        })
    }[type]
    return default_config
