from ml_collections import ConfigDict
from ml_collections.config_dict import config_dict


def get_config(type):
    default_config = dict(
        ### EXPERIMENT
        debug=False,
        # iter number
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
        # Reduce learning rate by 'lr_factor' when validation loss does not fall for 'lr_patience' epochs
        lr_factor=config_dict.placeholder(float),
        lr_patience=config_dict.placeholder(float),

        ### NNs
        # 'conv' - for VAE and \beta-VAE,
        # 'tc_conv' - for \beta-TCVAE
        model=config_dict.placeholder(str),

        # \beta for \beta-VAE and \beta-TCVAE
        beta=config_dict.placeholder(float),

        # if true, use latent code without spacial dimention
        latent_long=config_dict.placeholder(bool),

        # number of channel in the first layer of the encoder and decoder (to be doubled twice)
        num_ch=config_dict.placeholder(int),

        ### VAE
        # latent size
        z_dim=config_dict.placeholder(int),

        # prior: standard, realnvp
        prior=config_dict.placeholder(str),

        # type of the loglikelihood: 'bernoulli', 'gaussian'
        likelihood=config_dict.placeholder(str),

        # If > 0, increase beta uniformly from 0 for 'warmup' epochs
        warmup=config_dict.placeholder(int),

        # Number of IS for log-likelihood estimation on the test dataset
        is_k=config_dict.placeholder(int),
    )

    attack_config = dict(
        # supervised for attack with target and unsupervised for attack without target
        type='supervised',
        # number of reference points
        N_ref=10,

        # Loass type: ['kl_forward', 'kl_reverse', 'skl', 'means', 'clf']
        #   kl_forward = KL(q || q_a)
        #   kl_reverse = KL(q_a || q)
        #   means = ||mu_1 - mu_2||^2
        #   clf = CE(z, y)
        loss_type='skl',

        # Attack raduis
        eps_norm=1.,

        # L_p norm for attack radius. options: '1', '2', 'inf'
        p='inf',

        # Learning rate for an attack
        lr=1e-2,

        # Number of HMC steps during attack construction
        hmc_steps_attack=0,

        # Number of HMC steps during defence
        hmc_steps=0,

        # Steps size for HMC
        hmc_eps=0.02,

        # for attack with target (supervised). Number of target points
        N_trg=5
    )

    clf_config = dict(
        lr=1e-3,
        max_epoch=100
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
    }[type]
    return default_config
