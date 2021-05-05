from ml_collections import ConfigDict


def get_config(type):
    default_config = dict(
        ### EXPERIMENT
        debug=False,
        # iter number to evaluate std
        iter=0,
        # input batch size for training and testing
        batch_size=500,
        test_batch_size=500,
        max_epochs=100,
        gpus=1,
        seed=14,
        dataset_name='mnist',
        # If load pretrained model
        resume=False,

        ### OPTIMIZATION
        # learning rate (initial if scheduler is used)
        lr=5e-4,
        lr_factor=0.9,
        lr_patience=10,

        ### NNs
        # conv (later hierarcy, or other models)'
        model='conv',
        # number of channel in the first layer of the encoder and decoder (to be doubled twice)
        num_ch=0,

        ### VAE
        # latent size
        z_dim=40,

        # prior: standard, realnvp
        prior='standard',
        # type of the loglikelihood for continuous input: logistic, normal, l1
        likelihood='bernoulli',
        beta=1.,
        warmup=0,
        is_k=1000,

        ### VCD
        HMC_steps=0,
        HMC_burn_in=0
    )

    attack_config = dict(
        # unsupervised for chain and supervised for attack with target
        type='supervised',
        N_ref=10,

        # for chain
        N_chains=0,
        N_adv=0,

        # for attack with target
        N_trg=5,
        eps_reg=0,
    )

    default_config = {
        'train': ConfigDict(default_config),
        'attack': ConfigDict({'model': default_config,
                             'attack': ConfigDict(attack_config)}),
    }[type]

    return default_config