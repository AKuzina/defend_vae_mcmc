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
        # if true, use latent code without spacial dimention
        latent_long=False,
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
        eps_norm=1.,
        # ['kl_forward', 'kl_reverse', 'skl', 'means']
        # kl_forward = KL(q || q_a)
        # kl_reverse = KL(q_a || q)
        # means = ||mu_1 - mu_2||^2
        loss_type='skl',
        # ['projection', 'penalty']
        # projection - make projection after each gradient step
        # penalty - add regularization to the loss
        reg_type='projection',
        # weight of the penalty (if relevent)
        lbd=50,
        # Wheather to use perceptual loss as a regularization
        use_perp=0.,
        # For vae + hmc: num steps and step size
        hmc_steps=0,
        hmc_eps=0.02,

        # for unsupervised
        N_adv=0,

        # for attack with target (supervised)
        N_trg=5,
    )

    clf_config = dict(
        lr=1e-3,
        max_epoch=100
    )
    
    nvae_config = dict(
        chckpt_path='../NVAE/checkpoint/celeba_64.pt',
        dset_path='../NVAE/datasets/celeba64_lmdb',
        connect=1,
        temp=0.8,
#         n_samples=5,
#         use_perp=True,
        lr=5e-3,
    )


    default_config = {
        'train': ConfigDict(default_config),
        'attack': ConfigDict(
            {'model': default_config,
             'attack': ConfigDict(attack_config)}),
        'clf': ConfigDict(
            {'model': default_config,
             'classifier': ConfigDict(clf_config)}),
        'nvae': ConfigDict(
            {'model': ConfigDict(nvae_config),
             'attack': ConfigDict(attack_config)}),
        'clf_nvae': ConfigDict(
            {'model': ConfigDict(nvae_config),
             'classifier': ConfigDict(clf_config)})
    }[type]

    return default_config