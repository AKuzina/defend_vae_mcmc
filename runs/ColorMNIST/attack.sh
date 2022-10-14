#!/bin/bash

# attack VAE
for radius in 0.1 0.2
do
python run_attack.py \
            --config.model.dataset_name='color_mnist'\
            --config.model.model='conv'\
            --config.model.prior="standard"\
            --config.model.num_ch=32\
            --config.model.z_dim=64\
            --config.model.beta=1\
            --config.attack.N_ref=50\
            --config.attack.eps_norm=$radius\
            --config.attack.p='inf'\
            --config.attack.lr=1.\
            --config.attack.type='unsupervised'\
            --config.attack.loss_type='skl'\
            --config.attack.hmc_steps=0\
            --config.attack.hmc_eps=0.0 \
            --config.attack.hmc_steps_attack=0
done

# attack beta-VAE and beta-TCVAE
for model in 'conv' 'tc_conv'
do
for b in 2 5 10
do
for radius in 0.1 0.2
do
python run_attack.py \
            --config.model.dataset_name='color_mnist'\
            --config.model.model=$model\
            --config.model.prior="standard"\
            --config.model.num_ch=32\
            --config.model.z_dim=64\
            --config.model.beta=$b\
            --config.attack.N_ref=50\
            --config.attack.eps_norm=$radius\
            --config.attack.p='inf'\
            --config.attack.lr=1.\
            --config.attack.type='unsupervised'\
            --config.attack.loss_type='skl'\
            --config.attack.hmc_steps=0\
            --config.attack.hmc_eps=0.0 \
            --config.attack.hmc_steps_attack=0
done
done
done