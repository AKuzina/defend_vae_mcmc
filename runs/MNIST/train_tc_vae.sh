#!/bin/bash

for beta in 2 5 10
do
python run_experiment.py \
           --config.beta=$beta \
           --config.model=tc_conv \
           --config.debug=False \
           --config.iter=0 \
           --config.batch_size=128 \
           --config.test_batch_size=1024 \
           --config.max_epochs=300 \
           --config.dataset_name=mnist \
           --config.lr=0.0005 \
           --config.lr_patience=10 \
           --config.lr_factor=0.5 \
           --config.num_ch=32 \
           --config.z_dim=64 \
           --config.prior=standard\
           --config.likelihood=bernoulli\
           --config.warmup=0\
           --config.is_k=1000\
           --config.latent_long=True\
           --config.gpus=1
done

