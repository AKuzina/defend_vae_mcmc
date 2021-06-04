#!/bin/bash

# # Job requirements:
# #SBATCH -N 1
# #SBATCH -t 0-11:00:00
# #SBATCH -p gpu_titanrtx_shared
# ##SBATCH -p gpu_shared
# #SBATCH --gres=gpu:1
# #SBATCH -n 4
# module load 2019  #pre2019
# module load Python/3.7.5-foss-2019b
# module load Miniconda3
# module load CUDA/10.1.243

# source activate ckconv_vae
# cp -R $HOME/VAE/ckconv_vae "$TMPDIR"
# dir "$TMPDIR"
# # run python file
# export PYTHONPATH=.

for h in 40 80
do
# for lbd in 20 25 30 35
# do
for eps in 7
do
python run_attack.py \
       --config.model.dataset_name='celeba'\
            --config.model.model='conv'\
            --config.model.prior="realnvp"\
            --config.model.num_ch=64\
            --config.model.iter=0\
            --config.model.batch_size=1024\
            --config.model.test_batch_size=1024\
            --config.model.max_epochs=250\
            --config.model.z_dim=128\
            --config.model.beta=1\
            --config.model.warmup=0 \
            --config.model.lr=0.002\
            --config.model.lr_patience=10\
            --config.model.lr_factor=0.25\
            --config.model.likelihood='logistic'\
            --config.model.is_k=1000 \
            --config.model.latent_long=True\
            --config.attack.N_ref=50\
            --config.attack.N_adv=6\
            --config.attack.eps_norm=$eps\
            --config.attack.reg_type='projection'\
            --config.attack.use_perp=0.\
            --config.attack.type='unsupervised'\
            --config.attack.loss_type='skl'\
            --config.attack.hmc_steps=$h\
            --config.attack.hmc_eps=0.04
done
done


