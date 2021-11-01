#!/bin/bash

## Job requirements:
#SBATCH -N 1
#SBATCH -t 0-10:00:00
#SBATCH -p gpu_titanrtx_shared
##SBATCH -p gpu_titanrtx
##SBATCH -p gpu_shared
##SBATCH -p gpu_short
#SBATCH --gres=gpu:1
#SBATCH -n 4
module load 2019  #pre2019
module load Python/3.7.5-foss-2019b
module load Miniconda3
module load CUDA/10.1.243

source activate ckconv_vae
#cp -R $HOME/VAE/ckconv_vae "$TMPDIR"
#dir "$TMPDIR"
 # run python file
#export PYTHONPATH=.

for h in 20
do
for n in 2
do
python run_attack_nvae.py \
            --config.model.connect=$n\
            --config.model.dset='mnist'\
            --config.model.temp=0.6\
            --config.attack.N_ref=2\
            --config.attack.N_trg=5\
            --config.attack.reg_type='projection'\
            --config.attack.eps_norm=0.3\
            --config.attack.type='supervised'\
            --config.attack.loss_type='skl'\
            --config.attack.hmc_steps=$h\
            --config.attack.hmc_eps=1e-6\
            --config.attack.lr=0.2\
            --config.attack.p='inf'

done
done


