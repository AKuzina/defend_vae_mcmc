#!/bin/bash

# Job requirements:
#SBATCH -N 1
#SBATCH -t 0-11:00:00
#SBATCH -p gpu_titanrtx_shared
##SBATCH -p gpu_shared
#SBATCH --gres=gpu:1
#SBATCH -n 4
module load 2019  #pre2019
module load Python/3.7.5-foss-2019b
module load Miniconda3
module load CUDA/10.1.243

source activate ckconv_vae
#cp -R $HOME/VAE/ckconv_vae "$TMPDIR"
#dir "$TMPDIR"
## run python file
#export PYTHONPATH=.

for h in 0
do
for b in 1
do
for loss in  'skl' #'means' 'kl_forward' 'kl_reverse' 'skl'
do
python run_attack.py \
            --config.model.dataset_name='mnist'\
            --config.model.model='conv'\
            --config.model.prior="standard"\
            --config.model.num_ch=32\
            --config.model.z_dim=64\
            --config.model.beta=$b\
            --config.attack.N_ref=50\
            --config.attack.reg_type='projection'\
            --config.attack.eps_norm=0.25\
            --config.attack.p='inf'\
            --config.attack.lr=1\
            --config.attack.type='supervised'\
            --config.attack.loss_type=$loss\
            --config.attack.N_trg=10\
            --config.attack.hmc_steps=$h\
            --config.attack.hmc_eps=0.03
done
done
done

#
#--config.model.iter=0\
#            --config.model.batch_size=256\
#            --config.model.test_batch_size=1024\
#            --config.model.max_epochs=200\
#--config.model.warmup=0 \
#            --config.model.is_k=1000 \
