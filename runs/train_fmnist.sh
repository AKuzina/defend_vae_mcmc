#!/bin/bash

# Job requirements:
#SBATCH -N 1
#SBATCH -t 0-10:00:00
#SBATCH -p gpu_titanrtx_shared
##SBATCH -p gpu_shared
#SBATCH --gres=gpu:1
#SBATCH -n 4
module load 2019  #pre2019
module load Python/3.7.5-foss-2019b
module load Miniconda3
module load CUDA/10.1.243

source activate ckconv_vae
cp -R $HOME/VAE/ckconv_vae "$TMPDIR"
dir "$TMPDIR"
# run python file
export PYTHONPATH=.


for b in 2 5 10
do
python run_experiment.py \
            --config.debug=False\
            --config.iter=0\
            --config.batch_size=256\
            --config.test_batch_size=1024\
            --config.max_epochs=200\
            --config.dataset_name='fashion_mnist'\
            --config.lr=0.0005\
            --config.lr_patience=10\
            --config.lr_factor=0.5\
            --config.num_ch=32\
            --config.z_dim=64\
            --config.prior="standard"\
            --config.likelihood='bernoulli'\
            --config.beta=$b \
            --config.warmup=0 \
            --config.is_k=1000 \
            --config.latent_long=True\
            --config.model='conv'
done
#            --config.HMC_steps=8\
#            --config.HMC_burn_in=2


