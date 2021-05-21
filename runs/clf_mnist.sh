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

for b in 1 2 5 10
do
python run_classifier.py \
            --config.model.dataset_name='fashion_mnist'\
            --config.model.model='conv'\
            --config.model.prior="standard"\
            --config.model.num_ch=32\
            --config.model.iter=0\
            --config.model.batch_size=256\
            --config.model.test_batch_size=1024\
            --config.model.max_epochs=200\
            --config.model.z_dim=64\
            --config.model.beta=$b\
            --config.model.warmup=0 \
            --config.model.lr=0.0005\
            --config.model.lr_patience=10\
            --config.model.lr_factor=0.5\
            --config.model.likelihood='bernoulli'\
            --config.model.is_k=1000 \
            --config.model.latent_long=True\
            --config.classifier.lr=5e-3\
            --config.classifier.max_epoch=100
done




