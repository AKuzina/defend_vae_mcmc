#!/bin/bash

# # Job requirements:
# #SBATCH -N 1
# #SBATCH -t 0-10:00:00
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

for n in 6
do
python run_classifier_nvae.py \
            --config.model.connect=$n\
            --config.classifier.lr=1e-3\
            --config.classifier.max_epoch=50
done


