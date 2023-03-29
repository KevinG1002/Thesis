#!/bin/bash
#SBATCH --output=/scratch_net/bmicdl03/kgolan/Thesis/experiments/experimental_results/logs/%j.out
#SBATCH --gres=gpu:1
#!SBATCH --mem=12G
#! What e-mail address to use for notifications?
#! Insert your mail address here for job notifications
#! Remove the ! to uncomment

#SBATCH --mail-user=kgolan@student.ethz.ch

#! What types of email messages do you wish to receive?
#! Remove the ! to uncomment
#SBATCH --mail-type=ALL

source /scratch_net/bmicdl03/kgolan/conda/etc/profile.d/conda.sh
cd /scratch_net/bmicdl03/kgolan/Thesis/
conda activate m_env
# conda activate gvae_env
# pip install -e .
cd /scratch_net/bmicdl03/kgolan/Thesis/experiments/
# echo "Virtual Environment activated"
# python -u  /scratch_net/bmicdl03/kgolan/Thesis/experiments/svi_on_mlp.py -lr 2e-2 -b 300 --n_its 10000 --n_samples 20 
# python -u /scratch_net/bmicdl03/kgolan/Thesis/utils/model_dataset_gen.py -lr 0.001 -e 10 -b 32 --n_runs 2000
# python -u /scratch_net/bmicdl03/kgolan/Thesis/datasets/graph_dataset.py
# python -u  /scratch_net/bmicdl03/kgolan/Thesis/experiments/gvae_training.py -e 200 -b 8 -lr 1e-4 --save_every 5
# python -u  /scratch_net/bmicdl03/kgolan/Thesis/experiments/diffusion_training.py -e 300 -b 4 -lr 2e-6 --save_every 5 --n_steps 1000

echo "Deactivating Virtual Environment"
conda deactivate