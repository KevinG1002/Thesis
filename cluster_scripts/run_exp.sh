#!/bin/bash

# BATCH_SIZE=$1
# EPOCHS=$2
# LR=$3
# NAME_JOBS=$4

#SBATCH --output=logs/%j.out
#SBATCH --gres=gpu:1
#SBATCH --mem=30G
#!SBATCH --time=10:00:00

#! What e-mail address to use for notifications?
#! Insert your mail address here for job notifications
#! Remove the ! to uncomment


source /scratch/kgolan/t_env/bin/activate
echo "Virtual Environment activated"
python -u  /scratch/kgolan/Thesis/experiments/diffusion_training.py 
echo "Deactivating Virtual Environment"