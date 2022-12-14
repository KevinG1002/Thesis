#!/bin/bash

BATCH_SIZE=$1
EPOCHS=$2
LR=$3
NAME_JOBS=$4

#SBATCH  --output=log/%j.out
#SBATCH  --gres=gpu:1
#SBATCH  --mem=30G
#SBATCH --time=10:00:00

#! What e-mail address to use for notifications?
#! Insert your mail address here for job notifications
#! Remove the ! to uncomment


source ~/Documents/t_env/activate
python -u training_good.py "$@"
echo "Deactivating Virtual Environment"