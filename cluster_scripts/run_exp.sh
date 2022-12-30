#!/bin/bash
#SBATCH --output=/scratch_net/bmicdl03/kgolan/Thesis/cluster_scripts/logs/%j.out
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#! What e-mail address to use for notifications?
#! Insert your mail address here for job notifications
#! Remove the ! to uncomment

#SBATCH --mail-user=kgolan@student.ethz.ch

#! What types of email messages do you wish to receive?
#! Remove the ! to uncomment
#SBATCH --mail-type=ALL

source /scratch_net/bmicdl03/kgolan/venv/bin/activate
cd /scratch_net/bmicdl03/kgolan/Thesis
pip install -e .
echo "Virtual Environment activated"
python -u  /scratch_net/bmicdl03/kgolan/Thesis/experiments/diffusion_training.py 
echo "Deactivating Virtual Environment"
deactivate