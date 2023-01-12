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
conda activate m_env
cd /scratch_net/bmicdl03/kgolan/Thesis/
pip install -e .
cd /scratch_net/bmicdl03/kgolan/Thesis/experiments/
echo "Virtual Environment activated"
python -u  /scratch_net/bmicdl03/kgolan/Thesis/experiments/diffusion_training.py 
# python -u /scratch_net/bmicdl03/kgolan/Thesis/utils/model_dataset_gen.py -lr 0.001 -e 10 -b 64 --n_runs 750
echo "Deactivating Virtual Environment"
conda deactivate