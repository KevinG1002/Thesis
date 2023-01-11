#!/bin/bash
#SBATCH --output=/scratch_net/bmicdl03/kgolan/Thesis/experiments/experimental_results/%j.out
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
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
python -u  /scratch_net/bmicdl03/kgolan/Thesis/experiments/snapshot_training.py -e 10 --M_snapshots 5 --save_every 2 -lr 0.01
echo "Deactivating Virtual Environment"
conda deactivate