#!/bin/bash
#SBATCH --output=/scratch_net/bmicdl03/kgolan/Thesis/cluster_scripts/logs/%j.out
#SBATCH --gres=gpu:1
#SBATCH --mem=8G

# cd /scratch_net/bmicdl03/kgolan/Thesis
# source /scratch_net/bmicdl03/kgolan/conda/etc/profile.d/conda.sh
source /scratch_net/bmicdl03/kgolan/venv/bin/activate
# conda activate /scratch_net/bmicdl03/kgolan/conda_envs/t_env

echo "Virtual Environment activated"
python -u  /scratch_net/bmicdl03/kgolan/Thesis/experiments/diffusion_training.py 
echo "Deactivating Virtual Environment"
# conda deactivate
deactivate