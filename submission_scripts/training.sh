#!/bin/bash
#SBATCH --output=sbatch_log/%j.out
#SBATCH --gpus=4
#SBATCH --mem-per-cpu=16000
#SBATCH --gres=gpumem:20g
#SBATCH --time 1000
source /cluster/home/vbozic/miniconda3/etc/profile.d/conda.sh
conda activate pytorch-transformer 
python -u training_script.py "$@"
