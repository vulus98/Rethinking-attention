#!/bin/bash
#SBATCH --output=../sbatch_log/%j.out
#SBATCH --gpus=1
#SBATCH --mem-per-cpu=16000
#SBATCH --time 250
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate pytorch-transformer 
python -u preprocess.py "$@"

#sbatch --output=sbatch_log/%j.out --gpus=1 submission_scripts/sub.sh