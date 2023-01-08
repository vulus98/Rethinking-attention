#!/bin/bash
#SBATCH --output=sbatch_log/%j.out
#SBATCH --gpus=1
#SBATCH --mem-per-cpu=4000
#SBATCH --time 250
source /cluster/home/vbozic/miniconda3/etc/profile.d/conda.sh
conda activate pytorch-transformer 
python -u validation_script.py "$@"
