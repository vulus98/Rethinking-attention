#!/bin/bash
#SBATCH --output=../sbatch_log/%j.out
#SBATCH --gpus=2
#SBATCH --mem-per-cpu=4000
#SBATCH --time 250
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate pytorch-transformer 
python -u translation_script.py --model_name transformer_128.pth --source_sentence "Hey, what is the weather like today?" "$@"

#sbatch --output=sbatch_log/%j.out --gpus=1 submission_scripts/sub.sh