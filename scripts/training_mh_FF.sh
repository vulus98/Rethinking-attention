#!/bin/bash
#SBATCH --output=../sbatch_log/%j.out
#SBATCH --gpus=1
#SBATCH --mem-per-cpu=16000
#SBATCH --gres=gpumem:20g
#SBATCH --time 180
#SBATCH --mail-user=dcoppola@ethz.ch
#SBATCH --mail-type=ALL
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate pytorch-transformer 
python -u training_mh_FF.py "$@"