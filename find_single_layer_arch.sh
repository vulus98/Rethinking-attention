#!/usr/bin/env bash
#SBATCH --output=../slurm_log/%j.out
#SBATCH --mem-per-cpu=32000
#SBATCH --gpus=1
#SBATCH --gres=gpumem:11g
#SBATCH --time=1440
eval "$(conda shell.bash hook)"
conda activate pytorch-transformer
python -u simulator_train.py --input 0 --output 0
