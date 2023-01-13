#!/usr/bin/env bash
#SBATCH --output=../slurm_log/%j.out
#SBATCH --mem-per-cpu=32000
#SBATCH --gpus=1
#SBATCH --gres=gpumem:11g
#SBATCH --time=240
eval "$(conda shell.bash hook)"
conda activate pytorch-transformer
python -u sim_all_pretrain.py
