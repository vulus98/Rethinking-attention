#!/usr/bin/env bash
#SBATCH --output=../slurm_log/%j.out
#SBATCH --mem-per-cpu=16000
#SBATCH --gpus=1
#SBATCH --gres=gpumem:11g
#SBATCH --time=240
#SBATCH --exclude=eu-g4-[001-012,015-019,020-033]
eval "$(conda shell.bash hook)"
conda activate pytorch-transformer
python -u sim_all_together.py
