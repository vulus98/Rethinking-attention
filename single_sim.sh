#!/usr/bin/env bash
#SBATCH --output=../slurm_log/%j.out
#SBATCH --mem-per-cpu=16000
#SBATCH --gpus=1
#SBATCH --gres=gpumem:11g
#SBATCH --time=1440
eval "$(conda shell.bash hook)"
conda activate pytorch-transformer
python -u single_sim.py --input 0 --output 5
