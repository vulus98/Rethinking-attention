#!/usr/bin/env bash
#SBATCH --output=../slurm_log/%j.out
#SBATCH --mem-per-cpu=16000
#SBATCH --gres=gpumem:11g
#SBATCH --gpus=gtx_1080_ti:1
#SBATCH --time=1440
eval "$(conda shell.bash hook)"
conda activate pytorch-transformer
python -u single_sim.py --input 0 --output norm
