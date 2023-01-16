#!/usr/bin/env bash
#SBATCH --output=../sbatch_log/%j.out
#SBATCH --mem-per-cpu=16000
#SBATCH --gpus=titan_rtx:1
#SBATCH --time=1440
eval "$(conda shell.bash hook)"
conda activate pytorch-transformer
python -u scripts/averaging/evaluate.py $1
