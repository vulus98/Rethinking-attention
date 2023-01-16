#!/usr/bin/env bash
#SBATCH --output=../sbatch_log/%j.out
#SBATCH --gpus=1
#SBATCH --mem-per-cpu=4000
#SBATCH --time=300
eval "$(conda shell.bash hook)"
conda activate pytorch-transformer
python -u scripts/averaging/training_script.py --num_of_epochs 20 --batch_size 1400 --dataset_name IWSLT --language_direction E2G