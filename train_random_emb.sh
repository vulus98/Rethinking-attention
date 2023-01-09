#!/usr/bin/env bash
#SBATCH --output=../slurm_log/%j.out
#SBATCH --gpus=1
#SBATCH --mem-per-cpu=4000
#SBATCH --time=300
eval "$(conda shell.bash hook)"
conda activate pytorch-transformer
python -u training_random_embedding.py --num_of_epochs 40 --start_point 20 --batch_size 1400 --dataset_name IWSLT --language_direction E2G
