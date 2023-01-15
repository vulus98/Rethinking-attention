#!/usr/bin/env bash
#SBATCH --output=../slurm_log/%j.out
#SBATCH --mem-per-cpu=4000
#SBATCH --gres=gpumem:11g
#SBATCH --gpus=gtx_1080_ti:1
#SBATCH --time=240
eval "$(conda shell.bash hook)"
conda activate pytorch-transformer
python -u extract.py --batch_size 1400 --dataset_name IWSLT --language_direction E2G --model_name 128emb_20ep
