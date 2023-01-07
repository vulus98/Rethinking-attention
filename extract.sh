#!/usr/bin/env bash
#SBATCH --output=../slurm_log/%j.out
#SBATCH --gpus=1
#SBATCH --mem-per-cpu=4000
#SBATCH --time=240
eval "$(conda shell.bash hook)"
conda activate pytorch-transformer
python -u extract.py --batch_size 1400 --dataset_name IWSLT --language_direction E2G --model_name 128emb_20ep --path_to_weights ~/transformer_ckpt_epoch_20.pth
