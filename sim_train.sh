#!/usr/bin/env bash
#SBATCH --output=../slurm_log/%j.out
#SBATCH --mem-per-cpu=32000
#SBATCH --gpus=1
#SBATCH --gres=gpumem:11g
#SBATCH --time=720
eval "$(conda shell.bash hook)"
conda activate pytorch-transformer
PREFIX=$SCRATCH"/layer_outputs/128emb_20ep_IWSLT_E2G_"
python -u simulator_train.py --num_of_epochs 500 --batch_size 1024 --checkpoint_freq 20 --input $PREFIX"layer0_inputs" --output $PREFIX"layer5_outputs" --mask $PREFIX"masks"
