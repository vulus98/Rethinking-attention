#!/usr/bin/env bash
#SBATCH --output=../slurm_log/%j.out
#SBATCH --mem-per-cpu=32000
#SBATCH --gpus=1
#SBATCH --time=30
eval "$(conda shell.bash hook)"
conda activate pytorch-transformer
PREFIX=$SCRATCH"/layer_outputs/128emb_20ep_IWSLT_E2G_"
python -u simulator_train.py --num_of_epochs 20 --batch_size 1400 --train_input $PREFIX"layer0_inputs_train" --train_output $PREFIX"layer0_outputs_train" --val_input $PREFIX"layer0_inputs_val" --val_output $PREFIX"layer0_outputs_val" --train_mask $PREFIX"masks_train" --val_mask $PREFIX"masks_val"
