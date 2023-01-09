#!/bin/bash
#SBATCH --output=../sbatch_log/%j.out
#SBATCH --gpus=1
#SBATCH --mem-per-cpu=16000
#SBATCH --time 250
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate pytorch-transformer 
python -u extract_mha.py --batch_size 1400 --dataset_name IWSLT --language_direction E2G --model_name 128emb_20ep --path_to_weights $HOME/pytorch-original-transformer/models/binaries/transformer_128.pth
 "$@"