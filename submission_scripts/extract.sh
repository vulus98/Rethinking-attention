#!/usr/bin/env bash
#SBATCH --output=../sbatch_log/%j.out
#SBATCH --mem-per-cpu=16000
#SBATCH --gres=gpumem:20g
#SBATCH --gpus=1
#SBATCH --time=240
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate pytorch-transformer
python3 -u scripts/extraction/extract.py --path_to_weights $HOME/pytorch-original-transformer/models/binaries/transformer_128.pth --batch_size 1400 --dataset_name IWSLT --language_direction E2G --model_name 128emb_20ep 
