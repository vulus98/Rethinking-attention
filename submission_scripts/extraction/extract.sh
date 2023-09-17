#!/usr/bin/env bash
#SBATCH --output=sbatch_log/%j.out
#SBATCH --mem-per-cpu=16000
#SBATCH --gres=gpumem:20g
#SBATCH --gpus=1
#SBATCH --time=240
source $SCRATCH/miniconda3/etc/profile.d/conda.sh
source activate pytorch-transformer
python3 -u scripts/extraction/extract.py --path_to_weights $SCRATCH/pytorch-original-transformer/models/binaries/Transformer_None_None_20.pth --batch_size 1400 --dataset_name IWSLT --language_direction E2F --model_name 128emb_20ep