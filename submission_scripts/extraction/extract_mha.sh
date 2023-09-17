#!/bin/bash
#SBATCH --output=sbatch_log/%j.out
#SBATCH --gpus=1
#SBATCH --mem-per-cpu=16000
#SBATCH --time 250
source $SCRATCH/miniconda3/etc/profile.d/conda.sh
source activate pytorch-transformer
python -u scripts/extraction/extract_mha.py --batch_size 1400 --dataset_name IWSLT --language_direction E2F --model_name 128emb_20ep --path_to_weights $SCRATCH/pytorch-original-transformer/models/binaries/Transformer_None_None_20.pth  --output_path $SCRATCH/pytorch-original-transformer/mha_outputs "$@"
