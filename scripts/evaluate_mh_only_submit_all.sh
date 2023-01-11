#! /bin/bash
for i in {0..5}; do
sbatch scripts/evaluate.sh --substitute_model_path $SCRATCH/models/checkpoints/mha/medium/  --epoch 41 --substitute_type mha_only --substitute_class FFNetwork_medium --layer $i
done