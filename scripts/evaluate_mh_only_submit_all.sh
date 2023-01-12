#! /bin/bash
if [ $# -ne 1 ]; then
    echo "evaluation_mh_only_submit_all.sh <size in [shrink, small, medium, large]"
    exit
fi

for i in {0..5}; do
    sbatch scripts/evaluate.sh --substitute_model_path $SCRATCH/models/checkpoints/mha/$1/  --epoch 41 --substitute_type mha_only --substitute_class FFNetwork_$1 --layers $i 
done
sbatch scripts/evaluate.sh --substitute_model_path $SCRATCH/models/checkpoints/mha/$1/  --epoch 41 --substitute_type mha_only --substitute_class FFNetwork_$1 
sbatch scripts/evaluate.sh --substitute_model_path $SCRATCH/models/checkpoints/mha/$1/  --epoch 41 --substitute_type mha_only --substitute_class FFNetwork_$1 --untrained

