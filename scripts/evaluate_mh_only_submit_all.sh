#! /bin/bash
if [ $# -ne 1 ]; then
    echo "evaluation_mh_only_submit_all.sh <size in [shrink, small, medium, large]"
    exit
fi

for i in {0..5}; do
sbatch scripts/evaluate.sh --output=../sbatch_log/mha_only/$1/training_outputs/%j.out --substitute_model_path_d $SCRATCH/models/checkpoints/mha/$1/  --epoch_d 21 --substitute_type_d mha_only --substitute_class_d FFNetwork_$1 --layers_d $i 
done
sbatch scripts/evaluate.sh --output=../sbatch_log/mha_only/$1/training_outputs/%j.out --substitute_model_path_d $SCRATCH/models/checkpoints/mha/$1/  --epoch_d 21 --substitute_type_d mha_only --substitute_class_d FFNetwork_$1 
sbatch scripts/evaluate.sh --output=../sbatch_log/mha_only/$1/training_outputs/%j.out --substitute_model_path_d $SCRATCH/models/checkpoints/mha/$1/  --epoch_d 21 --substitute_type_d mha_only --substitute_class_d FFNetwork_$1 --untrained_d

# python3 validation_script.py --model_name transformer_128.pth --substitute_class_d FFNetwork_decoder_shrink128 --layers_d 0 --untrained_d
