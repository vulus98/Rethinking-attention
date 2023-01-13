#! /bin/bash
if [ $# -ne 1 ]; then
    echo "evaluation_mh_only_submit_all.sh <size in [shrink, small, medium, large]"
    exit
fi

# for i in {0..5}; do
# sbatch  --output=../sbatch_log/mha_only/$1/evaluation_outputs/%j.out  scripts/evaluate.sh --substitute_model_path_d $SCRATCH/models/checkpoints/mha/$1/  --epoch_d 21 --substitute_type_d mha_only --substitute_class_d FFNetwork_$1 --layers_d $i 
# done
# sbatch  --output=../sbatch_log/mha_only/$1/evaluation_outputs/%j.out scripts/evaluate.sh --substitute_model_path_d $SCRATCH/models/checkpoints/mha/$1/  --epoch_d 21 --substitute_type_d mha_only --substitute_class_d FFNetwork_$1 
# sbatch  --output=../sbatch_log/mha_only/$1/evaluation_outputs/%j.out  scripts/evaluate.sh --substitute_model_path_d $SCRATCH/models/checkpoints/mha/$1/  --epoch_d 21 --substitute_type_d mha_only --substitute_class_d FFNetwork_$1 --untrained_d

# for i in {0..5}; do
# sbatch  --output=../sbatch_log/mha_only/$1/evaluation_outputs/%j.out  scripts/evaluate.sh --substitute_model_path $SCRATCH/models/checkpoints/mha/$1/  --epoch 41 --substitute_type mha_only --substitute_class FFNetwork_$1 --layers $i 
# done
# sbatch  --output=../sbatch_log/mha_only/$1/evaluation_outputs/%j.out scripts/evaluate.sh --substitute_model_path $SCRATCH/models/checkpoints/mha/$1/  --epoch 41 --substitute_type mha_only --substitute_class FFNetwork_$1 
# sbatch  --output=../sbatch_log/mha_only/$1/evaluation_outputs/%j.out  scripts/evaluate.sh --substitute_model_path $SCRATCH/models/checkpoints/mha/$1/  --epoch 41 --substitute_type mha_only --substitute_class FFNetwork_$1 --untrained

# Subsittute all
# sbatch  --output=../sbatch_log/all_substituted/%j.out scripts/evaluate.sh --substitute_model_path_d $SCRATCH/models/checkpoints/mha/decoder_shrink/  --epoch_d 41 --substitute_type_d mha_only --substitute_class_d FFNetwork_decoder_shrink  --substitute_model_path $SCRATCH/models/checkpoints/mha/shrink/  --epoch 41 --substitute_type mha_only --substitute_class FFNetwork_shrink 

# Increasing substitution decoder
layers=''
# for i in {0..5}; do
#         layers=$layers' '$i
#         sbatch  --output=../sbatch_log/increasing_layers/decoder/%j.out  scripts/evaluate.sh --substitute_model_path_d $SCRATCH/models/checkpoints/mha/$1/  --epoch_d 41 --substitute_type_d mha_only --substitute_class_d FFNetwork_$1 --layers_d $layers
# done
# layers=''

# for i in {0..5}; do
#     layers=$layers' '$i
#     sbatch  --output=../sbatch_log/increasing_layers/encoder/%j.out  scripts/evaluate.sh --substitute_model_path $SCRATCH/models/checkpoints/mha/$1/  --epoch 41 --substitute_type mha_only --substitute_class FFNetwork_$1 --layers $layers 
# done
