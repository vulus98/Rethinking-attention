#! /bin/bash
if [ $# -ne 1 ]; then
    echo "evaluation_mh_only_submit_all.sh <size in [shrink, small, medium, large]"
    exit
fi
./scripts/create_folder_outputs.sh mha_only $1
for i in {0..5};do
    echo sbatch --output=../sbatch_log/mha_only/$1/training_outputs/ $HOME/pytorch-original-transformer/scripts/training_mh_FF.sh --checkpoints_folder $1/l$i --num_of_curr_trained_layer $i --substitute_class FFNetwork_$1 
    echo ""
    sbatch --output=../sbatch_log/mha_only/$1/training_outputs/%j.out $HOME/pytorch-original-transformer/scripts/training_mh_FF.sh --checkpoints_folder $1/l$i --num_of_curr_trained_layer $i --substitute_class FFNetwork_$1 
    #sbatch $HOME/pytorch-original-transformer/scripts/training_mh_FF.sh --checkpoints_folder $1/l$i --num_of_curr_trained_layer $i --substitute_class FFNetwork_$1 --decoder
done
