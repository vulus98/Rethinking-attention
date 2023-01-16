#! /bin/bash
if [ $# -ne 1 ]; then
    echo "evaluation_mh_only_submit_all.sh <size in [ FFNetwork_shrink,  FFNetwork_small,  FFNetwork_shrink256,  FFNetwork_shrink128]"
    echo "add --decoder in the for loop if you want to train a layer of the decoder"
    exit
fi
./submission_scripts/utils/create_folder_outputs.sh mha_only $1
for i in {0..5};do
    echo sbatch --output=../sbatch_log/mha_only/$1/training_outputs/ submission_scripts/full_sentence/training_mha_only_FF.sh --num_of_curr_trained_layer $i --substitute_class $1
    echo ""
    sbatch --output=../sbatch_log/mha_only/$1/training_outputs/%j.out  submission_scripts/training_mha_only_FF.sh  --num_of_curr_trained_layer $i --substitute_class $1
    #sbatch $HOME/pytorch-original-transformer/scripts/training_mh_FF.sh --checkpoints_folder $1/l$i --num_of_curr_trained_layer $i --substitute_class FFNetwork_$1 --decoder
done
