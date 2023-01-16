#! /bin/bash
if [ $# == 0 ]; then
    echo ""
    echo "submission_scripts/full_sentence/mha_only_FF_submit_all.sh <architecture_name> [--decoder]?"
    echo
    echo "Args:"
    echo "<architecture_name> in [ FFNetwork_shrink,FFNetwork_shrink8,  FFNetwork_small,  FFNetwork_shrink256,  FFNetwork_shrink128]"
    echo -e "\t\t\t [ FFNetwork_decoder_shrink, FFNetwork_decoder_shrink8,  FFNetwork_decoder_small, FFNetwork_decoder_shrink256, FFNetwork_decoder_shrink128]"
    echo "--decoder: set it if the layer you want to train is part of the decoder"
    exit
fi

if [ $# == 2 ]; then
    if [ "$2" -ne "--decoder" ];then
        echo "Invalid argument"
        echo "submission_scripts/full_sentence/mha_only_FF_submit_all.sh <architecture_name> [--decoder]?"
        echo "<architecture_name> in [ FFNetwork_shrink,FFNetwork_shrink8,  FFNetwork_small,  FFNetwork_shrink256,  FFNetwork_shrink128]"
        echo "--decoder: set it if the layer you want to train is part of the decoder"
    fi
fi

./submission_scripts/utils/create_folder_outputs.sh mha_only $1
for i in {0..5};do
    echo sbatch --output=../sbatch_log/mha_only/$1/training_outputs/ submission_scripts/full_sentence/training_mha_only_FF.sh --num_of_curr_trained_layer $i --substitute_class $1 $2
    echo ""
    echo sbatch --output=../sbatch_log/mha_only/$1/training_outputs/ submission_scripts/full_sentence/training_mha_only_FF.sh --num_of_curr_trained_layer $i --substitute_class $1 $2
done
