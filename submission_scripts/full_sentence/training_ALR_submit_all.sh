#! /bin/bash
if [ $# == 0 ]; then
    echo ""
    echo "submission_scripts/full_sentence/training_ALR_submit_all.sh <architecture_name> [--decoder]?"
    echo
    echo "Args:"
    echo "<architecture_name> in [ FFNetwork_L,FFNetwork_M,  FFNetwork_XL,  FFNetwork_XS,  FFNetwork_S]"
    echo -e "\t\t\t [ FFNetwork_decoder_L, FFNetwork_decoder_M,  FFNetwork_decoder_XL, FFNetwork_decoder_XS, FFNetwork_decoder_S]"
    echo "--decoder: set it if the layer you want to train is part of the decoder"
    exit
fi

if [ $# == 2 ]; then
    if [ "$2" -ne "--decoder" ];then
        echo "Invalid argument"
        echo "submission_scripts/full_sentence/training_ALR_submit_all.sh <architecture_name> [--decoder]?"
        echo "<architecture_name> in [ FFNetwork_L,FFNetwork_M,  FFNetwork_XL,  FFNetwork_XS,  FFNetwork_S]"
        echo "--decoder: set it if the layer you want to train is part of the decoder"
    fi
fi

./submission_scripts/utils/create_folder_outputs.sh ALR $1
for i in {0..5};do
    echo sbatch --output=../sbatch_log/ALR/$1/training_outputs/ submission_scripts/full_sentence/training_ALR.sh --num_of_curr_trained_layer $i --substitute_class $1 $2
    echo ""
    echo sbatch --output=../sbatch_log/ALR/$1/training_outputs/ submission_scripts/full_sentence/training_ALR.sh --num_of_curr_trained_layer $i --substitute_class $1 $2
done