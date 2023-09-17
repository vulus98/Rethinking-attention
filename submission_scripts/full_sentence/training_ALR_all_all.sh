#! /bin/bash
if [ $# == 0 ]; then
    echo ""
    echo "submission_scripts/full_sentence/training_ALR_all_all.sh <architecture_name> <att_replacement>?"
    echo
    echo "Args:"
    echo "<architecture_name> in [ FFNetwork_L,FFNetwork_M,  FFNetwork_XL,  FFNetwork_XS,  FFNetwork_S]"
    echo "<att_replacement> in encoder,  set it if the layer you want to train is part of the encoder"
    echo -e "\t\t\t [ FFNetwork_decoder_L, FFNetwork_decoder_M,  FFNetwork_decoder_XL, FFNetwork_decoder_XS, FFNetwork_decoder_S]"
    echo "<att_replacement> in decoder,  set it if the layer you want to train is part of the self decoder"
    echo -e "\t\t\t [ FFNetwork_cross_decoder_L, FFNetwork_cross_decoder_M,  FFNetwork_cross_decoder_XL, FFNetwork_cross_decoder_XS, FFNetwork_cross_decoder_S]"
    echo "<att_replacement> in decoder_ca,  set it if the layer you want to train is part of the cross decoder"
    exit
fi


./submission_scripts/utils/create_folder_outputs.sh ALR $1
for i in {0..5};do
    echo sbatch --output=./sbatch_log/ALR/$1/training_outputs/%j.out submission_scripts/full_sentence/training_ALR.sh --num_of_curr_trained_layer $i --substitute_class $1 --att_replacement $2
    echo ""
    sbatch --output=./sbatch_log/ALR/$1/training_outputs/%j.out submission_scripts/full_sentence/training_ALR.sh --num_of_curr_trained_layer $i --substitute_class $1 --att_replacement $2
done
