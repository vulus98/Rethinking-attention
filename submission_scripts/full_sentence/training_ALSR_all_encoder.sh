#! /bin/bash
if [ $# == 0 ]; then
    echo ""
    echo "submission_scripts/full_sentence/training_ALSR_submit_all.sh <architecture_name>?"
    echo
    echo "Args:"
    echo "<architecture_name> in [ FFNetwork_L,FFNetwork_M,  FFNetwork_XL,  FFNetwork_XS,  FFNetwork_S]"
    exit
fi

./submission_scripts/utils/create_folder_outputs.sh ALSR $1
for i in {0..5};do
    echo sbatch --output=./sbatch_log/ALSR/$1/training_outputs/%j.out submission_scripts/full_sentence/training_ALSR.sh --num_of_curr_trained_layer $i --substitute_class $1
    echo ""
    sbatch --output=./sbatch_log/ALSR/$1/training_outputs/%j.out submission_scripts/full_sentence/training_ALSR.sh --num_of_curr_trained_layer $i --substitute_class $1 
done
