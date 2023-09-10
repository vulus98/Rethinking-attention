#! /bin/bash
if [ $# == 0 ]; then
    echo ""
    echo "submission_scripts/full_sentence/validation_script_ALSR_submit_all.sh <architecture_name>?"
    echo
    echo "Args:"
    echo "<architecture_name> in [ FFNetwork_L,FFNetwork_M,  FFNetwork_XL,FF Network_XS, FFNetwork_S]"
    echo "modify the parameter epoch as you need in the script"
    exit
fi
epoch=41
./submission_scripts/utils/create_folder_outputs.sh ALSR $1

# for i in {0..5}; do
#     echo "Substiting layer $i..."
#     sbatch  --output=../sbatch_log/ALSR/$1/evaluation_outputs/%j.out  submission_scripts/full_sentence/validation_script.sh --substitute_model_path$suffix $SCRATCH/models/checkpoints/ALSR/$1/  --epoch$suffix $epoch --substitute_type$suffix ALSR --substitute_class$suffix $1 --layers$suffix $i 
# done

echo "Substituting all layers"
sbatch  --output=./sbatch_log/ALSR/$1/evaluation_outputs/%j.out submission_scripts/full_sentence/validation_script.sh --substitute_model_path $SCRATCH/pytorch-original-transformer/models/checkpoints/ALSR/$1/  --epoch $epoch --substitute_type ALSR --substitute_class $1 
