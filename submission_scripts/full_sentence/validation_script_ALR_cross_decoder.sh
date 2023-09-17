#! /bin/bash
if [ $# == 0 ]; then
    echo ""
    echo "submission_scripts/full_sentence/validation_script_ALR_submit_all.sh <architecture_name>?"
    echo
    echo "Args:"
    echo "<architecture_name> in [ FFNetwork_cross_decoder_L,FFNetwork_cross_decoder_M,  FFNetwork_cross_decoder_XL,FF Network_cross_decoder_XS, FFNetwork_cross_decoder_S]"
    exit
fi
epoch=21
./submission_scripts/utils/create_folder_outputs.sh ALR $1

# for i in {0..5}; do
#     echo "Substiting layer $i..."
#     sbatch  --output=../sbatch_log/ALR/$1/evaluation_outputs/%j.out  submission_scripts/full_sentence/validation_script.sh --substitute_model_path$suffix $SCRATCH/models/checkpoints/ALR/$1/  --epoch$suffix $epoch --substitute_type$suffix ALR --substitute_class$suffix $1 --layers$suffix $i 
# done

echo "Substituting all layers"
sbatch  --output=./sbatch_log/ALR/$1/evaluation_outputs/%j.out submission_scripts/full_sentence/validation_script.sh --substitute_model_path_d_ca $SCRATCH/pytorch-original-transformer/models/checkpoints/ALR/$1/  --epoch_d_ca $epoch --substitute_type_d_ca ALR --substitute_class_d_ca $1 
