#! /bin/bash
if [ $# == 0 ]; then
    echo ""
    echo "submission_scripts/full_sentence/validation_script_ALR_submit_all.sh <architecture_name> [--encoder] <architecture_name> [--decoder] <architecture_name> [--decoder_ca]?"
    echo
    echo "Args:"
    echo "<architecture_name> in [ FFNetwork_L,FFNetwork_M,  FFNetwork_XL,FF Network_XS, FFNetwork_S]"
    echo "--encoder: set it if the layer you want to substitute encoder layers"
    echo -e "\t\t\t [ FFNetwork_decoder_L, FFNetwork_decoder_M,  FFNetwork_decoder_XL, FFNetwork_decoder_XS, FFNetwork_decoder_S]"
    echo "--decoder: set it if the layer you want to substitute decoder layers"
    echo "modify the parameter epoch as you need in the script"
    exit
fi
epoch=21
./submission_scripts/utils/create_folder_outputs.sh ALR $1

suffix_1=_d
# for i in {0..5}; do
#     echo "Substiting layer $i..."
#     sbatch  --output=../sbatch_log/ALR/$1/evaluation_outputs/%j.out  submission_scripts/full_sentence/validation_script.sh --substitute_model_path$suffix $SCRATCH/models/checkpoints/ALR/$1/  --epoch$suffix $epoch --substitute_type$suffix ALR --substitute_class$suffix $1 --layers$suffix $i 
# done

echo "Substituting all layers"
sbatch  --output=./sbatch_log/ALR/$1/evaluation_outputs/%j.out submission_scripts/full_sentence/validation_script.sh --substitute_model_path $SCRATCH/pytorch-original-transformer/models/checkpoints/ALR/$1/  --epoch $epoch --substitute_type ALR --substitute_class $1 --substitute_model_path$suffix_1 $SCRATCH/pytorch-original-transformer/models/checkpoints/ALR/$3/  --epoch$suffix_1 $epoch --substitute_type$suffix_1 ALR --substitute_class$suffix_1 $3 
