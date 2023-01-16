#! /bin/bash
if [ $# == 0 ]; then
    echo ""
    echo "submission_scripts/full_sentence/validation_script_mha_only_submit_all.sh <architecture_name> [--decoder]?"
    echo
    echo "Args:"
    echo "<architecture_name> in [ FFNetwork_shrink,FFNetwork_shrink8,  FFNetwork_small,FF Network_shrink256, FFNetwork_shrink128]"
    echo -e "\t\t\t [ FFNetwork_decoder_shrink, FFNetwork_decoder_shrink8,  FFNetwork_decoder_small, FFNetwork_decoder_shrink256, FFNetwork_decoder_shrink128]"
    echo "--decoder: set it if the layer you want to substitute decoder layers"
    echo "modify the parameter epoch as you need in the script"
    exit
fi
epoch=41
./submission_scripts/utils/create_folder_outputs.sh mha_only $1
if [ $2 == "--decoder" ]; then
    suffix=_d
fi
for i in {0..5}; do
    echo "Substiting layer $i..."
    echo sbatch  --output=../sbatch_log/mha_only/$1/evaluation_outputs/%j.out  submission_scripts/validation.sh --substitute_model_path$suffix $SCRATCH/models/checkpoints/mha/$1/  --epoch$suffix $epoch --substitute_type$suffix mha_only --substitute_class$suffix $1 --layers$suffix $i 
done

echo "Substituting all layers"
echo sbatch  --output=../sbatch_log/mha_only/$1/evaluation_outputs/%j.out submission_scripts/validation.sh --substitute_model_path$suffix $SCRATCH/models/checkpoints/mha/$1/  --epoch$suffix $epoch --substitute_type$suffix mha_only --substitute_class$suffix $1 

echo "Substituting all layers with untrained FF..."
echo sbatch  --output=../sbatch_log/mha_only/$1/evaluation_outputs/%j.out  submission_scripts/validation.sh --substitute_model_path$suffix $SCRATCH/models/checkpoints/mha/$1/  --epoch$suffix $epoch --substitute_type$suffix mha_only --substitute_class$suffix $1 --untrained$suffix
