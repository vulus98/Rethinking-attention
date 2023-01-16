#! /bin/bash
if [ $# == 0 ]; then
    echo ""
    echo "submission_scripts/full_sentence/validation_script_ALR_submit_all.sh <architecture_name> [--decoder]?"
    echo
    echo "Args:"
    echo "<architecture_name> in [ FFNetwork_L,FFNetwork_M,  FFNetwork_XL,FF Network_XS, FFNetwork_S]"
    echo -e "\t\t\t [ FFNetwork_decoder_L, FFNetwork_decoder_M,  FFNetwork_decoder_XL, FFNetwork_decoder_XS, FFNetwork_decoder_S]"
    echo "--decoder: set it if the layer you want to substitute decoder layers"
    echo "modify the parameter epoch as you need in the script"
    exit
fi
epoch=41
./submission_scripts/utils/create_folder_outputs.sh ALR $1
if [ $2 == "--decoder" ]; then
    suffix=_d
fi
for i in {0..5}; do
    echo "Substiting layer $i..."
    sbatch  --output=../sbatch_log/ALR/$1/evaluation_outputs/%j.out  submission_scripts/validation.sh --substitute_model_path$suffix $SCRATCH/models/checkpoints/ALR/$1/  --epoch$suffix $epoch --substitute_type$suffix ALR --substitute_class$suffix $1 --layers$suffix $i 
done

echo "Substituting all layers"
sbatch  --output=../sbatch_log/ALR/$1/evaluation_outputs/%j.out submission_scripts/validation.sh --substitute_model_path$suffix $SCRATCH/models/checkpoints/ALR/$1/  --epoch$suffix $epoch --substitute_type$suffix ALR --substitute_class$suffix $1 

echo "Substituting all layers with untrained FF..."
sbatch  --output=../sbatch_log/ALR/$1/evaluation_outputs/%j.out  submission_scripts/validation.sh --substitute_model_path$suffix $SCRATCH/models/checkpoints/ALR/$1/  --epoch$suffix $epoch --substitute_type$suffix ALR --substitute_class$suffix $1 --untrained$suffix
