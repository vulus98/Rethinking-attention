#! /bin/bash
for i in {0..5};do
    echo sbatch $HOME/pytorch-original-transformer/scripts/training_mh_FF.sh --checkpoints_folder small/l$i --num_of_curr_trained_layer $i --substitute_class FFNetwork_small
    
    sbatch $HOME/pytorch-original-transformer/scripts/training_mh_FF.sh --checkpoints_folder small/l$i --num_of_curr_trained_layer $i --substitute_class FFNetwork_small
    
done