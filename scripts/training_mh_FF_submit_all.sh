#! /bin/bash
for i in {0..5};do
    echo sbatch $HOME/pytorch-original-transformer/scripts/training_mh_FF.sh --checkpoints_folder large/l$i --num_of_curr_trained_layer $i
    
    sbatch $HOME/pytorch-original-transformer/scripts/training_mh_FF.sh --checkpoints_folder large/l$i --num_of_curr_trained_layer $i
    
done