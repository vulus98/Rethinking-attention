#!/bin/bash
DIR="$HOME/sbatch_log/$1"
if [ ! -d $DIR ];then
    mkdir $DIR 
fi
model_name=$DIR/$2
mkdir $model_name
mkdir $model_name/evaluation_outputs
mkdir $model_name/training_outputs
