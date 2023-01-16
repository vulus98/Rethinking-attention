#!/usr/bin/env bash
sbatch evaluate.sh --vanilla
sbatch evaluate.sh --single_sim
sbatch evaluate.sh --ELR
sbatch evaluate.sh --ALR
sbatch evaluate.sh --ALRR
