#!/usr/bin/env bash
sbatch submission_scripts/averaging/find_single_layer_arch.sh --ELR
sbatch submission_scripts/averaging/find_single_layer_arch.sh --ALR
sbatch submission_scripts/averaging/find_single_layer_arch.sh --ALRR
