#!/usr/bin/env bash
sbatch submission_scripts/averaging/sim_all_pretrain.sh --ELR
sbatch submission_scripts/averaging/sim_all_pretrain.sh --ALR
sbatch submission_scripts/averaging/sim_all_pretrain.sh --ALRR
