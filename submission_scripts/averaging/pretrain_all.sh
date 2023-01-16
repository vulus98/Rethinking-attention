#!/usr/bin/env bash
sbatch sim_all_pretrain.sh --ELR
sbatch sim_all_pretrain.sh --ALR
sbatch sim_all_pretrain.sh --ALRR
