#!/usr/bin/env bash
sbatch sim_all_pretrain.sh --whole
sbatch sim_all_pretrain.sh --just_attention
sbatch sim_all_pretrain.sh --with_residual
