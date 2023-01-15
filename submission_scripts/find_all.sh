#!/usr/bin/env bash
sbatch find_single_layer_arch.sh --whole
sbatch find_single_layer_arch.sh --just_attention
sbatch find_single_layer_arch.sh --with_residual
