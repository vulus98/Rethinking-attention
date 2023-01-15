#!/usr/bin/env bash
sbatch evaluate.sh --vanilla
sbatch evaluate.sh --single_sim
sbatch evaluate.sh --multi
sbatch evaluate.sh --just_attention
sbatch evaluate.sh --with_residual
