#!/usr/bin/env bash
sbatch submission_scripts/averaging/evaluate.sh --vanilla
sbatch submission_scripts/averaging/evaluate.sh --single_sim
sbatch submission_scripts/averaging/evaluate.sh --ELR
sbatch submission_scripts/averaging/evaluate.sh --ALR
sbatch submission_scripts/averaging/evaluate.sh --ALRR
