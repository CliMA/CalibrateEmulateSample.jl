#!/bin/bash
# Single job that loads calibration output and produces diagnostic plots:
# data, solution (mean ensemble), and full-ensemble spread figures.
# Runs all (N_ens, rng_idx) cases serially in one job (no array needed).
#
# Invoked automatically by submit_l96_*.sh after calibrate completes.
# Manual use:
#   sbatch --dependency=afterok:<calib_jobid> \
#          --export=ALL,EXPERIMENT=l96_vec calibration_diagnostic_plots_l96.sbatch

#SBATCH --job-name=calib_diag
#SBATCH --output=output/slurm/calib_diag_%j.out
#SBATCH --error=output/slurm/calib_diag_%j.err
#SBATCH --time=00:30:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --constraint=cascadelake

set -euo pipefail

module load julia/1.12.2

export JULIA_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}

cd "${SLURM_SUBMIT_DIR}"

export JULIA_PKG_PRECOMPILE_AUTO=0

julia --project=. calibration_diagnostic_plots_l96.jl
