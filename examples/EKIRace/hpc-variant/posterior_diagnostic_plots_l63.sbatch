#!/bin/bash
# Array job: one task per (N_ens, rng_idx) pair (180 tasks for the L63 case).
# Each task generates pushforward + posterior-ribbon diagnostic figures for L63.
#
# Invoked automatically by submit_l63.sh after pushforward_from_posterior completes.
# Manual use:
#   sbatch --dependency=afterok:<pushfwd_jobid> posterior_diagnostic_plots_l63.sbatch

#SBATCH --job-name=post_diag_l63
#SBATCH --output=output/slurm/post_diag_l63_%A_%a.out
#SBATCH --error=output/slurm/post_diag_l63_%A_%a.err
#SBATCH --array=1-180%100
#SBATCH --time=00:15:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=1
#SBATCH --constraint=cascadelake

set -euo pipefail

module load julia/1.12.2

export JULIA_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}

cd "${SLURM_SUBMIT_DIR}"

export JULIA_PKG_PRECOMPILE_AUTO=0

julia --project=. posterior_diagnostic_plots_l63.jl "${SLURM_ARRAY_TASK_ID}"
