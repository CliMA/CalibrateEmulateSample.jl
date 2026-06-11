#!/bin/bash
# Array job: one task per (N_ens, rng_idx) pair (60 tasks for each L96 case).
# Each task pushes its posterior forward through the Lorenz96 forward map and
# saves pushforward + posterior-ribbon diagnostic figures.
#
# Invoked automatically by submit_l96_*.sh after emulate_sample completes.
# Manual use:
#   sbatch --dependency=afterok:<emu_jobid> \
#          --export=ALL,EXPERIMENT=l96_vec posterior_diagnostic_plots_l96.sbatch

#SBATCH --job-name=post_diag
#SBATCH --output=output/slurm/post_diag_%A_%a.out
#SBATCH --error=output/slurm/post_diag_%A_%a.err
#SBATCH --array=1-180%100
#SBATCH --time=00:30:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=1

set -euo pipefail

module load julia/1.12.2

export JULIA_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}

cd "${SLURM_SUBMIT_DIR}"

export JULIA_PKG_PRECOMPILE_AUTO=0

julia --project=. posterior_diagnostic_plots_l96.jl "${SLURM_ARRAY_TASK_ID}"
