#!/bin/bash
# Array job: one task per (N_ens, rng_idx) pair.
# Each task loads the emulate_sample posterior, pushes n_pushforward_samples through
# the forward map for every k-iteration, and saves forcing + output samples back into
# the posterior JLD2 file.  Downstream consumers (leaderboard utilities, diagnostic
# plots) load these stored arrays instead of rerunning the expensive forward map.
#
# L63 (180 tasks for default config):
#   sbatch --dependency=afterok:<emu_jobid> push_forward_from_posterior.sbatch
#
# L96 (180 tasks per case):
#   sbatch --dependency=afterok:<emu_jobid> \
#          --export=ALL,EXPERIMENT=l96_flux push_forward_from_posterior.sbatch

#SBATCH --job-name=pushfwd
#SBATCH --output=output/slurm/pushfwd_%A_%a.out
#SBATCH --error=output/slurm/pushfwd_%A_%a.err
#SBATCH --array=1-180%100
#SBATCH --time=00:30:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=1
#SBATCH --constraint=cascadelake

set -euo pipefail

module load julia/1.12.2

export JULIA_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}

cd "${SLURM_SUBMIT_DIR}"

export JULIA_PKG_PRECOMPILE_AUTO=0

if [[ -z "${EXPERIMENT:-}" || "${EXPERIMENT}" == "l63" ]]; then
    julia --project=. pushforward_from_posterior_l63.jl "${SLURM_ARRAY_TASK_ID}"
else
    julia --project=. pushforward_from_posterior_l96.jl "${SLURM_ARRAY_TASK_ID}"
fi
