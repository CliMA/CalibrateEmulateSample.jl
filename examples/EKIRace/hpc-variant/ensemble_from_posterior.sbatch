#!/bin/bash
# Single job that pushes all posteriors forward through the Lorenz96 forward map
# and saves pushforward plots. Runs all (N_ens, rng_idx) cases serially in one job
# (no array needed — the script loops internally).
#
# L96:
#   sbatch --dependency=afterok:<emu_jobid> \
#          --export=ALL,EXPERIMENT=l96_vec ensemble_from_posterior.sbatch
#   sbatch --dependency=afterok:<emu_jobid> \
#          --export=ALL,EXPERIMENT=l96_flux ensemble_from_posterior.sbatch

#SBATCH --job-name=postfwd
#SBATCH --output=output/slurm/postfwd_%j.out
#SBATCH --error=output/slurm/postfwd_%j.err
#SBATCH --time=08:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=1

set -euo pipefail

module load julia/1.12.2

export JULIA_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}

cd "${SLURM_SUBMIT_DIR}"

export JULIA_PKG_PRECOMPILE_AUTO=0

julia --project=. ensemble_from_posterior.jl
