#!/bin/bash
# Single job: converts emulate_sample posterior files to a leaderboard NetCDF.
# Runs all (N_ens, rng_idx) cases serially in one job (no array needed).
#
# Invoked automatically by submit_l*.sh after emulate_sample completes.
# Manual use:
#   L96: sbatch --dependency=afterok:<emu_jobid> \
#               --export=ALL,EXPERIMENT=l96_flux exp_to_leaderboard.sbatch
#   L63: sbatch --dependency=afterok:<emu_jobid> exp_to_leaderboard.sbatch

#SBATCH --job-name=leaderboard
#SBATCH --output=output/slurm/leaderboard_%j.out
#SBATCH --error=output/slurm/leaderboard_%j.err
#SBATCH --time=1:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=1

set -euo pipefail

module load julia/1.12.2

export JULIA_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}

cd "${SLURM_SUBMIT_DIR}"

export JULIA_PKG_PRECOMPILE_AUTO=0

if [[ -z "${EXPERIMENT:-}" || "${EXPERIMENT}" == "l63" ]]; then
    julia --project=. l63_exp_to_leaderboard_utilities.jl
else
    julia --project=. l96_exp_to_leaderboard_utilities.jl
fi
