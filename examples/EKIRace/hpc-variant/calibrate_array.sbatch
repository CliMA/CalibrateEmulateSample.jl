#!/bin/bash
# SLURM array job for the calibrate stage.
# array=1-60%20 means run array 1-60, with max of 20 at a time.
#
# L63 (60 tasks = 3 N_ens × 20 repeats):
#   sbatch --export=ALL,SCRIPT=calibrate_l63.jl calibrate_array.sbatch
#
# L96 (same count; adjust SCRIPT and set EXPERIMENT):
#   sbatch --export=ALL,SCRIPT=calibrate_l96.jl,EXPERIMENT=l96_const calibrate_array.sbatch
#   sbatch --export=ALL,SCRIPT=calibrate_l96.jl,EXPERIMENT=l96_vec   calibrate_array.sbatch
#   sbatch --export=ALL,SCRIPT=calibrate_l96.jl,EXPERIMENT=l96_flux  calibrate_array.sbatch
#
# If N_ens_sizes or n_repeats change, update --array upper bound to match
# length(N_ens_sizes) * n_repeats.
#
# One-time precompile before submitting (avoids 60 simultaneous precompiles):
#   mkdir -p output/slurm
#   julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'

#SBATCH --job-name=calib
#SBATCH --output=output/slurm/calib_%A_%a.out
#SBATCH --error=output/slurm/calib_%A_%a.err
#SBATCH --array=1-60%20
#SBATCH --time=04:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=1

set -euo pipefail

module load julia/1.12.2   # pin version as needed, e.g.: module load julia/1.10.4

export JULIA_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}

# Default to L63 if SCRIPT not set via --export.
SCRIPT=${SCRIPT:-calibrate_l63.jl}

cd "${SLURM_SUBMIT_DIR}"

export JULIA_PKG_PRECOMPILE_AUTO=0 # already precompiled so should not get all jobs to re-precompile

julia --project=. "${SCRIPT}" "${SLURM_ARRAY_TASK_ID}"
