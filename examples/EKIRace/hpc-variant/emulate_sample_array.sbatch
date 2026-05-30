#!/bin/bash
# SLURM array job for the emulate-sample stage.
# array=1-60%20 means run array 1-60, with max of 20 at a time.
#
# L63:
#   sbatch --dependency=afterok:<calib_jobid> \
#          --export=ALL,SCRIPT=emulate_sample_l63.jl emulate_sample_array.sbatch
#
# L96:
#   sbatch --dependency=afterok:<calib_jobid> \
#          --export=ALL,SCRIPT=emulate_sample_l96.jl,EXPERIMENT=l96_const emulate_sample_array.sbatch
#
# Full chained submission example for L63:
#   cid=$(sbatch --parsable --export=ALL,SCRIPT=calibrate_l63.jl calibrate_array.sbatch)
#   sbatch --dependency=afterok:$cid \
#          --export=ALL,SCRIPT=emulate_sample_l63.jl emulate_sample_array.sbatch

#SBATCH --job-name=emusample
#SBATCH --output=output/slurm/emusample_%A_%a.out
#SBATCH --error=output/slurm/emusample_%A_%a.err
#SBATCH --array=1-60%100
#SBATCH --time=12:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=1

set -euo pipefail

module load julia/1.12.2   # pin version as needed, e.g.: module load julia/1.10.4

export JULIA_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}

SCRIPT=${SCRIPT:-emulate_sample_l63.jl}

cd "${SLURM_SUBMIT_DIR}"

export JULIA_PKG_PRECOMPILE_AUTO=0 # already precompiled so should not get all jobs to re-precompile

julia --project=. "${SCRIPT}" "${SLURM_ARRAY_TASK_ID}"
