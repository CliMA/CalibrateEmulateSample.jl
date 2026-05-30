#!/bin/bash
# Single SLURM job that instantiates and precompiles the project.
# The submit_*.sh scripts depend on this job before launching array jobs,
# so that 60 array tasks never race to precompile simultaneously.

#SBATCH --job-name=precompile
#SBATCH --output=output/slurm/precompile_%j.out
#SBATCH --error=output/slurm/precompile_%j.err
#SBATCH --time=00:30:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1

set -euo pipefail

module load julia/1.12.2   # keep in sync with calibrate_array.sbatch

cd "${SLURM_SUBMIT_DIR}"

julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'
