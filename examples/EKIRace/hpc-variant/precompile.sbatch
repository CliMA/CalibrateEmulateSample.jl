#!/bin/bash
# Single SLURM job that instantiates and precompiles the project.
# Run via submit_precompile.sh before launching array jobs so that the 60+
# array tasks start with a warm package cache and never race to precompile.

#SBATCH --job-name=precompile
#SBATCH --output=output/slurm/precompile_%j.out
#SBATCH --error=output/slurm/precompile_%j.err
#SBATCH --time=00:30:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=1
#SBATCH --constraint=cascadelake

set -euo pipefail

module load julia/1.12.2   # keep in sync with calibrate_array.sbatch

cd "${SLURM_SUBMIT_DIR}"

export JULIA_NUM_THREADS=$SLURM_CPUS_PER_TASK # parallelize

# To see what is available call: sinfo -o "%20N %10c %20f"
# We explicitly compile on one set of nodes, and then run on another
export JULIA_CPU_TARGET="cascadelake" # explicitly target an architecture
export JULIA_PKG_PRECOMPILE_AUTO=1

echo "JULIA_CPU_TARGET=$JULIA_CPU_TARGET"

julia -e '
println("CPU = ", Sys.CPU_NAME)
println("JULIA_CPU_TARGET = ", get(ENV,"JULIA_CPU_TARGET","<unset>"))
'
# these two should be the same

julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'

