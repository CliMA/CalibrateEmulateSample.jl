#!/bin/bash
# Submit calibrate + emulate_sample for the L96 const-force case.
#
# Usage: bash submit_l96_const.sh [EXP_ID]
#   EXP_ID (optional): label appended to SLURM job names so the queue stays
#                      readable when all four cases run simultaneously,
#                      e.g. "run2" → jobs appear as "calib_l96c_run2".

set -euo pipefail

EXP_ID=${1:-}
LABEL="l96c${EXP_ID:+_${EXP_ID}}"
DIR="$(cd "$(dirname "$0")" && pwd)"

cd "$DIR"
mkdir -p output/slurm

echo "=== Precompiling ==="
julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'

echo "=== Submitting calibrate (L96 const-force) ==="
CALIB_JID=$(sbatch --parsable \
    --job-name="calib_${LABEL}" \
    --export=ALL,SCRIPT=calibrate_l96.jl,EXPERIMENT=l96_const \
    calibrate_array.sbatch)
echo "  calibrate job ID: ${CALIB_JID}"

echo "=== Submitting emulate_sample (L96 const-force, after ${CALIB_JID}) ==="
EMU_JID=$(sbatch --parsable \
    --job-name="emu_${LABEL}" \
    --dependency=afterok:${CALIB_JID} \
    --export=ALL,SCRIPT=emulate_sample_l96.jl,EXPERIMENT=l96_const \
    emulate_sample_array.sbatch)
echo "  emulate_sample job ID: ${EMU_JID}"

echo "=== Done. Monitor with: squeue -u \$USER ==="
