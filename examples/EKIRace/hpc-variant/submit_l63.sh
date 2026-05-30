#!/bin/bash
# Submit calibrate + emulate_sample for the L63 case.
#
# Usage: bash submit_l63.sh [EXP_ID]
#   EXP_ID (optional): label appended to SLURM job names so the queue stays
#                      readable when all four cases run simultaneously,
#                      e.g. "run2" → jobs appear as "calib_l63_run2".

set -euo pipefail

EXP_ID=${1:-}
LABEL="l63${EXP_ID:+_${EXP_ID}}"
DIR="$(cd "$(dirname "$0")" && pwd)"

cd "$DIR"
mkdir -p output/slurm

echo "=== Submitting precompile ==="
PRECOMPILE_JID=$(sbatch --parsable \
			-A esm \
			--job-name="precompile_${LABEL}" \
			precompile.sbatch)
echo "  precompile job ID: ${PRECOMPILE_JID}"

echo "=== Submitting calibrate (L63, after ${PRECOMPILE_JID}) ==="
CALIB_JID=$(sbatch --parsable \
		   -A esm \
		   --job-name="calib_${LABEL}" \
		   --dependency=afterok:${PRECOMPILE_JID} \
		   --kill-on-invalid-dep=yes \
		   --export=ALL,SCRIPT=calibrate_l63.jl \
		   calibrate_array.sbatch)
echo "  calibrate job ID: ${CALIB_JID}"

echo "=== Submitting emulate_sample (L63, after ${CALIB_JID}) ==="
EMU_JID=$(sbatch --parsable \
		 -A esm \
		 --job-name="emu_${LABEL}" \
		 --dependency=afterok:${CALIB_JID} \
		 --kill-on-invalid-dep=yes \
		 --export=ALL,SCRIPT=emulate_sample_l63.jl \
		 emulate_sample_array.sbatch)
echo "  emulate_sample job ID: ${EMU_JID}"

echo "=== Done. Monitor with: squeue -u \$USER ==="
