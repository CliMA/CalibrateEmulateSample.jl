#!/bin/bash
# Submit calibrate + emulate_sample for the L96 vec-force case.
#
# Usage: bash submit_l96_vec.sh [EXP_ID]
#   EXP_ID (optional): label appended to SLURM job names so the queue stays
#                      readable when all four cases run simultaneously,
#                      e.g. "run2" → jobs appear as "calib_l96v_run2".

set -euo pipefail

EXP_ID=${1:-}
LABEL="l96v${EXP_ID:+_${EXP_ID}}"
DIR="$(cd "$(dirname "$0")" && pwd)"

cd "$DIR"
mkdir -p output/slurm

echo "NOTE: This script does not precompile. Run bash submit_precompile.sh first"
echo "      if you haven't done so recently (e.g. after a fresh checkout or package update)."

echo "=== Submitting calibrate (L96 vec-force) ==="
CALIB_JID=$(sbatch --parsable \
		   -A esm \
		   --job-name="calib_${LABEL}" \
		   --export=ALL,SCRIPT=calibrate_l96.jl,EXPERIMENT=l96_vec \
		   calibrate_array.sbatch)
echo "  calibrate job ID: ${CALIB_JID}"

echo "=== Submitting emulate_sample (L96 vec-force, after ${CALIB_JID}) ==="
EMU_JID=$(sbatch --parsable \
		 -A esm \
		 --job-name="emu_${LABEL}" \
		 --dependency=afterok:${CALIB_JID} \
		 --kill-on-invalid-dep=yes \
		 --export=ALL,SCRIPT=emulate_sample_l96.jl,EXPERIMENT=l96_vec \
		 emulate_sample_array.sbatch)
echo "  emulate_sample job ID: ${EMU_JID}"

echo "=== Submitting ensemble_from_posterior (L96 vec-force, after ${EMU_JID}) ==="
POST_JID=$(sbatch --parsable \
		 -A esm \
		 --job-name="post_${LABEL}" \
		 --dependency=afterok:${EMU_JID} \
		 --kill-on-invalid-dep=yes \
		 --export=ALL,EXPERIMENT=l96_vec \
		 ensemble_from_posterior.sbatch)
echo "  ensemble_from_posterior job ID: ${POST_JID}"

echo "=== Done. Monitor with: squeue -u \$USER ==="
