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

echo "NOTE: This script does not precompile. Run bash submit_precompile.sh first"
echo "      if you haven't done so recently (e.g. after a fresh checkout or package update)."

echo "=== Submitting calibrate (L63) ==="
CALIB_JID=$(sbatch --parsable \
		   -A esm \
		   --job-name="calib_${LABEL}" \
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

echo "=== Submitting pushforward_from_posterior (L63, after ${EMU_JID}) ==="
PUSHFWD_JID=$(sbatch --parsable \
		 -A esm \
		 --job-name="pushfwd_${LABEL}" \
		 --dependency=afterany:${EMU_JID} \
		 pushforward_from_posterior.sbatch)
echo "  pushforward_from_posterior job ID: ${PUSHFWD_JID}"

echo "=== Submitting posterior_diagnostic_plots (L63, after ${PUSHFWD_JID}) ==="
POST_DIAG_JID=$(sbatch --parsable \
		 -A esm \
		 --job-name="post_diag_${LABEL}" \
		 --dependency=afterany:${PUSHFWD_JID} \
		 posterior_diagnostic_plots_l63.sbatch)
echo "  posterior_diagnostic_plots job ID: ${POST_DIAG_JID}"

echo "=== Submitting exp_to_leaderboard (L63, after ${PUSHFWD_JID}) ==="
LB_JID=$(sbatch --parsable \
		 -A esm \
		 --job-name="leaderboard_${LABEL}" \
		 --dependency=afterany:${PUSHFWD_JID} \
		 --export=ALL,EXPERIMENT=l63 \
		 exp_to_leaderboard.sbatch)
echo "  exp_to_leaderboard job ID: ${LB_JID}"

echo "=== Done. Monitor with: squeue -u \$USER ==="

# sbatch --parsable -A esm --job-name="pushfwd_l63" --export=ALL pushforward_from_posterior.sbatch

# sbatch -A esm --job-name="post_diag_l63" posterior_diagnostic_plots_l63.sbatch

# sbatch -A esm --job-name="leaderboard_l63" --export=ALL,EXPERIMENT=l63 exp_to_leaderboard.sbatch
