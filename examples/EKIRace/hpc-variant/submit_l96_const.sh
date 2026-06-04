#!/bin/bash
# Submit calibrate + emulate_sample + diagnostic plots for the L96 const-force case.
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

echo "NOTE: This script does not precompile. Run bash submit_precompile.sh first"
echo "      if you haven't done so recently (e.g. after a fresh checkout or package update)."

echo "=== Submitting calibrate (L96 const-force) ==="
CALIB_JID=$(sbatch --parsable \
		   -A esm \
		   --job-name="calib_${LABEL}" \
		   --export=ALL,SCRIPT=calibrate_l96.jl,EXPERIMENT=l96_const \
		   calibrate_array.sbatch)
echo "  calibrate job ID: ${CALIB_JID}"

echo "=== Submitting calibration_diagnostic_plots (L96 const-force, after ${CALIB_JID}) ==="
CALIB_DIAG_JID=$(sbatch --parsable \
		 -A esm \
		 --job-name="calib_diag_${LABEL}" \
		 --dependency=afterok:${CALIB_JID} \
		 --kill-on-invalid-dep=yes \
		 --export=ALL,EXPERIMENT=l96_const \
		 calibration_diagnostic_plots_l96.sbatch)
echo "  calibration_diagnostic_plots job ID: ${CALIB_DIAG_JID}"

echo "=== Submitting emulate_sample (L96 const-force, after ${CALIB_JID}) ==="
EMU_JID=$(sbatch --parsable \
		 -A esm \
		 --job-name="emu_${LABEL}" \
		 --dependency=afterok:${CALIB_JID} \
		 --kill-on-invalid-dep=yes \
		 --export=ALL,SCRIPT=emulate_sample_l96.jl,EXPERIMENT=l96_const \
		 emulate_sample_array.sbatch)
echo "  emulate_sample job ID: ${EMU_JID}"

echo "=== Submitting posterior_diagnostic_plots (L96 const-force, after ${EMU_JID}) ==="
POST_DIAG_JID=$(sbatch --parsable \
		 -A esm \
		 --job-name="post_diag_${LABEL}" \
		 --dependency=afterany:${EMU_JID} \
		 --export=ALL,EXPERIMENT=l96_const \
		 posterior_diagnostic_plots_l96.sbatch)
echo "  posterior_diagnostic_plots job ID: ${POST_DIAG_JID}"

echo "=== Submitting exp_to_leaderboard (L96 const-force, after ${EMU_JID}) ==="
LB_JID=$(sbatch --parsable \
		 -A esm \
		 --job-name="leaderboard_${LABEL}" \
		 --dependency=afterany:${EMU_JID} \
		 --export=ALL,EXPERIMENT=l96_const \
		 exp_to_leaderboard.sbatch)
echo "  exp_to_leaderboard job ID: ${LB_JID}"

echo "=== Done. Monitor with: squeue -u \$USER ==="

# sbatch -A esm --job-name="post_diag_l96_const" --export=ALL,EXPERIMENT=l96_const posterior_diagnostic_plots_l96.sbatch

# sbatch -A esm --job-name="leaderboard_l96_const" --export=ALL,EXPERIMENT=l96_const exp_to_leaderboard.sbatch
