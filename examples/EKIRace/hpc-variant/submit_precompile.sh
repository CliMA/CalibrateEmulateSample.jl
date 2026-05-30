#!/bin/bash
# Submit the one-off precompile job.
# Run this once before (or instead of re-running) the submit_l*.sh scripts
# whenever the environment changes (new package versions, fresh checkout, etc.).
#
# Usage: bash submit_precompile.sh [EXP_ID]
#   EXP_ID (optional): label appended to the SLURM job name.

set -euo pipefail

EXP_ID=${1:-}
LABEL="precompile${EXP_ID:+_${EXP_ID}}"
DIR="$(cd "$(dirname "$0")" && pwd)"

cd "$DIR"
mkdir -p output/slurm

echo "=== Submitting precompile ==="
PRECOMPILE_JID=$(sbatch --parsable \
			-A esm \
			--job-name="${LABEL}" \
			precompile.sbatch)
echo "  precompile job ID: ${PRECOMPILE_JID}"
echo "=== Done. Monitor with: squeue -u \$USER ==="
