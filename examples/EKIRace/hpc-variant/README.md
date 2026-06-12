# EKIRace HPC variant

Parallelised versions of the `calibrate` and `emulate_sample` scripts.
Each script can run serially (identical behaviour to the originals) or as a
SLURM job array where every `(N_ens, rng_idx)` cell is an independent task.

## One-time setup

In `experiment_config.jl`, pin the calibrate date before starting a run, e.g.

```julia
calibrate_date = Date("2026-06-04", "yyyy-mm-dd")
```

All subsequent stages read this value to locate the right output directory;
keeping it fixed avoids mismatches when jobs run past midnight or across days.

Precompilation is handled by a dedicated `submit_precompile.sh` script that
queues `precompile.sbatch` as a compute job. Run it once before your experiments
and again whenever the environment changes (fresh checkout, package updates).
The `submit_l*.sh` scripts do not precompile — they will remind you at
submission time.

## Pipeline

### L63

```
calibrate_array  ──afterok──►  emulate_sample_array  ──afterany──►  pushforward_from_posterior  ──afterany──►  exp_to_leaderboard
```

### L96 (const / vec / flux)

```
                         ┌──afterok──►  calibration_diagnostic_plots_l96
calibrate_array  ──afterok──►  emulate_sample_array  ──afterany──►  pushforward_from_posterior  ──afterany──►  posterior_diagnostic_plots_l96
                                                                                                 ──afterany──►  exp_to_leaderboard
```

`calibration_diagnostic_plots` and `emulate_sample` both start once calibrate
succeeds (they run in parallel).  `pushforward_from_posterior` starts once
`emulate_sample` finishes and runs the Lorenz forward map for every posterior
cell in parallel, saving forcing and output samples back into each posterior
JLD2 file.  `posterior_diagnostic_plots` and `exp_to_leaderboard` both start
once `pushforward_from_posterior` finishes — they simply load the precomputed
samples rather than re-running the forward map.

## Standalone (serial)

Run with no arguments to sweep all `(N_ens, rng_idx)` cells sequentially.

```bash
# L63
julia --project=. calibrate_l63.jl
julia --project=. emulate_sample_l63.jl
julia --project=. pushforward_from_posterior_l63.jl

# L96 — set EXPERIMENT env var or edit the toggle in experiment_config.jl
EXPERIMENT=l96_const julia --project=. calibrate_l96.jl
EXPERIMENT=l96_const julia --project=. calibration_diagnostic_plots_l96.jl
EXPERIMENT=l96_const julia --project=. emulate_sample_l96.jl
EXPERIMENT=l96_const julia --project=. pushforward_from_posterior_l96.jl
EXPERIMENT=l96_const julia --project=. posterior_diagnostic_plots_l96.jl
```

You can also run a single cell by passing its 1-based task index for the
array-capable scripts:

```bash
julia --project=. calibrate_l63.jl 1                    # first (N_ens, rng_idx) cell only
julia --project=. emulate_sample_l63.jl 5               # fifth cell only
julia --project=. pushforward_from_posterior_l63.jl 5   # fifth cell only
EXPERIMENT=l96_const julia --project=. pushforward_from_posterior_l96.jl 3
EXPERIMENT=l96_const julia --project=. posterior_diagnostic_plots_l96.jl 3
```

Leaderboard conversion runs all cells serially in a single call (requires
pushforward to have been run first):

```bash
julia --project=. l63_exp_to_leaderboard_utilities.jl
EXPERIMENT=l96_const julia --project=. l96_exp_to_leaderboard_utilities.jl
```

## HPC (Caltech Resnick cluster, SLURM)

### Submission scripts (recommended)

Precompile once (or whenever the environment changes), then submit the cases:

```bash
bash submit_precompile.sh [EXP_ID]

bash submit_l63.sh        [EXP_ID]
bash submit_l96_const.sh  [EXP_ID]
bash submit_l96_vec.sh    [EXP_ID]
bash submit_l96_flux.sh   [EXP_ID]
```

Each `submit_l*.sh` script chains the full pipeline for its case automatically.
All four cases can be launched simultaneously — output files are case-specific
so there are no write conflicts.  The optional `EXP_ID` argument suffixes SLURM
job names to keep the queue readable.

If you are launching all four cases together you only need one precompile run:

```bash
bash submit_precompile.sh run1
for s in submit_l63.sh submit_l96_const.sh submit_l96_vec.sh submit_l96_flux.sh; do
    bash "$s" run1 &
done
wait
```

### Manual submission

Precompile via `submit_precompile.sh` (or directly), then submit each stage:

```bash
# L63
CALIB_JID=$(sbatch --parsable -A esm \
            --export=ALL,SCRIPT=calibrate_l63.jl calibrate_array.sbatch)
EMU_JID=$(sbatch --parsable -A esm \
          --dependency=afterok:${CALIB_JID} --kill-on-invalid-dep=yes \
          --export=ALL,SCRIPT=emulate_sample_l63.jl emulate_sample_array.sbatch)
PUSHFWD_JID=$(sbatch --parsable -A esm \
              --dependency=afterany:${EMU_JID} \
              pushforward_from_posterior.sbatch)
sbatch -A esm \
       --dependency=afterany:${PUSHFWD_JID} \
       exp_to_leaderboard.sbatch

# L96
CALIB_JID=$(sbatch --parsable -A esm \
            --export=ALL,SCRIPT=calibrate_l96.jl,EXPERIMENT=l96_const \
            calibrate_array.sbatch)
sbatch -A esm \
       --dependency=afterok:${CALIB_JID} --kill-on-invalid-dep=yes \
       --export=ALL,EXPERIMENT=l96_const \
       calibration_diagnostic_plots_l96.sbatch
EMU_JID=$(sbatch --parsable -A esm \
          --dependency=afterok:${CALIB_JID} --kill-on-invalid-dep=yes \
          --export=ALL,SCRIPT=emulate_sample_l96.jl,EXPERIMENT=l96_const \
          emulate_sample_array.sbatch)
PUSHFWD_JID=$(sbatch --parsable -A esm \
              --dependency=afterany:${EMU_JID} \
              --export=ALL,EXPERIMENT=l96_const \
              pushforward_from_posterior.sbatch)
sbatch -A esm \
       --dependency=afterany:${PUSHFWD_JID} \
       --export=ALL,EXPERIMENT=l96_const \
       posterior_diagnostic_plots_l96.sbatch
sbatch -A esm \
       --dependency=afterany:${PUSHFWD_JID} \
       --export=ALL,EXPERIMENT=l96_const \
       exp_to_leaderboard.sbatch
```

### Sbatch files reference

| File | Type | Description |
|------|------|-------------|
| `calibrate_array.sbatch` | array (1–180) | One task per `(N_ens, rng_idx)` cell |
| `emulate_sample_array.sbatch` | array (1–180) | One task per `(N_ens, rng_idx)` cell |
| `pushforward_from_posterior.sbatch` | array (1–180) | Posterior pushforward (forcing + output), one task per cell; saves results into the posterior JLD2 |
| `calibration_diagnostic_plots_l96.sbatch` | single job | Calibration figures, all cells serially (L96) |
| `posterior_diagnostic_plots_l96.sbatch` | array (1–180) | Posterior ribbon/scatter figures, one task per cell (L96); loads pushforward from JLD2 |
| `exp_to_leaderboard.sbatch` | single job | NetCDF leaderboard file, all cells serially; loads pushforward from JLD2 |
| `precompile.sbatch` | single job | `Pkg.instantiate()` + `Pkg.precompile()` |

### Adjusting array size

The sbatch files default to `--array=1-180` (9 ensemble sizes × 20 repeats).
If you change `N_ens_sizes` or `n_repeats` in `experiment_config.jl`, update
the upper bound to `length(N_ens_sizes) * n_repeats` in every array sbatch file,
including `pushforward_from_posterior.sbatch`.  The `%100` suffix caps concurrent
tasks as a cluster-courtesy limit; raise or remove it if you want faster turnaround.

### Smoke test

Before a full submission, run a single-task array to verify the job finds its
input files and writes output correctly:

```bash
sbatch --array=1-1 --export=ALL,SCRIPT=calibrate_l63.jl calibrate_array.sbatch
```
