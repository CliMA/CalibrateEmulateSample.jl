# EKIRace HPC variant

Parallelised versions of the `calibrate` and `emulate_sample` scripts.
Each script can run serially (identical behaviour to the originals) or as a
SLURM job array where every `(N_ens, rng_idx)` cell is an independent task.

## One-time setup

In `experiment_config.jl`, pin the calibrate date to today's date before
starting a run, e.g.

```julia
calibrate_date = Date("2026-05-29", "yyyy-mm-dd")
```

All subsequent stages (emulate_sample, leaderboard) read this value to locate
the right output directory; keeping it fixed avoids mismatches when jobs run
past midnight or across days.

The submit scripts (`submit_*.sh`) submit a `precompile.sbatch` job first and
chain calibrate → emulate_sample behind it, so nothing runs on the login node.

## Standalone (serial)

Run with no arguments to sweep all `(N_ens, rng_idx)` cells sequentially,
exactly as the original scripts did.

```bash
# L63
julia --project=. calibrate_l63.jl
julia --project=. emulate_sample_l63.jl

# L96 — set EXPERIMENT or edit the toggle in experiment_config.jl
EXPERIMENT=l96_const julia --project=. calibrate_l96.jl
EXPERIMENT=l96_const julia --project=. emulate_sample_l96.jl
```

You can also run a single cell by passing its 1-based task index:

```bash
julia --project=. calibrate_l63.jl 1        # first (N_ens, rng_idx) cell only
julia --project=. emulate_sample_l63.jl 5   # fifth cell only
```

## HPC (Caltech Resnick cluster, SLURM)

### Submission scripts (recommended)

One script per case submits three chained SLURM jobs:
`precompile.sbatch` → `calibrate_array.sbatch` → `emulate_sample_array.sbatch`.
All four cases can be launched simultaneously — output files are case-specific
so there are no write conflicts. The optional `EXP_ID` argument suffixes SLURM
job names to keep the queue readable:

```bash
bash submit_l63.sh        [EXP_ID]
bash submit_l96_const.sh  [EXP_ID]
bash submit_l96_vec.sh    [EXP_ID]
bash submit_l96_flux.sh   [EXP_ID]

# example: run all four with a shared label
for s in submit_l63.sh submit_l96_const.sh submit_l96_vec.sh submit_l96_flux.sh; do
    bash "$s" run1 &
done
wait
```

### Manual submission

Precompile first, then chain calibrate and emulate_sample behind it:

```bash
# precompile (shared across all cases — only one run needed per environment change)
pid=$(sbatch --parsable -A esm precompile.sbatch)

# L63
cid=$(sbatch --parsable -A esm \
             --dependency=afterok:$pid --kill-on-invalid-dep=yes \
             --export=ALL,SCRIPT=calibrate_l63.jl calibrate_array.sbatch)
sbatch -A esm \
       --dependency=afterok:$cid --kill-on-invalid-dep=yes \
       --export=ALL,SCRIPT=emulate_sample_l63.jl emulate_sample_array.sbatch

# L96 (repeat for each forcing case, reusing the same $pid)
cid=$(sbatch --parsable -A esm \
             --dependency=afterok:$pid --kill-on-invalid-dep=yes \
             --export=ALL,SCRIPT=calibrate_l96.jl,EXPERIMENT=l96_const calibrate_array.sbatch)
sbatch -A esm \
       --dependency=afterok:$cid --kill-on-invalid-dep=yes \
       --export=ALL,SCRIPT=emulate_sample_l96.jl,EXPERIMENT=l96_const \
       emulate_sample_array.sbatch
```

Each task writes its per-method `ekp` and `results` files, which are consumed
directly by the emulate_sample stage.

### Adjusting array size

The sbatch files default to `--array=1-60` (3 ensemble sizes × 20 repeats).
If you change `N_ens_sizes` or `n_repeats` in `experiment_config.jl`, update
the upper bound to `length(N_ens_sizes) * n_repeats`. The `%20` suffix caps
concurrent tasks to 20 at a time as a cluster-courtesy limit; raise or remove
it if you want faster turnaround.

### Smoke test

Before a full submission, run a single-task array to verify the job finds its
input files and writes output correctly:

```bash
sbatch --array=1-1 --export=ALL,SCRIPT=calibrate_l63.jl calibrate_array.sbatch
```
