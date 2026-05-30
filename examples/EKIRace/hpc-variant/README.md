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

The submit scripts (`submit_*.sh`) handle precompilation automatically before
queuing any jobs.

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

One script per case handles precompilation and chains calibrate → emulate_sample
automatically. All four can be launched simultaneously — output files are
case-specific so there are no write conflicts. The optional `EXP_ID` argument
suffixes SLURM job names to keep the queue readable:

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

```bash
# L63
sbatch --export=ALL,SCRIPT=calibrate_l63.jl calibrate_array.sbatch

# L96 (submit once per forcing case)
sbatch --export=ALL,SCRIPT=calibrate_l96.jl,EXPERIMENT=l96_const calibrate_array.sbatch
sbatch --export=ALL,SCRIPT=calibrate_l96.jl,EXPERIMENT=l96_vec   calibrate_array.sbatch
sbatch --export=ALL,SCRIPT=calibrate_l96.jl,EXPERIMENT=l96_flux  calibrate_array.sbatch
```

Each task writes its per-method `ekp` and `results` files, which are consumed
directly by the emulate_sample stage.

### Emulate-sample

Submit after the calibrate array is complete. Use `--dependency` to chain
automatically:

```bash
# L63
cid=$(sbatch --parsable --export=ALL,SCRIPT=calibrate_l63.jl calibrate_array.sbatch)
sbatch --dependency=afterok:$cid \
       --export=ALL,SCRIPT=emulate_sample_l63.jl emulate_sample_array.sbatch

# L96
cid=$(sbatch --parsable --export=ALL,SCRIPT=calibrate_l96.jl,EXPERIMENT=l96_const \
             calibrate_array.sbatch)
sbatch --dependency=afterok:$cid \
       --export=ALL,SCRIPT=emulate_sample_l96.jl,EXPERIMENT=l96_const \
       emulate_sample_array.sbatch
```

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
