# EKIRace

Serial (local) versions of the calibrate, emulate-sample, and posterior-pushforward
scripts for the Lorenz 63 and Lorenz 96 benchmark experiments.

## Experiments

| Symbol | Model | Forcing type |
|--------|-------|-------------|
| `:l63` | Lorenz 63 | — |
| `:l96_const` | Lorenz 96 | scalar constant |
| `:l96_vec` | Lorenz 96 | spatial vector |
| `:l96_flux` | Lorenz 96 | neural-network flux |

## One-time setup

In `experiment_config.jl`, set `EXPERIMENT` to the case you want to run and pin
`calibrate_date` to today's date before starting a run, e.g.

```julia
EXPERIMENT     = :l96_vec
calibrate_date = Date("2026-05-31", "yyyy-mm-dd")
```

All subsequent stages (emulate_sample, ensemble_from_posterior, leaderboard
utilities) include `experiment_config.jl` and read these values, so keeping them
fixed ensures every stage finds the right output directory.

## Running scripts

Run all scripts from the `examples/EKIRace/` directory using:

```bash
julia --project -e 'include("scriptname.jl")'
```

Each script loops over all `(N_ens, rng_idx)` cells defined in `experiment_config.jl`
and writes output to `output/<method>_<date>/`.

## Stages

### 1. Calibrate

```bash
# L63 (EXPERIMENT setting is ignored)
julia --project -e 'include("calibrate_l63.jl")'

# L96 — reads EXPERIMENT from experiment_config.jl
julia --project -e 'include("calibrate_l96.jl")'
```

The L96 calibration also writes a `l96_computed_preliminaries_<force_case>.jld2`
file to `output/` that the later stages require.

### 2. Emulate and sample

```bash
# L63
julia --project -e 'include("emulate_sample_l63.jl")'

# L96
julia --project -e 'include("emulate_sample_l96.jl")'
```

Reads the calibration output for each `(N_ens, rng_idx)` cell, trains an
emulator, runs MCMC, and writes per-cell posterior `.jld2` files plus
summary netCDF files to the calibration output directory.

### 3. Posterior pushforward (L96 only)

```bash
julia --project -e 'include("ensemble_from_posterior.jl")'
```

Requires `emulate_sample_l96.jl` to have completed. Loops over all valid
`(N_ens, rng_idx)` cells, samples 100 points from each posterior, runs them
through the Lorenz 96 forward map, and writes
`pushforward_from_posterior_*_k<k>_full_ens.{png,pdf}` plots into the
calibration output directory.

### 4. Leaderboard

The leaderboard utility scripts (`l63_exp_to_leaderboard_utilities.jl`,
`l96_exp_to_leaderboard_utilities.jl`) are not standalone — they are included
by downstream analysis.

To compute summary metrics from a netCDF result file, edit the `filename`
variable at the top of `compute_leaderboard_metrics.jl` to point to the
target file, then run:

```bash
julia --project -e 'include("compute_leaderboard_metrics.jl")'
```

This prints per-ensemble-size Mahalanobis and log-posterior scores against
chi-squared reference quantiles.

## Full run example (L96 vector forcing)

```julia
# In experiment_config.jl:
EXPERIMENT     = :l96_vec
calibrate_date = Date("2026-05-31", "yyyy-mm-dd")
```

```bash
julia --project -e 'include("calibrate_l96.jl")'
julia --project -e 'include("emulate_sample_l96.jl")'
julia --project -e 'include("ensemble_from_posterior.jl")'
```

## HPC variant

For parallelised SLURM job-array runs on the Caltech Resnick cluster, see
[`hpc-variant/README.md`](hpc-variant/README.md).
